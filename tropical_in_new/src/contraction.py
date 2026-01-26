"""Contraction ordering and binary contraction tree execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Tuple

import torch

import omeco

from .network import TensorNode
from .tropical_einsum import tropical_einsum, tropical_reduce_max, Backpointer


# =============================================================================
# Optimization Configuration
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for contraction order optimization.
    
    Attributes:
        method: Optimization method to use:
            - "greedy": Fast greedy method (default), may have high space complexity
            - "treesa": TreeSA with simulated annealing, slower but can target lower space
            - "treesa_fast": Fast TreeSA with fewer iterations
        sc_target: Target space complexity in log2 scale (e.g., 25 means 2^25 elements).
            Only used with TreeSA methods. Lower values use less memory but may be slower.
        ntrials: Number of trials for TreeSA (default: 1 for fast, 10 for full).
        niters: Number of iterations per trial for TreeSA (default: 50 for fast, 500 for full).
    """
    method: Literal["greedy", "treesa", "treesa_fast"] = "greedy"
    sc_target: Optional[float] = None
    ntrials: int = 1
    niters: int = 50
    
    @classmethod
    def greedy(cls) -> "OptimizationConfig":
        """Create a greedy optimization config (fast, potentially high memory)."""
        return cls(method="greedy")
    
    @classmethod
    def treesa(cls, sc_target: float = 30.0, ntrials: int = 10, niters: int = 500) -> "OptimizationConfig":
        """Create a TreeSA optimization config (slower, memory-constrained).
        
        Args:
            sc_target: Target space complexity in log2 scale.
            ntrials: Number of independent trials.
            niters: Iterations per trial.
        """
        return cls(method="treesa", sc_target=sc_target, ntrials=ntrials, niters=niters)
    
    @classmethod
    def treesa_fast(cls, sc_target: float = 30.0) -> "OptimizationConfig":
        """Create a fast TreeSA config (balance between speed and memory)."""
        return cls(method="treesa_fast", sc_target=sc_target, ntrials=1, niters=50)


@dataclass
class ContractNode:
    vars: Tuple[int, ...]
    values: torch.Tensor
    left: "TreeNode"
    right: "TreeNode"
    elim_vars: Tuple[int, ...]
    backpointer: Backpointer | None


@dataclass
class ReduceNode:
    vars: Tuple[int, ...]
    values: torch.Tensor
    child: "TreeNode"
    elim_vars: Tuple[int, ...]
    backpointer: Backpointer | None


TreeNode = TensorNode | ContractNode | ReduceNode


def estimate_contraction_cost(
    nodes: list[TensorNode],
    config: Optional[OptimizationConfig] = None,
) -> dict:
    """Estimate the time and space complexity of contracting the tensor network.
    
    Args:
        nodes: List of tensor nodes to contract.
        config: Optimization configuration. Defaults to greedy method.
        
    Returns:
        Dictionary with:
        - "tc": Time complexity in log2 scale (log2 of FLOP count)
        - "sc": Space complexity in log2 scale (log2 of max intermediate tensor size)
        - "memory_bytes": Estimated peak memory usage in bytes (assuming float64)
    """
    if not nodes:
        return {"tc": 0, "sc": 0, "memory_bytes": 0}
    
    if config is None:
        config = OptimizationConfig.greedy()
    
    ixs = [list(node.vars) for node in nodes]
    sizes = _infer_var_sizes(nodes)
    
    method = _create_omeco_method(config)
    tree = omeco.optimize_code(ixs, [], sizes, method)
    
    # Use omeco.contraction_complexity to get tc, sc, rwc
    complexity = omeco.contraction_complexity(tree, ixs, sizes)
    tc = complexity.tc  # Time complexity in log2
    sc = complexity.sc  # Space complexity in log2
    
    # Memory estimate: 2^sc elements * 8 bytes per float64
    memory_bytes = int(2 ** sc * 8)
    
    return {
        "tc": tc,
        "sc": sc,
        "memory_bytes": memory_bytes,
    }


def _infer_var_sizes(nodes: Iterable[TensorNode]) -> dict[int, int]:
    sizes: dict[int, int] = {}
    for node in nodes:
        for var, dim in zip(node.vars, node.values.shape):
            if var in sizes and sizes[var] != dim:
                raise ValueError(
                    f"Variable {var} has inconsistent sizes: {sizes[var]} vs {dim}."
                )
            sizes[var] = int(dim)
    return sizes


def _find_connected_components(ixs: list[list[int]]) -> list[list[int]]:
    """Find connected components among factors based on shared variables.
    
    Args:
        ixs: List of variable lists for each factor.
        
    Returns:
        List of lists, where each inner list contains factor indices in one component.
    """
    n = len(ixs)
    if n == 0:
        return []
    
    # Build adjacency based on shared variables
    var_to_factors: dict[int, list[int]] = {}
    for i, vars in enumerate(ixs):
        for v in vars:
            if v not in var_to_factors:
                var_to_factors[v] = []
            var_to_factors[v].append(i)
    
    # Find connected components using union-find
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union factors that share variables
    for factors in var_to_factors.values():
        for i in range(1, len(factors)):
            union(factors[0], factors[i])
    
    # Group by component
    components: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)
    
    return list(components.values())


def _create_omeco_method(config: OptimizationConfig):
    """Create an omeco optimization method from config."""
    if config.method == "greedy":
        return omeco.GreedyMethod()
    elif config.method in ("treesa", "treesa_fast"):
        # TreeSA with optional space complexity target
        sc_target = config.sc_target if config.sc_target is not None else 30.0
        score = omeco.ScoreFunction(sc_target=sc_target)
        return omeco.TreeSA(ntrials=config.ntrials, niters=config.niters, score=score)
    else:
        raise ValueError(f"Unknown optimization method: {config.method}")


def get_omeco_tree(
    nodes: list[TensorNode],
    config: Optional[OptimizationConfig] = None,
) -> dict:
    """Get the optimized contraction tree from omeco.
    
    Handles disconnected components by contracting each component
    separately and combining the results.

    Args:
        nodes: List of tensor nodes to contract.
        config: Optimization configuration. Defaults to greedy method.

    Returns:
        The omeco tree as a dictionary with structure:
        - Leaf: {"tensor_index": int}
        - Node: {"args": [...], "eins": {"ixs": [[...], ...], "iy": [...]}}
    """
    if not nodes:
        raise ValueError("Cannot contract empty list of nodes")
    
    if config is None:
        config = OptimizationConfig.greedy()
    
    ixs = [list(node.vars) for node in nodes]
    sizes = _infer_var_sizes(nodes)
    
    # Find connected components
    components = _find_connected_components(ixs)
    
    method = _create_omeco_method(config)
    
    if len(components) == 1:
        # Single component - use omeco directly
        tree = omeco.optimize_code(ixs, [], sizes, method)
        return tree.to_dict()
    
    # Multiple components - contract each separately and combine
    component_trees = []
    for comp_indices in components:
        if len(comp_indices) == 1:
            # Single factor - just reference it
            component_trees.append({"tensor_index": comp_indices[0]})
        else:
            # Multiple factors - use omeco for this component
            comp_ixs = [ixs[i] for i in comp_indices]
            comp_sizes = {}
            for i in comp_indices:
                for v in ixs[i]:
                    if v in sizes:
                        comp_sizes[v] = sizes[v]
            
            # Create fresh method for each component (some methods have state)
            comp_method = _create_omeco_method(config)
            tree = omeco.optimize_code(comp_ixs, [], comp_sizes, comp_method)
            tree_dict = tree.to_dict()
            
            # Remap tensor indices to original
            def remap_indices(node):
                if "tensor_index" in node:
                    return {"tensor_index": comp_indices[node["tensor_index"]]}
                args = node.get("args", node.get("children", []))
                return {
                    "args": [remap_indices(a) for a in args],
                    "eins": node.get("eins", {})
                }
            
            component_trees.append(remap_indices(tree_dict))
    
    # Combine component trees into a single tree iteratively
    # This avoids recursion depth issues with many disconnected components
    def get_output_vars(tree):
        """Get output variables from a tree node."""
        if "tensor_index" in tree:
            return list(nodes[tree["tensor_index"]].vars)
        eins = tree.get("eins")
        if not isinstance(eins, dict) or "iy" not in eins:
            raise ValueError(
                "Invalid contraction tree node: non-leaf nodes must have an "
                "'eins' mapping with an 'iy' key specifying output variables."
            )
        return list(eins["iy"])
    
    def combine_trees(trees):
        """Combine a list of component trees into a single tree without recursion."""
        trees = list(trees)
        if not trees:
            raise ValueError("combine_trees expects at least one tree")
        if len(trees) == 1:
            return trees[0]

        # Iteratively combine trees in pairs until a single tree remains
        while len(trees) > 1:
            new_trees = []
            i = 0
            while i < len(trees):
                if i + 1 >= len(trees):
                    # Odd tree out, carry to next round unchanged
                    new_trees.append(trees[i])
                    break

                left_tree = trees[i]
                right_tree = trees[i + 1]

                out0 = get_output_vars(left_tree)
                out1 = get_output_vars(right_tree)
                combined_out = list(dict.fromkeys(out0 + out1))

                new_trees.append({
                    "args": [left_tree, right_tree],
                    "eins": {"ixs": [out0, out1], "iy": combined_out},
                })
                i += 2

            trees = new_trees

        return trees[0]
    
    return combine_trees(component_trees)


def contract_omeco_tree(
    tree_dict: dict,
    nodes: list[TensorNode],
    track_argmax: bool = True,
) -> TreeNode:
    """Contract tensors following omeco's optimized tree structure.

    Uses tropical-gemm for accelerated binary contractions when available.

    Args:
        tree_dict: The omeco tree dictionary from get_omeco_tree().
        nodes: List of input tensor nodes.
        track_argmax: Whether to track argmax for MPE backtracing.

    Returns:
        Root TreeNode with contracted result and backpointers.
    """

    def recurse(node: dict) -> TreeNode:
        # Leaf node - return the input tensor
        if "tensor_index" in node:
            return nodes[node["tensor_index"]]

        # Internal node - contract children
        args = node["args"]
        eins = node["eins"]
        out_vars = tuple(eins["iy"])

        # Recursively contract children
        children = [recurse(arg) for arg in args]

        # Use tropical_einsum for the contraction
        tensors = [c.values for c in children]
        child_ixs = [c.vars for c in children]

        values, backpointer = tropical_einsum(
            tensors, list(child_ixs), out_vars, track_argmax=track_argmax
        )

        # Build result node (for binary, use ContractNode)
        if len(children) == 2:
            all_input = set(children[0].vars) | set(children[1].vars)
            elim_vars = tuple(v for v in all_input if v not in out_vars)

            return ContractNode(
                vars=out_vars,
                values=values,
                left=children[0],
                right=children[1],
                elim_vars=elim_vars,
                backpointer=backpointer,
            )
        else:
            # For n-ary, chain as binary
            result = children[0]
            for i, child in enumerate(children[1:], 1):
                is_final = (i == len(children) - 1)
                target_out = out_vars if is_final else tuple(dict.fromkeys(result.vars + child.vars))

                step_tensors = [result.values, child.values]
                step_ixs = [result.vars, child.vars]

                step_values, step_bp = tropical_einsum(
                    step_tensors, list(step_ixs), target_out, track_argmax=track_argmax
                )

                all_input = set(result.vars) | set(child.vars)
                elim_vars = tuple(v for v in all_input if v not in target_out)

                result = ContractNode(
                    vars=target_out,
                    values=step_values,
                    left=result,
                    right=child,
                    elim_vars=elim_vars,
                    backpointer=step_bp,
                )
            return result

    return recurse(tree_dict)


# =============================================================================
# Slicing Support
# =============================================================================

@dataclass
class SlicedContraction:
    """A sliced contraction plan for memory-efficient tensor network contraction.
    
    Slicing reduces memory usage by fixing certain variables to specific values
    and contracting over all possible values in a loop. The results are then
    combined using tropical addition (max).
    
    Attributes:
        base_tree_dict: The base contraction tree from omeco.
        sliced_vars: Variables that have been sliced.
        sliced_sizes: Sizes of each sliced variable.
        num_slices: Total number of slice combinations (product of sliced_sizes).
        original_nodes: Original tensor nodes before slicing.
    """
    base_tree_dict: dict
    sliced_vars: Tuple[int, ...]
    sliced_sizes: Tuple[int, ...]
    num_slices: int
    original_nodes: list


def get_sliced_contraction(
    nodes: list[TensorNode],
    sc_target: float = 25.0,
    config: Optional[OptimizationConfig] = None,
) -> SlicedContraction:
    """Create a sliced contraction plan that fits within memory constraints.
    
    Uses omeco's slice_code() to determine which variables to slice to achieve
    the target space complexity.
    
    Args:
        nodes: List of tensor nodes to contract.
        sc_target: Target space complexity in log2 scale after slicing.
        config: Optimization configuration for the base tree. Defaults to greedy.
        
    Returns:
        SlicedContraction plan ready for execution.
    """
    if not nodes:
        raise ValueError("Cannot slice empty list of nodes")
    
    if config is None:
        config = OptimizationConfig.greedy()
    
    ixs = [list(node.vars) for node in nodes]
    sizes = _infer_var_sizes(nodes)
    
    # Get base tree
    method = _create_omeco_method(config)
    tree = omeco.optimize_code(ixs, [], sizes, method)
    
    # Check if slicing is needed
    complexity = omeco.contraction_complexity(tree, ixs, sizes)
    current_sc = complexity.sc
    
    if current_sc <= sc_target:
        # No slicing needed
        return SlicedContraction(
            base_tree_dict=tree.to_dict(),
            sliced_vars=(),
            sliced_sizes=(),
            num_slices=1,
            original_nodes=nodes,
        )
    
    # Use omeco to find slicing with TreeSASlicer
    score = omeco.ScoreFunction(sc_target=sc_target)
    slicer = omeco.TreeSASlicer.fast(score=score)
    sliced_einsum = omeco.slice_code(tree, ixs, sizes, slicer)
    
    # Get sliced indices
    sliced_indices = sliced_einsum.slicing()
    
    # Get sizes of sliced variables
    sliced_sizes = tuple(sizes.get(idx, 2) for idx in sliced_indices)
    num_slices = 1
    for s in sliced_sizes:
        num_slices *= s
    
    # Note: For sliced contraction, we'll need to rebuild the tree for each slice
    # since the original tree structure doesn't account for slicing
    return SlicedContraction(
        base_tree_dict=tree.to_dict(),  # Keep original tree structure
        sliced_vars=tuple(sliced_indices),
        sliced_sizes=sliced_sizes,
        num_slices=num_slices,
        original_nodes=nodes,
    )


def _slice_nodes(
    nodes: list[TensorNode],
    sliced_vars: Tuple[int, ...],
    slice_values: Tuple[int, ...],
) -> list[TensorNode]:
    """Create sliced versions of tensor nodes by fixing sliced variables.
    
    Args:
        nodes: Original tensor nodes.
        sliced_vars: Variables to slice.
        slice_values: Values to fix each sliced variable to.
        
    Returns:
        New tensor nodes with sliced variables fixed.
    """
    slice_map = dict(zip(sliced_vars, slice_values))
    sliced_nodes = []
    
    for node in nodes:
        # Check which sliced vars are in this node
        indices_to_fix = []
        for i, v in enumerate(node.vars):
            if v in slice_map:
                indices_to_fix.append((i, v, slice_map[v]))
        
        if not indices_to_fix:
            # No sliced variables in this node
            sliced_nodes.append(node)
            continue
        
        # Fix the sliced variables by indexing
        values = node.values
        new_vars = list(node.vars)
        
        # Process in reverse order to maintain correct indices
        for i, v, val in sorted(indices_to_fix, reverse=True):
            # Index into the tensor to fix this variable
            slices = [slice(None)] * values.ndim
            slices[i] = val
            values = values[tuple(slices)]
            new_vars.pop(i)
        
        sliced_nodes.append(TensorNode(vars=tuple(new_vars), values=values))
    
    return sliced_nodes


def _contract_connected_component(
    nodes: list[TensorNode],
    indices: list[int],
    ixs: list[list[int]],
    sizes: dict[int, int],
    track_argmax: bool = True,
) -> TreeNode:
    """Contract a single connected component of the tensor network.
    
    Args:
        nodes: All tensor nodes.
        indices: Indices of nodes in this component.
        ixs: Variable lists for all nodes.
        sizes: Variable size dict.
        track_argmax: Whether to track argmax.
        
    Returns:
        Root TreeNode with contracted result.
    """
    if len(indices) == 1:
        # Single node - reduce all its variables
        node = nodes[indices[0]]
        if not node.vars:
            return node
        values, backpointer = tropical_reduce_max(
            node.values, node.vars, tuple(node.vars), track_argmax=track_argmax
        )
        return ReduceNode(
            vars=(),
            values=values,
            child=node,
            elim_vars=tuple(node.vars),
            backpointer=backpointer,
        )
    
    # Multiple nodes - use omeco for this component
    comp_nodes = [nodes[i] for i in indices]
    comp_ixs = [ixs[i] for i in indices]
    comp_sizes = {}
    for i in indices:
        for v in ixs[i]:
            if v in sizes:
                comp_sizes[v] = sizes[v]
    
    # Optimize contraction for this component
    import omeco
    tree = omeco.optimize_code(comp_ixs, [], comp_sizes, omeco.GreedyMethod())
    tree_dict = tree.to_dict()
    
    # Remap tensor indices to component-local indices
    index_map = {orig: local for local, orig in enumerate(indices)}
    
    def remap_indices(node):
        if "tensor_index" in node:
            return {"tensor_index": node["tensor_index"]}  # Already local
        args = node.get("args", node.get("children", []))
        return {
            "args": [remap_indices(a) for a in args],
            "eins": node.get("eins", {})
        }
    
    tree_dict = remap_indices(tree_dict)
    
    # Contract this component
    root = contract_omeco_tree(tree_dict, comp_nodes, track_argmax=track_argmax)
    
    # Reduce any remaining variables to scalar
    if root.vars:
        values, backpointer = tropical_reduce_max(
            root.values, root.vars, tuple(root.vars), track_argmax=track_argmax
        )
        root = ReduceNode(
            vars=(),
            values=values,
            child=root,
            elim_vars=tuple(root.vars),
            backpointer=backpointer,
        )
    
    return root


def _contract_with_components(
    nodes: list[TensorNode],
    track_argmax: bool = True,
) -> Tuple[TreeNode, list]:
    """Contract tensor network handling disconnected components separately.
    
    For tropical tensor networks, disconnected components can be solved
    independently and their scalar results summed (in log space).
    
    Args:
        nodes: List of tensor nodes.
        track_argmax: Whether to track argmax for MPE.
        
    Returns:
        Tuple of (combined root node, list of component roots).
    """
    ixs = [list(node.vars) for node in nodes]
    sizes = _infer_var_sizes(nodes)
    components = _find_connected_components(ixs)
    
    if len(components) == 1:
        # Single component - use standard contraction
        tree_dict = get_omeco_tree(nodes)
        root = contract_omeco_tree(tree_dict, nodes, track_argmax=track_argmax)
        if root.vars:
            values, backpointer = tropical_reduce_max(
                root.values, root.vars, tuple(root.vars), track_argmax=track_argmax
            )
            root = ReduceNode(
                vars=(),
                values=values,
                child=root,
                elim_vars=tuple(root.vars),
                backpointer=backpointer,
            )
        return root, [root]
    
    # Multiple components - contract each separately
    component_roots = []
    total_score = 0.0
    
    for comp_indices in components:
        comp_root = _contract_connected_component(
            nodes, comp_indices, ixs, sizes, track_argmax=track_argmax
        )
        component_roots.append(comp_root)
        total_score += float(comp_root.values.item())
    
    # Create a combined root with the sum of scores
    # Note: In tropical semiring, combining independent components means
    # summing their log-probabilities (= multiplying probabilities)
    import torch
    combined_values = torch.tensor(total_score, dtype=component_roots[0].values.dtype)
    
    # We use a ReduceNode to represent the combination
    # The first component root serves as the "child" for backtracing
    combined_root = ReduceNode(
        vars=(),
        values=combined_values,
        child=component_roots[0],
        elim_vars=(),
        backpointer=None,
    )
    
    return combined_root, component_roots


def contract_sliced_tree(
    sliced: SlicedContraction,
    track_argmax: bool = True,
) -> Tuple[TreeNode, Optional[dict]]:
    """Contract a sliced tensor network.
    
    Iterates over all slice combinations, contracts each, and combines
    results using tropical addition (max). For MPE, tracks which slice
    produced the maximum value.
    
    Handles disconnected components (created by slicing) by contracting
    each component separately and summing their scalar results.
    
    Args:
        sliced: SlicedContraction plan from get_sliced_contraction().
        track_argmax: Whether to track argmax for MPE backtracing.
        
    Returns:
        Tuple of (root TreeNode, slice_info dict).
        slice_info contains:
        - "best_slice_values": The slice values that produced the max result
        - "best_slice_root": The root node from the best slice (for backtracing)
        - "component_roots": List of component roots (for multi-component backtracing)
    """
    if sliced.num_slices == 1:
        # No actual slicing, just contract normally
        root, comp_roots = _contract_with_components(
            sliced.original_nodes, track_argmax=track_argmax
        )
        return root, {
            "best_slice_values": (),
            "best_slice_root": root,
            "component_roots": comp_roots,
        }
    
    best_value = None
    best_slice_values = None
    best_root = None
    best_comp_roots = None
    
    # Iterate over all slice combinations
    for slice_idx in range(sliced.num_slices):
        # Convert flat index to slice values
        slice_values = []
        remaining = slice_idx
        for size in reversed(sliced.sliced_sizes):
            slice_values.append(remaining % size)
            remaining //= size
        slice_values = tuple(reversed(slice_values))
        
        # Create sliced nodes
        sliced_nodes = _slice_nodes(
            sliced.original_nodes,
            sliced.sliced_vars,
            slice_values,
        )
        
        # Contract this slice using component-aware contraction
        root, comp_roots = _contract_with_components(
            sliced_nodes, track_argmax=track_argmax
        )
        
        # Get the scalar value
        current_value = float(root.values.item())
        
        # Track the best (max) result
        if best_value is None or current_value > best_value:
            best_value = current_value
            best_slice_values = slice_values
            best_root = root
            best_comp_roots = comp_roots
    
    slice_info = {
        "best_slice_values": best_slice_values,
        "best_slice_root": best_root,
        "sliced_vars": sliced.sliced_vars,
        "component_roots": best_comp_roots,
    }
    
    return best_root, slice_info


# =============================================================================
# Legacy API for backward compatibility
# =============================================================================

@dataclass(frozen=True)
class ContractionTree:
    """Legacy contraction tree structure."""
    order: Tuple[int, ...]
    nodes: Tuple[TensorNode, ...]


def choose_order(nodes: list[TensorNode], heuristic: str = "omeco") -> list[int]:
    """Legacy: Select elimination order. Use get_omeco_tree() instead."""
    if heuristic != "omeco":
        raise ValueError("Only the 'omeco' heuristic is supported.")
    tree_dict = get_omeco_tree(nodes)
    ixs = [list(node.vars) for node in nodes]
    return _elim_order_from_tree_dict(tree_dict, ixs)


def _elim_order_from_tree_dict(tree_dict: dict, ixs: list[list[int]]) -> list[int]:
    """Extract elimination order from omeco tree (legacy support)."""
    total_counts: dict[int, int] = {}
    for vars in ixs:
        for var in vars:
            total_counts[var] = total_counts.get(var, 0) + 1

    eliminated: set[int] = set()

    def visit(node: dict) -> tuple[dict[int, int], list[int]]:
        if "tensor_index" in node:
            counts: dict[int, int] = {}
            for var in ixs[node["tensor_index"]]:
                counts[var] = counts.get(var, 0) + 1
            return counts, []

        children = node.get("args") or node.get("children", [])
        if not isinstance(children, list) or not children:
            return {}, []

        counts: dict[int, int] = {}
        order: list[int] = []
        for child in children:
            child_counts, child_order = visit(child)
            order.extend(child_order)
            for var, count in child_counts.items():
                counts[var] = counts.get(var, 0) + count

        newly_eliminated = [
            var
            for var, count in counts.items()
            if count == total_counts.get(var, 0) and var not in eliminated
        ]
        for var in sorted(newly_eliminated):
            eliminated.add(var)
            order.append(var)
        return counts, order

    _, order = visit(tree_dict)
    remaining = sorted([var for var in total_counts if var not in eliminated])
    return order + remaining


def build_contraction_tree(order: Iterable[int], nodes: list[TensorNode]) -> ContractionTree:
    """Legacy: Prepare a contraction plan from order and leaf nodes."""
    return ContractionTree(order=tuple(order), nodes=tuple(nodes))


def contract_tree(
    tree: ContractionTree,
    einsum_fn=None,
    track_argmax: bool = True,
) -> TreeNode:
    """Legacy: Contract using elimination order. Use contract_omeco_tree() instead."""
    active_nodes: list[TreeNode] = list(tree.nodes)

    for var in tree.order:
        bucket = [node for node in active_nodes if var in node.vars]
        if not bucket:
            continue
        bucket_ids = {id(node) for node in bucket}
        active_nodes = [node for node in active_nodes if id(node) not in bucket_ids]

        combined: TreeNode = bucket[0]
        for i, other in enumerate(bucket[1:]):
            is_last = i == len(bucket) - 2
            elim_vars = (var,) if is_last else ()

            # Use tropical_einsum
            target_out = tuple(v for v in dict.fromkeys(combined.vars + other.vars) if v not in elim_vars)
            values, backpointer = tropical_einsum(
                [combined.values, other.values],
                [combined.vars, other.vars],
                target_out,
                track_argmax=track_argmax if is_last else False,
            )

            combined = ContractNode(
                vars=target_out,
                values=values,
                left=combined,
                right=other,
                elim_vars=elim_vars,
                backpointer=backpointer,
            )

        if var in combined.vars:
            # Single-node bucket: eliminate via reduce
            values, backpointer = tropical_reduce_max(
                combined.values, combined.vars, (var,), track_argmax=track_argmax
            )
            combined = ReduceNode(
                vars=tuple(v for v in combined.vars if v != var),
                values=values,
                child=combined,
                elim_vars=(var,),
                backpointer=backpointer,
            )
        active_nodes.append(combined)

    while len(active_nodes) > 1:
        left = active_nodes.pop(0)
        right = active_nodes.pop(0)
        target_out = tuple(dict.fromkeys(left.vars + right.vars))
        values, _ = tropical_einsum(
            [left.values, right.values],
            [left.vars, right.vars],
            target_out,
            track_argmax=False,
        )
        combined = ContractNode(
            vars=target_out,
            values=values,
            left=left,
            right=right,
            elim_vars=(),
            backpointer=None,
        )
        active_nodes.append(combined)

    if not active_nodes:
        raise ValueError("Contraction produced no nodes.")
    return active_nodes[0]
