"""Contraction ordering and binary contraction tree execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch

import omeco

from .network import TensorNode
from .tropical_einsum import tropical_einsum, tropical_reduce_max, Backpointer


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


def get_omeco_tree(nodes: list[TensorNode]) -> dict:
    """Get the optimized contraction tree from omeco.
    
    Handles disconnected components by contracting each component
    separately and combining the results.

    Args:
        nodes: List of tensor nodes to contract.

    Returns:
        The omeco tree as a dictionary with structure:
        - Leaf: {"tensor_index": int}
        - Node: {"args": [...], "eins": {"ixs": [[...], ...], "iy": [...]}}
    """
    if not nodes:
        raise ValueError("Cannot contract empty list of nodes")
    
    ixs = [list(node.vars) for node in nodes]
    sizes = _infer_var_sizes(nodes)
    
    # Find connected components
    components = _find_connected_components(ixs)
    
    if len(components) == 1:
        # Single component - use omeco directly
        method = omeco.GreedyMethod()
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
            
            method = omeco.GreedyMethod()
            tree = omeco.optimize_code(comp_ixs, [], comp_sizes, method)
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
    
    # Combine component trees into a single tree
    # We chain them: ((tree0, tree1), tree2), ...
    def combine_trees(trees):
        if len(trees) == 1:
            return trees[0]
        elif len(trees) == 2:
            # Combine two trees
            # Get output indices for each
            def get_output_vars(tree):
                if "tensor_index" in tree:
                    return list(nodes[tree["tensor_index"]].vars)
                eins = tree.get("eins", {})
                return eins.get("iy", [])
            
            out0 = get_output_vars(trees[0])
            out1 = get_output_vars(trees[1])
            combined_out = list(dict.fromkeys(out0 + out1))
            
            return {
                "args": trees,
                "eins": {"ixs": [out0, out1], "iy": combined_out}
            }
        else:
            # Recursively combine
            mid = len(trees) // 2
            left = combine_trees(trees[:mid])
            right = combine_trees(trees[mid:])
            return combine_trees([left, right])
    
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
