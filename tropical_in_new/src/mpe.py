"""Top-level MPE API for tropical tensor networks.

This module provides the main entry points for MPE (Most Probable Explanation)
inference using tropical tensor networks.

Key functions:
- mpe_tropical: Exact MPE using optimized contraction (requires low tree width)
- mpe_tropical_approximate: Approximate MPE using MPS/sweep contraction (scalable)
"""

from __future__ import annotations

from typing import Dict, Iterable, Literal, Optional

from .contraction import (
    ContractNode,
    ReduceNode,
    contract_omeco_tree,
    get_omeco_tree,
    estimate_contraction_cost,
)
from .network import TensorNode, build_network
from .tropical_einsum import argmax_trace, tropical_reduce_max
from .utils import UAIModel, build_tropical_factors


def _unravel_argmax(values, vars: Iterable[int]) -> Dict[int, int]:
    vars = tuple(vars)
    if not vars:
        return {}
    flat = int(values.reshape(-1).argmax().item())
    shape = list(values.shape)
    assignments = []
    for size in reversed(shape):
        assignments.append(flat % size)
        flat //= size
    assignments = list(reversed(assignments))
    return {var: int(val) for var, val in zip(vars, assignments)}


def recover_mpe_assignment(root) -> Dict[int, int]:
    """Recover MPE assignment from a contraction tree with backpointers."""
    assignment: Dict[int, int] = {}

    def require_vars(required: Iterable[int], available: Dict[int, int]) -> None:
        missing = [v for v in required if v not in available]
        if missing:
            raise KeyError(
                "Missing assignment values for variables: "
                f"{missing}. Provided assignment keys: {sorted(available.keys())}"
            )

    def traverse(node, out_assignment: Dict[int, int]) -> None:
        assignment.update(out_assignment)
        if isinstance(node, TensorNode):
            return
        if isinstance(node, ReduceNode):
            elim_assignment = (
                argmax_trace(node.backpointer, out_assignment) if node.backpointer else {}
            )
            combined = {**out_assignment, **elim_assignment}
            require_vars(node.child.vars, combined)
            child_assignment = {v: combined[v] for v in node.child.vars}
            traverse(node.child, child_assignment)
            return
        if isinstance(node, ContractNode):
            elim_assignment = (
                argmax_trace(node.backpointer, out_assignment) if node.backpointer else {}
            )
            combined = {**out_assignment, **elim_assignment}
            require_vars(node.left.vars, combined)
            left_assignment = {v: combined[v] for v in node.left.vars}
            require_vars(node.right.vars, combined)
            right_assignment = {v: combined[v] for v in node.right.vars}
            traverse(node.left, left_assignment)
            traverse(node.right, right_assignment)

    initial = _unravel_argmax(root.values, root.vars)
    traverse(root, initial)
    return assignment


def mpe_tropical(
    model: UAIModel,
    evidence: Dict[int, int] | None = None,
    method: Literal["exact", "mps", "sweep", "auto"] = "exact",
    chi: Optional[int] = None,
) -> tuple[Dict[int, int], float, Dict[str, int | tuple[int, ...]]]:
    """Return MPE assignment, score, and contraction metadata.

    Uses omeco for optimized contraction order and tropical-gemm for acceleration.
    
    Args:
        model: UAI model to solve
        evidence: Dictionary of observed variable assignments
        method: Contraction method:
            - "exact": Exact contraction using omeco (default, requires low tree width)
            - "mps": Approximate MPS boundary contraction (scalable)
            - "sweep": Approximate sweep contraction (scalable)
            - "auto": Automatically choose based on estimated complexity
        chi: Bond dimension for approximate methods (default: auto-select)
        
    Returns:
        Tuple of (assignment, score, info) where:
        - assignment: Dict mapping variable indices to optimal values
        - score: Log-probability of the MPE assignment
        - info: Metadata about the computation
    """
    evidence = evidence or {}
    factors = build_tropical_factors(model, evidence)
    nodes = build_network(factors)

    # Auto-select method based on complexity
    if method == "auto":
        cost = estimate_contraction_cost(nodes)
        sc = cost.get("sc", 0)
        # If space complexity > 30 (2^30 ~ 1GB), use approximate method
        if sc > 30:
            method = "mps"
        else:
            method = "exact"
    
    # Use approximate methods if requested
    if method in ("mps", "sweep"):
        return mpe_tropical_approximate(
            model, evidence=evidence, method=method, chi=chi
        )

    # Exact contraction
    # Get optimized contraction tree from omeco
    tree_dict = get_omeco_tree(nodes)

    # Contract using the optimized tree
    root = contract_omeco_tree(tree_dict, nodes, track_argmax=True)

    # Final reduction if there are remaining variables
    if root.vars:
        values, backpointer = tropical_reduce_max(
            root.values, root.vars, tuple(root.vars), track_argmax=True
        )
        root = ReduceNode(
            vars=(),
            values=values,
            child=root,
            elim_vars=tuple(root.vars),
            backpointer=backpointer,
        )

    assignment = recover_mpe_assignment(root)
    assignment.update({int(k): int(v) for k, v in evidence.items()})
    score = float(root.values.item())
    info = {
        "num_nodes": len(nodes),
        "method": "exact",
    }
    return assignment, score, info


def mpe_tropical_approximate(
    model: UAIModel,
    evidence: Dict[int, int] | None = None,
    method: Literal["mps", "sweep"] = "mps",
    chi: Optional[int] = None,
    refine: bool = True,
    refine_method: Literal["local_search", "coordinate_descent", "simulated_annealing"] = "coordinate_descent",
    syndrome_projection: bool = False,
    H: Optional["np.ndarray"] = None,
    syndrome: Optional["np.ndarray"] = None,
    priors: Optional["np.ndarray"] = None,
) -> tuple[Dict[int, int], float, Dict[str, int | tuple[int, ...]]]:
    """Approximate MPE using MPS-based contraction methods.
    
    This function enables MPE inference for large tensor networks where
    exact contraction is infeasible due to high tree width. It trades
    exactness for scalability by limiting the bond dimension.
    
    Args:
        model: UAI model to solve
        evidence: Dictionary of observed variable assignments
        method: Approximate method to use:
            - "mps": MPS boundary contraction (Bravyi et al. style)
            - "sweep": Sweep line contraction (Chubb style)
        chi: Maximum bond dimension. Higher values give better accuracy
             but use more memory. If None, auto-selects based on problem size.
        refine: Whether to refine the approximate assignment using local search.
        refine_method: Refinement method to use:
            - "local_search": Greedy single-variable flipping
            - "coordinate_descent": Optimize each variable given others fixed
            - "simulated_annealing": Stochastic optimization with temperature schedule
        syndrome_projection: Whether to project assignment to satisfy syndrome constraint
        H: Parity check matrix (required if syndrome_projection=True)
        syndrome: Target syndrome (required if syndrome_projection=True)
        priors: Prior error probabilities (required if syndrome_projection=True)
             
    Returns:
        Tuple of (assignment, score, info) where:
        - assignment: Dict mapping variable indices to optimal values
        - score: Log-probability of the (approximate) MPE assignment
        - info: Metadata including chi_used, method, etc.
        
    Example:
        >>> model = read_model_file("large_surface_code.uai")
        >>> assignment, score, info = mpe_tropical_approximate(model, chi=32)
        >>> print(f"Bond dimension used: {info['chi_used']}")
    """
    from .approximate import (
        boundary_contract,
        BoundaryContractionResult,
        refine_assignment_local_search,
        refine_assignment_coordinate_descent,
        refine_assignment_simulated_annealing,
        project_to_syndrome,
    )
    from .sweep import sweep_contract, multi_direction_sweep, estimate_required_chi
    
    evidence = evidence or {}
    factors = build_tropical_factors(model, evidence)
    nodes = build_network(factors)
    
    # Auto-select chi if not specified
    if chi is None:
        chi = estimate_required_chi(nodes)
        chi = max(16, min(chi, 64))  # Clamp to reasonable range
    
    # Prepare tensors and variables for contraction
    tensors = [node.values for node in nodes]
    vars_list = [node.vars for node in nodes]
    
    if method == "mps":
        # MPS boundary contraction
        result = boundary_contract(
            tensors=tensors,
            vars_list=vars_list,
            chi=chi,
            track_assignment=True,
        )
        score = result.value
        chi_used = result.chi_used
        assignment = result.backpointer.get_best_assignment()
        
    elif method == "sweep":
        # Sweep contraction (tries multiple directions)
        result = multi_direction_sweep(
            nodes=nodes,
            chi=chi,
        )
        score = result.value
        chi_used = result.chi_used
        assignment = result.assignment
        
    else:
        raise ValueError(f"Unknown approximate method: {method}")
    
    # Ensure all variables have an assignment (fill missing with 0)
    all_vars = set()
    for node in nodes:
        all_vars.update(node.vars)
    for var in all_vars:
        if var not in assignment:
            assignment[var] = 0
    
    # Refine assignment using specified method
    refined = False
    if refine and assignment:
        if refine_method == "local_search":
            assignment, score = refine_assignment_local_search(
                assignment, nodes, max_iterations=100
            )
        elif refine_method == "coordinate_descent":
            assignment, score = refine_assignment_coordinate_descent(
                assignment, nodes, max_sweeps=10
            )
        elif refine_method == "simulated_annealing":
            assignment, score = refine_assignment_simulated_annealing(
                assignment, nodes, max_iterations=1000
            )
        refined = True
    
    # Project to valid syndrome space if requested
    projected = False
    if syndrome_projection and H is not None and syndrome is not None and priors is not None:
        assignment = project_to_syndrome(assignment, H, syndrome, priors)
        # Recompute score after projection
        score = 0.0
        for node in nodes:
            indices = []
            for var in node.vars:
                indices.append(assignment.get(var, 0))
            if indices:
                try:
                    score += node.values[tuple(indices)].item()
                except (IndexError, RuntimeError):
                    score += -1e10
        projected = True
    
    # Add evidence back to assignment
    assignment.update({int(k): int(v) for k, v in evidence.items()})
    
    # Build info dict
    info = {
        "num_nodes": len(nodes),
        "method": method,
        "chi": chi,
        "chi_used": chi_used,
        "approximate": True,
        "refined": refined,
        "projected": projected,
    }
    
    return assignment, score, info
