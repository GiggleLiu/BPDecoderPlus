"""Top-level MPE API for tropical tensor networks."""

from __future__ import annotations

from typing import Dict, Iterable

from .contraction import (
    ContractNode,
    ReduceNode,
    contract_omeco_tree,
    get_omeco_tree,
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
) -> tuple[Dict[int, int], float, Dict[str, int | tuple[int, ...]]]:
    """Return MPE assignment, score, and contraction metadata.

    Uses omeco for optimized contraction order and tropical-gemm for acceleration.
    """
    evidence = evidence or {}
    factors = build_tropical_factors(model, evidence)
    nodes = build_network(factors)

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
    }
    return assignment, score, info
