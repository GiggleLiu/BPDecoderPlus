"""Contraction ordering and binary contraction tree execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch

import omeco

from .network import TensorNode
from .primitives import Backpointer, tropical_reduce_max
from .utils import build_index_map


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


@dataclass(frozen=True)
class ContractionTree:
    order: Tuple[int, ...]
    nodes: Tuple[TensorNode, ...]


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


def _extract_leaf_index(node_dict: dict) -> int | None:
    for key in ("leaf", "leaf_index", "index", "tensor"):
        if key in node_dict:
            value = node_dict[key]
            if isinstance(value, int):
                return value
    return None


def _elim_order_from_tree_dict(tree_dict: dict, ixs: list[list[int]]) -> list[int]:
    total_counts: dict[int, int] = {}
    for vars in ixs:
        for var in vars:
            total_counts[var] = total_counts.get(var, 0) + 1

    eliminated: set[int] = set()

    def visit(node: dict) -> tuple[dict[int, int], list[int]]:
        leaf_index = _extract_leaf_index(node)
        if leaf_index is not None:
            counts: dict[int, int] = {}
            for var in ixs[leaf_index]:
                counts[var] = counts.get(var, 0) + 1
            return counts, []

        children = node.get("children", [])
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


def choose_order(nodes: list[TensorNode], heuristic: str = "omeco") -> list[int]:
    """Select elimination order over variable indices using omeco."""
    if heuristic != "omeco":
        raise ValueError("Only the 'omeco' heuristic is supported.")
    ixs = [list(node.vars) for node in nodes]
    sizes = _infer_var_sizes(nodes)
    method = omeco.GreedyMethod() if hasattr(omeco, "GreedyMethod") else None
    tree = (
        omeco.optimize_code(ixs, [], sizes, method)
        if method is not None
        else omeco.optimize_code(ixs, [], sizes)
    )
    tree_dict = tree.to_dict() if hasattr(tree, "to_dict") else tree
    if not isinstance(tree_dict, dict):
        raise ValueError("omeco.optimize_code did not return a usable tree.")
    return _elim_order_from_tree_dict(tree_dict, ixs)


def build_contraction_tree(order: Iterable[int], nodes: list[TensorNode]) -> ContractionTree:
    """Prepare a contraction plan from order and leaf nodes."""
    return ContractionTree(order=tuple(order), nodes=tuple(nodes))


def contract_tree(
    tree: ContractionTree,
    einsum_fn,
    track_argmax: bool = True,
) -> TreeNode:
    """Contract along the tree using the tropical einsum."""
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
            index_map = build_index_map(combined.vars, other.vars, elim_vars=elim_vars)
            values, backpointer = einsum_fn(
                combined.values, other.values, index_map,
                track_argmax=track_argmax if is_last else False,
            )
            combined = ContractNode(
                vars=index_map.out_vars,
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
        index_map = build_index_map(left.vars, right.vars, elim_vars=())
        values, _ = einsum_fn(left.values, right.values, index_map, track_argmax=False)
        combined = ContractNode(
            vars=index_map.out_vars,
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
