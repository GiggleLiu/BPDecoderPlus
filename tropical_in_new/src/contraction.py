"""Contraction ordering and binary contraction tree execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch

try:  # Optional heuristic provider
    import omeco  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    omeco = None

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


def _build_var_graph(nodes: Iterable[TensorNode]) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = {}
    for node in nodes:
        vars = list(node.vars)
        for var in vars:
            graph.setdefault(var, set()).update(v for v in vars if v != var)
    return graph


def _min_fill_order(graph: dict[int, set[int]]) -> list[int]:
    order: list[int] = []
    graph = {k: set(v) for k, v in graph.items()}
    while graph:
        best_var = None
        best_fill = None
        best_degree = None
        for var, neighbors in graph.items():
            fill = 0
            neighbor_list = list(neighbors)
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    if neighbor_list[j] not in graph[neighbor_list[i]]:
                        fill += 1
            degree = len(neighbors)
            if best_fill is None or (fill, degree) < (best_fill, best_degree):
                best_var = var
                best_fill = fill
                best_degree = degree
        if best_var is None:
            break
        neighbors = list(graph[best_var])
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                graph[neighbors[i]].add(neighbors[j])
                graph[neighbors[j]].add(neighbors[i])
        for neighbor in neighbors:
            graph[neighbor].discard(best_var)
        graph.pop(best_var, None)
        order.append(best_var)
    return order


def choose_order(nodes: list[TensorNode], heuristic: str = "min_fill") -> list[int]:
    """Select elimination order over variable indices."""
    if heuristic == "omeco" and omeco is not None:
        if hasattr(omeco, "min_fill_order"):
            return list(omeco.min_fill_order([node.vars for node in nodes]))
    graph = _build_var_graph(nodes)
    if heuristic in ("min_fill", "omeco"):
        return _min_fill_order(graph)
    if heuristic == "min_degree":
        return sorted(graph, key=lambda v: len(graph[v]))
    raise ValueError(f"Unknown heuristic: {heuristic!r}")


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
        for other in bucket[1:]:
            index_map = build_index_map(combined.vars, other.vars, elim_vars=())
            values, _ = einsum_fn(combined.values, other.values, index_map, track_argmax=False)
            combined = ContractNode(
                vars=index_map.out_vars,
                values=values,
                left=combined,
                right=other,
                elim_vars=(),
                backpointer=None,
            )
        if var in combined.vars:
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
