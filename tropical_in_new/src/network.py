"""Tensor network construction for tropical MPE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch

from .primitives import safe_log


@dataclass(frozen=True)
class TensorNode:
    """Node representing a tensor factor in log-domain."""

    vars: Tuple[int, ...]
    values: torch.Tensor


def build_network(factors: Iterable) -> list[TensorNode]:
    """Convert factors into tensor nodes (log-domain)."""
    nodes: list[TensorNode] = []
    for factor in factors:
        nodes.append(TensorNode(vars=tuple(factor.vars), values=safe_log(factor.values)))
    return nodes
