"""Utility helpers for tropical tensor networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch

from .primitives import IndexMap


@dataclass
class Factor:
    """Factor class representing a factor in the factor graph."""

    vars: Tuple[int, ...]
    values: torch.Tensor


@dataclass
class UAIModel:
    """UAI model containing variables, cardinalities, and factors."""

    nvars: int
    cards: List[int]
    factors: List[Factor]


def read_model_file(filepath: str, factor_eltype=torch.float64) -> UAIModel:
    """Parse UAI format model file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return read_model_from_string(content, factor_eltype=factor_eltype)


def read_model_from_string(content: str, factor_eltype=torch.float64) -> UAIModel:
    """Parse UAI model from string."""
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    if len(lines) < 4:
        raise ValueError("Malformed UAI model: expected at least 4 header lines.")
    network_type = lines[0]
    if network_type not in ("MARKOV", "BAYES"):
        raise ValueError(
            f"Unsupported UAI network type: {network_type!r}. Expected 'MARKOV' or 'BAYES'."
        )
    nvars = int(lines[1])
    cards = [int(x) for x in lines[2].split()]
    if len(cards) != nvars:
        raise ValueError(f"Expected {nvars} cardinalities, got {len(cards)}.")
    ntables = int(lines[3])
    if len(lines) < 4 + ntables:
        raise ValueError(
            f"Malformed UAI model: expected {ntables} scope lines, got {len(lines) - 4}."
        )

    scopes: list[list[int]] = []
    for i in range(ntables):
        parts = lines[4 + i].split()
        scope_size = int(parts[0])
        if len(parts) - 1 != scope_size:
            raise ValueError(
                f"Scope size mismatch on line {4 + i}: "
                f"declared {scope_size}, found {len(parts) - 1} variables."
            )
        scope = [int(x) + 1 for x in parts[1:]]
        scopes.append(scope)

    idx = 4 + ntables
    tokens: List[str] = []
    while idx < len(lines):
        tokens.extend(lines[idx].split())
        idx += 1
    cursor = 0

    factors: List[Factor] = []
    for scope in scopes:
        if cursor >= len(tokens):
            raise ValueError("Unexpected end of UAI factor table data.")
        nelements = int(tokens[cursor])
        cursor += 1
        expected_size = 1
        for card in (cards[v - 1] for v in scope):
            expected_size *= card
        if nelements != expected_size:
            raise ValueError(
                f"Factor table size mismatch for scope {scope}: "
                f"expected {expected_size}, got {nelements}."
            )
        if cursor + nelements > len(tokens):
            raise ValueError("Unexpected end of UAI factor table data.")
        values = torch.tensor(
            [float(x) for x in tokens[cursor : cursor + nelements]], dtype=factor_eltype
        )
        cursor += nelements
        shape = tuple([cards[v - 1] for v in scope])
        values = values.reshape(shape)
        factors.append(Factor(tuple(scope), values))

    return UAIModel(nvars, cards, factors)


def read_evidence_file(filepath: str) -> Dict[int, int]:
    """Parse evidence file (.evid format)."""
    if not filepath:
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return {}
    last_line = lines[-1].strip()
    parts = [int(x) for x in last_line.split()]
    nobsvars = parts[0]
    if len(parts) < 1 + 2 * nobsvars:
        raise ValueError(
            f"Malformed evidence line: expected {1 + 2 * nobsvars} entries, got {len(parts)}."
        )
    evidence: Dict[int, int] = {}
    for i in range(nobsvars):
        var_idx = parts[1 + 2 * i] + 1
        var_value = parts[2 + 2 * i]
        evidence[var_idx] = var_value
    return evidence


def apply_evidence_to_factor(factor: Factor, evidence: Dict[int, int]) -> Factor:
    """Clamp a factor with evidence and drop observed variables."""
    if not evidence:
        return factor
    index = []
    new_vars = []
    for var in factor.vars:
        if var in evidence:
            index.append(int(evidence[var]))
        else:
            index.append(slice(None))
            new_vars.append(var)
    values = factor.values[tuple(index)]
    if values.ndim == 0:
        values = values.reshape(())
    return Factor(tuple(new_vars), values)


def build_tropical_factors(model: UAIModel, evidence: Dict[int, int] | None = None) -> list[Factor]:
    """Apply evidence and return factors in original domain (log later)."""
    evidence = evidence or {}
    factors = [apply_evidence_to_factor(factor, evidence) for factor in model.factors]
    return factors


def build_index_map(
    a_vars: Iterable[int], b_vars: Iterable[int], elim_vars: Iterable[int]
) -> IndexMap:
    a_vars = tuple(a_vars)
    b_vars = tuple(b_vars)
    elim_vars = tuple(elim_vars)
    target_vars = tuple(dict.fromkeys(a_vars + b_vars))
    out_vars = tuple(v for v in target_vars if v not in elim_vars)
    return IndexMap(a_vars=a_vars, b_vars=b_vars, out_vars=out_vars, elim_vars=elim_vars)
