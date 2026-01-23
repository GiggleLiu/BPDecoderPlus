"""Tropical semiring primitives and backpointer helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch

try:  # Optional accelerator; falls back to pure torch.
    import tropical_gemm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tropical_gemm = None


@dataclass(frozen=True)
class IndexMap:
    """Index mapping for binary tropical contractions."""

    a_vars: Tuple[int, ...]
    b_vars: Tuple[int, ...]
    out_vars: Tuple[int, ...]
    elim_vars: Tuple[int, ...]


@dataclass
class Backpointer:
    """Stores argmax metadata for eliminated variables."""

    elim_vars: Tuple[int, ...]
    elim_shape: Tuple[int, ...]
    out_vars: Tuple[int, ...]
    argmax_flat: torch.Tensor


@dataclass(frozen=True)
class TropicalTensor:
    """Lightweight wrapper for tropical (max-plus) tensors."""

    vars: Tuple[int, ...]
    values: torch.Tensor

    def __add__(self, other: "TropicalTensor") -> "TropicalTensor":
        if self.vars != other.vars:
            raise ValueError("TropicalTensor.__add__ requires identical variable order.")
        return TropicalTensor(self.vars, torch.maximum(self.values, other.values))


def safe_log(tensor: torch.Tensor) -> torch.Tensor:
    """Convert potentials to log domain; zeros map to -inf."""
    neg_inf = torch.tensor(float("-inf"), dtype=tensor.dtype, device=tensor.device)
    return torch.where(tensor > 0, torch.log(tensor), neg_inf)


def _align_tensor(
    tensor: torch.Tensor, tensor_vars: Tuple[int, ...], target_vars: Tuple[int, ...]
) -> torch.Tensor:
    if not target_vars:
        return tensor.reshape(())
    if not tensor_vars:
        return tensor.reshape((1,) * len(target_vars))
    present = [v for v in target_vars if v in tensor_vars]
    perm = [tensor_vars.index(v) for v in present]
    aligned = tensor if perm == list(range(len(tensor_vars))) else tensor.permute(perm)
    shape = []
    p = 0
    for var in target_vars:
        if var in tensor_vars:
            shape.append(aligned.shape[p])
            p += 1
        else:
            shape.append(1)
    return aligned.reshape(tuple(shape))


def tropical_reduce_max(
    tensor: torch.Tensor,
    vars: Tuple[int, ...],
    elim_vars: Iterable[int],
    track_argmax: bool = True,
) -> tuple[torch.Tensor, Backpointer | None]:
    elim_vars = tuple(elim_vars)
    if not elim_vars:
        return tensor, None
    target_vars = tuple(vars)
    elim_axes = [target_vars.index(v) for v in elim_vars]
    keep_axes = [i for i in range(len(target_vars)) if i not in elim_axes]
    perm = keep_axes + elim_axes
    permuted = tensor if perm == list(range(len(target_vars))) else tensor.permute(perm)
    out_shape = [tensor.shape[i] for i in keep_axes]
    elim_shape = [tensor.shape[i] for i in elim_axes]
    flat = permuted.reshape(*out_shape, -1)
    values, argmax_flat = torch.max(flat, dim=-1)
    if not track_argmax:
        return values, None
    backpointer = Backpointer(
        elim_vars=elim_vars,
        elim_shape=tuple(elim_shape),
        out_vars=tuple(target_vars[i] for i in keep_axes),
        argmax_flat=argmax_flat,
    )
    return values, backpointer


def tropical_einsum(
    a: torch.Tensor,
    b: torch.Tensor,
    index_map: IndexMap,
    track_argmax: bool = True,
) -> tuple[torch.Tensor, Backpointer | None]:
    """Binary tropical contraction: add in log-space, max over elim_vars."""
    if tropical_gemm is not None and hasattr(tropical_gemm, "einsum"):
        try:  # pragma: no cover - optional dependency
            return tropical_gemm.einsum(a, b, index_map, track_argmax=track_argmax)
        except Exception:
            pass

    target_vars = tuple(dict.fromkeys(index_map.a_vars + index_map.b_vars))
    expected_out = tuple(v for v in target_vars if v not in index_map.elim_vars)
    if index_map.out_vars and index_map.out_vars != expected_out:
        raise ValueError("index_map.out_vars does not match contraction result ordering.")

    aligned_a = _align_tensor(a, index_map.a_vars, target_vars)
    aligned_b = _align_tensor(b, index_map.b_vars, target_vars)
    combined = aligned_a + aligned_b

    if not index_map.elim_vars:
        return combined, None

    values, backpointer = tropical_reduce_max(
        combined, target_vars, index_map.elim_vars, track_argmax=track_argmax
    )
    return values, backpointer


def argmax_trace(backpointer: Backpointer, assignment: Dict[int, int]) -> Dict[int, int]:
    """Decode eliminated variable assignments from a backpointer."""
    if not backpointer.elim_vars:
        return {}
    if backpointer.out_vars:
        idx = tuple(assignment[v] for v in backpointer.out_vars)
        flat = int(backpointer.argmax_flat[idx].item())
    else:
        flat = int(backpointer.argmax_flat.item())
    values = []
    for size in reversed(backpointer.elim_shape):
        values.append(flat % size)
        flat //= size
    values = list(reversed(values))
    return {var: int(val) for var, val in zip(backpointer.elim_vars, values)}
