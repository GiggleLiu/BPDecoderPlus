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
    missing_elim_vars = [v for v in elim_vars if v not in target_vars]
    if missing_elim_vars:
        raise ValueError(
            "tropical_reduce_max: elim_vars "
            f"{missing_elim_vars} are not present in vars {target_vars}."
        )
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


def tropical_contract_binary(
    a: torch.Tensor,
    b: torch.Tensor,
    a_vars: Tuple[int, ...],
    b_vars: Tuple[int, ...],
    out_vars: Tuple[int, ...],
    track_argmax: bool = True,
) -> tuple[torch.Tensor, Backpointer | None]:
    """Binary tropical contraction using tropical-gemm for acceleration.

    Uses matrix multiplication formulation:
    - Non-contracted vars of A become M dimension
    - Non-contracted vars of B become N dimension
    - Contracted vars become K dimension

    Falls back to pure PyTorch if tropical_gemm unavailable.

    Args:
        a, b: Input tensors in log-space.
        a_vars, b_vars: Variable indices for each tensor.
        out_vars: Output variable indices (from omeco eins.iy).
        track_argmax: Whether to track argmax for backtracing.

    Returns:
        Contracted tensor and optional backpointer.
    """
    # Determine contracted variables
    a_set, b_set, out_set = set(a_vars), set(b_vars), set(out_vars)
    contract_vars = (a_set & b_set) - out_set

    if not contract_vars:
        # No contraction, just element-wise add after alignment
        all_vars = tuple(dict.fromkeys(a_vars + b_vars))
        aligned_a = _align_tensor(a, a_vars, all_vars)
        aligned_b = _align_tensor(b, b_vars, all_vars)
        result = aligned_a + aligned_b
        if all_vars != out_vars:
            result = _align_tensor(result, all_vars, out_vars)
        return result, None

    # Partition variables
    a_only = [v for v in a_vars if v not in b_set]  # M dims
    b_only = [v for v in b_vars if v not in a_set]  # N dims
    shared = [v for v in a_vars if v in b_set]      # Includes contracted + kept

    # Separate shared into contracted vs kept
    contract_list = [v for v in shared if v not in out_set]
    kept_shared = [v for v in shared if v in out_set]

    # Build dimension info
    a_var_to_dim = {v: a.shape[i] for i, v in enumerate(a_vars)}
    b_var_to_dim = {v: b.shape[i] for i, v in enumerate(b_vars)}

    m_vars = tuple(a_only + kept_shared)
    n_vars = tuple(b_only)
    k_vars = tuple(contract_list)

    m_shape = tuple(a_var_to_dim[v] for v in a_only) + tuple(a_var_to_dim[v] for v in kept_shared)
    n_shape = tuple(b_var_to_dim[v] for v in b_only)
    k_shape = tuple(a_var_to_dim[v] for v in contract_list)

    m_size = 1
    for d in m_shape:
        m_size *= d
    n_size = 1
    for d in n_shape:
        n_size *= d
    k_size = 1
    for d in k_shape:
        k_size *= d

    # Permute and reshape A to (M, K)
    a_perm_order = [a_vars.index(v) for v in a_only + kept_shared + contract_list]
    a_permuted = a.permute(a_perm_order) if a_perm_order != list(range(len(a_vars))) else a
    a_matrix = a_permuted.reshape(m_size, k_size)

    # Permute and reshape B to (K, N)
    b_perm_order = [b_vars.index(v) for v in contract_list + b_only]
    # Handle kept_shared: they should broadcast, so we need to be careful
    # Actually for B, we only need contract_list + b_only
    # kept_shared variables in B will need special handling
    if kept_shared:
        # For kept shared vars, we need to align B properly
        # B has shape for (shared + b_only), we need (contract + b_only)
        # This is tricky - fall back to pure PyTorch for this case
        return _tropical_contract_fallback(
            a, b, a_vars, b_vars, out_vars, track_argmax
        )

    b_permuted = b.permute(b_perm_order) if b_perm_order != list(range(len(b_vars))) else b
    b_matrix = b_permuted.reshape(k_size, n_size)

    # Use tropical-gemm if available
    if tropical_gemm is not None:
        # Convert to numpy for tropical_gemm
        a_np = a_matrix.detach().cpu().numpy().astype('float64')
        b_np = b_matrix.detach().cpu().numpy().astype('float64')

        if track_argmax:
            c_np, argmax_np = tropical_gemm.maxplus_matmul_with_argmax_f64(a_np, b_np)
            c_np = c_np.reshape(m_size, n_size)
            argmax_np = argmax_np.reshape(m_size, n_size)
            result = torch.from_numpy(c_np).to(a.device, a.dtype)
            argmax_flat = torch.from_numpy(argmax_np.astype('int64')).to(a.device)
        else:
            c_np = tropical_gemm.maxplus_matmul_f64(a_np, b_np)
            c_np = c_np.reshape(m_size, n_size)
            result = torch.from_numpy(c_np).to(a.device, a.dtype)
            argmax_flat = None
    else:
        # Fallback: use PyTorch
        # C[m,n] = max_k(A[m,k] + B[k,n])
        combined = a_matrix.unsqueeze(-1) + b_matrix.unsqueeze(0)  # (M, K, N)
        if track_argmax:
            result, argmax_flat = combined.max(dim=1)  # (M, N)
        else:
            result = combined.max(dim=1).values
            argmax_flat = None

    # Reshape result to output shape
    result_shape = m_shape + n_shape
    result = result.reshape(result_shape)

    # Reorder to match out_vars
    result_vars = m_vars + n_vars
    if result_vars != out_vars:
        perm = [result_vars.index(v) for v in out_vars]
        result = result.permute(perm)
        if argmax_flat is not None:
            argmax_flat = argmax_flat.reshape(result_shape).permute(perm)

    if not track_argmax or argmax_flat is None:
        return result, None

    # Build backpointer
    backpointer = Backpointer(
        elim_vars=k_vars,
        elim_shape=k_shape,
        out_vars=out_vars,
        argmax_flat=argmax_flat.reshape(result.shape),
    )
    return result, backpointer


def _tropical_contract_fallback(
    a: torch.Tensor,
    b: torch.Tensor,
    a_vars: Tuple[int, ...],
    b_vars: Tuple[int, ...],
    out_vars: Tuple[int, ...],
    track_argmax: bool = True,
) -> tuple[torch.Tensor, Backpointer | None]:
    """Fallback tropical contraction using pure PyTorch."""
    all_vars = tuple(dict.fromkeys(a_vars + b_vars))
    elim_vars = tuple(v for v in all_vars if v not in out_vars)

    aligned_a = _align_tensor(a, a_vars, all_vars)
    aligned_b = _align_tensor(b, b_vars, all_vars)
    combined = aligned_a + aligned_b

    if not elim_vars:
        if all_vars != out_vars:
            combined = _align_tensor(combined, all_vars, out_vars)
        return combined, None

    return tropical_reduce_max(combined, all_vars, elim_vars, track_argmax=track_argmax)


def argmax_trace(backpointer: Backpointer, assignment: Dict[int, int]) -> Dict[int, int]:
    """Decode eliminated variable assignments from a backpointer."""
    if not backpointer.elim_vars:
        return {}
    if backpointer.out_vars:
        missing = [v for v in backpointer.out_vars if v not in assignment]
        if missing:
            raise KeyError(
                "Missing assignment values for output variables: "
                f"{missing}. Provided assignment keys: {sorted(assignment.keys())}"
            )
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
