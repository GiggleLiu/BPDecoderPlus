"""Tropical einsum module following OMEinsum design.

This module implements tropical tensor contractions with rule-based dispatch:
- Unary rules: Identity, TropicalSum (max reduction), Permutedims, Diag
- Binary rules: Pattern-matched to tropical GEMM (maxplus matmul)

In tropical semiring:
- "multiplication" = addition in log-space
- "addition" = max operation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch

try:
    import tropical_gemm
except ImportError:
    tropical_gemm = None

from .primitives import Backpointer


# =============================================================================
# Rule Types
# =============================================================================

class EinRule(ABC):
    """Base class for einsum rules."""

    @abstractmethod
    def execute(
        self,
        tensors: List[torch.Tensor],
        ixs: List[Tuple[int, ...]],
        iy: Tuple[int, ...],
        track_argmax: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Backpointer]]:
        """Execute the rule."""
        pass


# -----------------------------------------------------------------------------
# Unary Rules
# -----------------------------------------------------------------------------

class Identity(EinRule):
    """Identity: ix == iy, just copy."""

    def execute(self, tensors, ixs, iy, track_argmax=False):
        return tensors[0].clone(), None


class TropicalSum(EinRule):
    """Tropical sum: max reduction over eliminated dimensions."""

    def execute(self, tensors, ixs, iy, track_argmax=False):
        x = tensors[0]
        ix = ixs[0]

        # Find variables to eliminate
        elim_vars = tuple(v for v in ix if v not in iy)

        if not elim_vars:
            return x.clone(), None

        # Use tropical_reduce_max which handles backpointer properly
        return tropical_reduce_max(x, ix, elim_vars, track_argmax)


class Permutedims(EinRule):
    """Permute dimensions."""

    def execute(self, tensors, ixs, iy, track_argmax=False):
        x = tensors[0]
        ix = ixs[0]
        perm = tuple(ix.index(v) for v in iy)
        return x.permute(perm).contiguous(), None


class Diag(EinRule):
    """Extract diagonal elements.

    Note: This is a simplified implementation that handles basic diagonal
    extraction. For complex cases with multiple repeated indices and
    permutation, use DefaultRule instead.
    """

    def execute(self, tensors, ixs, iy, track_argmax=False):
        x = tensors[0]
        ix = ixs[0]

        # Find repeated indices in ix
        seen = {}
        for i, v in enumerate(ix):
            if v in seen:
                seen[v].append(i)
            else:
                seen[v] = [i]

        result = x
        # Take diagonal for repeated indices
        for v, dims in seen.items():
            if len(dims) > 1:
                # Take diagonal along these dimensions
                result = torch.diagonal(result, dim1=dims[0], dim2=dims[1])

        return result, None


class Tr(EinRule):
    """Trace: ix = (a, a), iy = ()."""

    def execute(self, tensors, ixs, iy, track_argmax=False):
        x = tensors[0]
        # Tropical trace: max of diagonal elements
        diag = torch.diagonal(x)
        result = diag.max()
        return result.reshape(()), None


# -----------------------------------------------------------------------------
# Binary Rules
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SimpleBinaryRule(EinRule):
    """Binary contraction rule matching GEMM patterns.

    Pattern codes (from OMEinsum):
    - 'i': outer index from first tensor
    - 'j': inner (contracted) index
    - 'k': outer index from second tensor
    - 'l': batch index
    """
    pattern: str  # e.g., "ij,jk->ik" encoded as tuple of tuples

    def execute(self, tensors, ixs, iy, track_argmax=False):
        a, b = tensors
        ix1, ix2 = ixs
        return tropical_binary_einsum(a, b, ix1, ix2, iy, track_argmax)


class DefaultRule(EinRule):
    """Fallback rule using loop-based evaluation."""

    def execute(self, tensors, ixs, iy, track_argmax=False):
        if len(tensors) == 1:
            return tropical_unary_default(tensors[0], ixs[0], iy, track_argmax)
        elif len(tensors) == 2:
            return tropical_binary_default(
                tensors[0], tensors[1], ixs[0], ixs[1], iy, track_argmax
            )
        else:
            raise NotImplementedError(
                f"n-ary contractions (n={len(tensors)}) are not supported. "
                "Use contraction order optimization (e.g., omeco) to decompose "
                "into binary contractions first."
            )


# =============================================================================
# Rule Matching
# =============================================================================

def match_rule(ixs: List[Tuple[int, ...]], iy: Tuple[int, ...]) -> EinRule:
    """Match contraction pattern to optimized rule."""
    if len(ixs) == 1:
        return match_rule_unary(ixs[0], iy)
    elif len(ixs) == 2:
        return match_rule_binary(ixs[0], ixs[1], iy)
    else:
        return DefaultRule()


def match_rule_unary(ix: Tuple[int, ...], iy: Tuple[int, ...]) -> EinRule:
    """Match unary contraction pattern."""
    if len(iy) == 0 and len(ix) == 2 and ix[0] == ix[1]:
        return Tr()

    if len(set(iy)) == len(iy):  # iy is unique
        if ix == iy:
            return Identity()
        if len(set(ix)) == len(ix):  # ix is unique
            if len(ix) == len(iy) and set(ix) == set(iy):
                return Permutedims()
            elif set(iy).issubset(set(ix)):
                return TropicalSum()

    # Check for diagonal extraction
    if len(set(ix)) < len(ix):  # ix has repeated indices
        if set(iy).issubset(set(ix)):
            return Diag()

    return DefaultRule()


def match_rule_binary(
    ix1: Tuple[int, ...], ix2: Tuple[int, ...], iy: Tuple[int, ...]
) -> EinRule:
    """Match binary contraction pattern to GEMM form."""
    # Check if all indices are unique within each tensor
    if len(set(ix1)) != len(ix1) or len(set(ix2)) != len(ix2) or len(set(iy)) != len(iy):
        return DefaultRule()

    # Identify index types
    ix1_set, ix2_set, iy_set = set(ix1), set(ix2), set(iy)

    # Contracted indices: in both inputs but not in output
    contracted = (ix1_set & ix2_set) - iy_set

    # Check for valid GEMM-like pattern
    # For tropical GEMM: C[i,k] = max_j(A[i,j] + B[j,k])
    if len(contracted) <= 1 and len(ix1) <= 3 and len(ix2) <= 3:
        return SimpleBinaryRule(pattern=f"{ix1},{ix2}->{iy}")

    return DefaultRule()


# =============================================================================
# Execution Functions
# =============================================================================

def tropical_reduce_max(
    tensor: torch.Tensor,
    vars: Tuple[int, ...],
    elim_vars: Tuple[int, ...],
    track_argmax: bool = True,
) -> Tuple[torch.Tensor, Optional[Backpointer]]:
    """Tropical sum (max) over specified variables."""
    if not elim_vars:
        return tensor, None

    # Validate that all elimination variables are present in vars
    missing_vars = [v for v in elim_vars if v not in vars]
    if missing_vars:
        raise ValueError(
            f"Elimination variables {missing_vars} are not present in vars {vars}"
        )

    elim_axes = [vars.index(v) for v in elim_vars]
    keep_axes = [i for i in range(len(vars)) if i not in elim_axes]

    # Permute to put elimination axes at end
    perm = keep_axes + elim_axes
    permuted = tensor.permute(perm) if perm != list(range(len(vars))) else tensor

    out_shape = [tensor.shape[i] for i in keep_axes]
    elim_shape = [tensor.shape[i] for i in elim_axes]

    # Flatten elimination dims and take max
    flat = permuted.reshape(*out_shape, -1) if out_shape else permuted.reshape(-1)

    if track_argmax:
        values, argmax_flat = torch.max(flat, dim=-1)
    else:
        values = torch.max(flat, dim=-1).values
        argmax_flat = None

    if not track_argmax:
        return values, None

    backpointer = Backpointer(
        elim_vars=elim_vars,
        elim_shape=tuple(elim_shape),
        out_vars=tuple(vars[i] for i in keep_axes),
        argmax_flat=argmax_flat,
    )
    return values, backpointer


def _align_tensor(
    tensor: torch.Tensor,
    tensor_vars: Tuple[int, ...],
    target_vars: Tuple[int, ...]
) -> torch.Tensor:
    """Align tensor dimensions to target variable order."""
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


def tropical_binary_einsum(
    a: torch.Tensor,
    b: torch.Tensor,
    ix1: Tuple[int, ...],
    ix2: Tuple[int, ...],
    iy: Tuple[int, ...],
    track_argmax: bool = True,
) -> Tuple[torch.Tensor, Optional[Backpointer]]:
    """Binary tropical contraction using tropical-gemm when possible."""
    ix1_set, ix2_set, iy_set = set(ix1), set(ix2), set(iy)
    contracted = (ix1_set & ix2_set) - iy_set

    # Try to use tropical-gemm for single contraction index
    if tropical_gemm is not None and len(contracted) == 1:
        result = _tropical_gemm_contract(a, b, ix1, ix2, iy, track_argmax)
        if result is not None:
            return result

    # Fallback to default implementation
    return tropical_binary_default(a, b, ix1, ix2, iy, track_argmax)


def _tropical_gemm_contract(
    a: torch.Tensor,
    b: torch.Tensor,
    ix1: Tuple[int, ...],
    ix2: Tuple[int, ...],
    iy: Tuple[int, ...],
    track_argmax: bool,
) -> Optional[Tuple[torch.Tensor, Optional[Backpointer]]]:
    """Use tropical-gemm for matrix multiplication pattern."""
    ix1_set, ix2_set, iy_set = set(ix1), set(ix2), set(iy)
    contracted = list((ix1_set & ix2_set) - iy_set)

    if len(contracted) != 1:
        return None

    j = contracted[0]  # The contracted index

    # Identify M, N, K dimensions
    # M: indices only in ix1 that appear in iy
    # N: indices only in ix2 that appear in iy
    # K: contracted index j

    m_vars = tuple(v for v in ix1 if v not in ix2_set and v in iy_set)
    n_vars = tuple(v for v in ix2 if v not in ix1_set and v in iy_set)
    k_vars = (j,)

    # Batch indices: shared between inputs and in output
    batch_vars = tuple(v for v in ix1 if v in ix2_set and v in iy_set)

    # Get dimension sizes
    a_var_to_dim = {v: a.shape[i] for i, v in enumerate(ix1)}
    b_var_to_dim = {v: b.shape[i] for i, v in enumerate(ix2)}

    m_shape = tuple(a_var_to_dim[v] for v in m_vars)
    n_shape = tuple(b_var_to_dim[v] for v in n_vars)
    k_shape = tuple(a_var_to_dim[v] for v in k_vars)
    batch_shape = tuple(a_var_to_dim[v] for v in batch_vars)

    m_size = int(np.prod(m_shape)) if m_shape else 1
    n_size = int(np.prod(n_shape)) if n_shape else 1
    k_size = int(np.prod(k_shape)) if k_shape else 1
    batch_size = int(np.prod(batch_shape)) if batch_shape else 1

    # Permute A to (batch, M, K) and reshape to (batch*M, K) or (M, K)
    a_perm = [ix1.index(v) for v in batch_vars + m_vars + k_vars]
    a_permuted = a.permute(a_perm) if a_perm != list(range(len(ix1))) else a

    # Permute B to (batch, K, N) and reshape to (batch*K, N) or (K, N)
    b_perm = [ix2.index(v) for v in batch_vars + k_vars + n_vars]
    b_permuted = b.permute(b_perm) if b_perm != list(range(len(ix2))) else b

    if batch_vars:
        # Batched tropical GEMM
        # NOTE: Processing batches sequentially since tropical_gemm doesn't support
        # native batched operations. For large batch sizes, this may be inefficient.
        # Consider implementing batched support in tropical_gemm for better performance.
        a_matrix = a_permuted.reshape(batch_size, m_size, k_size)
        b_matrix = b_permuted.reshape(batch_size, k_size, n_size)

        results = []
        argmaxes = [] if track_argmax else None

        for i in range(batch_size):
            a_np = a_matrix[i].detach().cpu().numpy().astype('float64')
            b_np = b_matrix[i].detach().cpu().numpy().astype('float64')

            if track_argmax:
                c_np, argmax_np = tropical_gemm.maxplus_matmul_with_argmax_f64(a_np, b_np)
                argmaxes.append(torch.from_numpy(argmax_np.reshape(m_size, n_size).astype('int64')))
            else:
                c_np = tropical_gemm.maxplus_matmul_f64(a_np, b_np)

            results.append(torch.from_numpy(c_np.reshape(m_size, n_size)))

        result = torch.stack(results).to(a.device, a.dtype)
        if track_argmax:
            argmax_flat = torch.stack(argmaxes).to(a.device)
        else:
            argmax_flat = None

        # Reshape to (batch_shape, m_shape, n_shape)
        result = result.reshape(batch_shape + m_shape + n_shape)
        if argmax_flat is not None:
            argmax_flat = argmax_flat.reshape(batch_shape + m_shape + n_shape)
    else:
        # Non-batched tropical GEMM
        a_matrix = a_permuted.reshape(m_size, k_size)
        b_matrix = b_permuted.reshape(k_size, n_size)

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

        # Reshape to (m_shape, n_shape)
        result = result.reshape(m_shape + n_shape)
        if argmax_flat is not None:
            argmax_flat = argmax_flat.reshape(m_shape + n_shape)

    # Permute result to match iy
    result_vars = batch_vars + m_vars + n_vars
    if result_vars != iy:
        perm = [result_vars.index(v) for v in iy]
        result = result.permute(perm).contiguous()
        if argmax_flat is not None:
            argmax_flat = argmax_flat.permute(perm).contiguous()

    if not track_argmax:
        return result, None

    backpointer = Backpointer(
        elim_vars=k_vars,
        elim_shape=k_shape,
        out_vars=iy,
        argmax_flat=argmax_flat,
    )
    return result, backpointer


def tropical_binary_default(
    a: torch.Tensor,
    b: torch.Tensor,
    ix1: Tuple[int, ...],
    ix2: Tuple[int, ...],
    iy: Tuple[int, ...],
    track_argmax: bool = True,
) -> Tuple[torch.Tensor, Optional[Backpointer]]:
    """Default tropical binary contraction using PyTorch."""
    # Build combined variable order
    all_vars = tuple(dict.fromkeys(ix1 + ix2))
    elim_vars = tuple(v for v in all_vars if v not in iy)

    # Align and add (tropical multiply)
    aligned_a = _align_tensor(a, ix1, all_vars)
    aligned_b = _align_tensor(b, ix2, all_vars)
    combined = aligned_a + aligned_b

    if not elim_vars:
        # Reorder to match output
        if all_vars != iy:
            perm = [all_vars.index(v) for v in iy]
            combined = combined.permute(perm)
        return combined, None

    # Tropical sum (max) over eliminated variables
    return tropical_reduce_max(combined, all_vars, elim_vars, track_argmax)


def tropical_unary_default(
    x: torch.Tensor,
    ix: Tuple[int, ...],
    iy: Tuple[int, ...],
    track_argmax: bool = True,
) -> Tuple[torch.Tensor, Optional[Backpointer]]:
    """Default tropical unary operation."""
    elim_vars = tuple(v for v in ix if v not in iy)

    if not elim_vars:
        # Just permute
        if ix != iy:
            perm = [ix.index(v) for v in iy]
            return x.permute(perm).contiguous(), None
        return x.clone(), None

    return tropical_reduce_max(x, ix, elim_vars, track_argmax)


# =============================================================================
# Main API
# =============================================================================

def tropical_einsum(
    tensors: List[torch.Tensor],
    ixs: List[Tuple[int, ...]],
    iy: Tuple[int, ...],
    track_argmax: bool = True,
) -> Tuple[torch.Tensor, Optional[Backpointer]]:
    """Tropical einsum following OMEinsum design.

    Args:
        tensors: Input tensors in log-space.
        ixs: Index tuples for each tensor.
        iy: Output indices.
        track_argmax: Whether to track argmax for backtracing.

    Returns:
        Result tensor and optional backpointer.

    Example:
        # Matrix multiplication: C[i,k] = max_j(A[i,j] + B[j,k])
        result, bp = tropical_einsum(
            [A, B],
            [(0, 1), (1, 2)],
            (0, 2),
        )
    """
    rule = match_rule(ixs, iy)
    return rule.execute(tensors, ixs, iy, track_argmax)


def argmax_trace(backpointer: Backpointer, assignment: dict[int, int]) -> dict[int, int]:
    """Decode eliminated variable assignments from a backpointer."""
    if not backpointer.elim_vars:
        return {}

    if backpointer.out_vars:
        # Validate that all required output variables are present in the assignment
        missing = [v for v in backpointer.out_vars if v not in assignment]
        if missing:
            raise KeyError(
                f"Missing assignment values for output variables: {missing}"
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
