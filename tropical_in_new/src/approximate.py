"""Approximate tensor network contraction for tropical semiring.

This module implements MPS-based approximate contraction methods for
tensor networks in the tropical (max-plus) semiring, enabling scalable
decoding for surface codes with d>=5.

Key classes:
- TropicalMPS: Matrix Product State representation in tropical semiring
- TropicalMPO: Matrix Product Operator for row-by-row contraction

Key functions:
- boundary_contract: MPS boundary contraction with truncation
- tropical_svd_approx: SVD-like truncation for tropical tensors

References:
- Bravyi et al., arXiv:1405.4883 - MPS decoder for surface codes
- Chubb, arXiv:2101.04125 - General tensor network decoding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import math

import torch


@dataclass
class TropicalMPS:
    """Matrix Product State in the tropical (max-plus) semiring.
    
    An MPS represents a tensor as a product of local tensors (sites):
        T[i1, i2, ..., iN] ≈ A1[i1] @ A2[i2] @ ... @ AN[iN]
    
    In tropical semiring:
    - Matrix multiplication uses (max, +) instead of (+, *)
    - The "bond dimension" χ limits the number of paths tracked
    
    Attributes:
        sites: List of tensors, each with shape (chi_left, phys_dim, chi_right)
               For boundary sites: chi_left=1 or chi_right=1
        physical_dims: Physical dimensions at each site
        chi: Maximum bond dimension (None means unlimited)
    """
    sites: List[torch.Tensor]
    physical_dims: List[int] = field(default_factory=list)
    chi: Optional[int] = None
    
    def __post_init__(self):
        if not self.physical_dims and self.sites:
            self.physical_dims = [s.shape[1] for s in self.sites]
    
    @property
    def num_sites(self) -> int:
        return len(self.sites)
    
    @property
    def bond_dims(self) -> List[int]:
        """Return bond dimensions between sites."""
        if not self.sites:
            return []
        return [self.sites[i].shape[2] for i in range(len(self.sites) - 1)]
    
    def max_bond_dim(self) -> int:
        """Return maximum bond dimension."""
        if not self.sites:
            return 0
        if len(self.sites) == 1:
            return 1
        return max(s.shape[0] for s in self.sites[1:])
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, chi: Optional[int] = None) -> "TropicalMPS":
        """Convert a full tensor to MPS form with optional truncation.
        
        Uses sequential SVD decomposition with tropical approximation.
        
        Args:
            tensor: Full tensor of shape (d1, d2, ..., dN)
            chi: Maximum bond dimension. If None, no truncation.
            
        Returns:
            TropicalMPS representation of the tensor.
        """
        if tensor.ndim == 0:
            return cls(sites=[tensor.reshape(1, 1, 1)], chi=chi)
        
        if tensor.ndim == 1:
            return cls(sites=[tensor.reshape(1, tensor.shape[0], 1)], chi=chi)
        
        sites = []
        physical_dims = list(tensor.shape)
        
        # For 2D tensor, simple decomposition
        if tensor.ndim == 2:
            d0, d1 = tensor.shape
            # First site: (1, d0, chi) where chi = min(d0, d1, chi_max)
            mat = tensor
            u, s, v = tropical_svd_approx(mat, chi)
            chi_actual = u.shape[1]
            
            # First site
            sites.append(u.reshape(1, d0, chi_actual))
            # Second site: absorb s into v
            sv = s.unsqueeze(1) + v
            sites.append(sv.reshape(chi_actual, d1, 1))
            
            return cls(sites=sites, physical_dims=physical_dims, chi=chi)
        
        # For higher-order tensors, use iterative decomposition
        remaining = tensor.clone()
        chi_left = 1
        
        for i in range(tensor.ndim - 1):
            phys_dim = remaining.shape[0] if i == 0 else remaining.shape[1]
            
            if i == 0:
                # First iteration: remaining shape is (d0, d1, d2, ...)
                mat = remaining.reshape(physical_dims[0], -1)
            else:
                # Later iterations: remaining shape is (chi_left, d_i, d_{i+1}, ...)
                mat = remaining.reshape(chi_left * physical_dims[i], -1)
            
            # SVD decomposition
            u, s, v = tropical_svd_approx(mat, chi)
            chi_right = u.shape[1]
            
            if i == 0:
                site = u.reshape(1, physical_dims[0], chi_right)
            else:
                site = u.reshape(chi_left, physical_dims[i], chi_right)
            sites.append(site)
            
            # Prepare remaining: absorb s into v
            sv = s.unsqueeze(1) + v
            remaining_size = sv.shape[1]
            remaining_dims = physical_dims[i+1:]
            expected_size = 1
            for d in remaining_dims:
                expected_size *= d
            
            if remaining_size == expected_size:
                remaining = sv.reshape(chi_right, *remaining_dims)
            else:
                # Size mismatch - just use sv as-is for next iteration
                remaining = sv
            
            chi_left = chi_right
        
        # Final site
        if remaining.ndim == 1:
            sites.append(remaining.reshape(chi_left, remaining.shape[0], 1))
        elif remaining.ndim >= 2:
            final_phys = remaining.shape[1] if remaining.ndim >= 2 else remaining.numel()
            sites.append(remaining.reshape(chi_left, final_phys, 1))
        
        return cls(sites=sites, physical_dims=physical_dims, chi=chi)
    
    def to_tensor(self) -> torch.Tensor:
        """Contract MPS back to full tensor (for small tensors only)."""
        if not self.sites:
            raise ValueError("Empty MPS")
        
        if len(self.sites) == 1:
            # Single site: just squeeze boundary dims
            return self.sites[0].squeeze(0).squeeze(-1)
        
        # Contract sites sequentially
        # sites[i] has shape (chi_left, d_i, chi_right)
        result = self.sites[0]  # (1, d0, chi1)
        
        for site in self.sites[1:]:
            # result: (chi_l, d_prev, ..., chi_mid)
            # site: (chi_mid, d_i, chi_r)
            # Contract over chi_mid dimension
            
            # Get the last dim of result and first dim of site
            chi_mid_result = result.shape[-1]
            chi_mid_site = site.shape[0]
            
            if chi_mid_result != chi_mid_site:
                # Dimension mismatch - try to handle gracefully
                # This can happen with approximations
                min_chi = min(chi_mid_result, chi_mid_site)
                result = result[..., :min_chi]
                site = site[:min_chi, ...]
            
            # Reshape for contraction
            result_shape = result.shape[:-1]  # All but last
            site_shape = site.shape[1:]  # All but first
            
            # result: (*result_shape, chi) -> (*result_shape, chi, 1, 1, ...)
            # site: (chi, *site_shape) -> (1, 1, ..., chi, *site_shape)
            
            # Expand dimensions for broadcasting
            n_result_dims = len(result_shape)
            n_site_dims = len(site_shape)
            
            result_exp = result.reshape(*result_shape, result.shape[-1], *([1] * n_site_dims))
            site_exp = site.reshape(*([1] * n_result_dims), site.shape[0], *site_shape)
            
            # Tropical contraction: add and max over chi
            combined = result_exp + site_exp
            result = combined.max(dim=n_result_dims).values
        
        # Squeeze boundary dimensions (first and last should be 1)
        while result.ndim > 0 and result.shape[0] == 1:
            result = result.squeeze(0)
        while result.ndim > 0 and result.shape[-1] == 1:
            result = result.squeeze(-1)
        
        return result
    
    def copy(self) -> "TropicalMPS":
        """Create a deep copy of the MPS."""
        return TropicalMPS(
            sites=[s.clone() for s in self.sites],
            physical_dims=self.physical_dims.copy(),
            chi=self.chi
        )


@dataclass 
class TropicalMPO:
    """Matrix Product Operator in the tropical semiring.
    
    An MPO represents a linear operator as a product of local tensors:
        O[i1,j1; i2,j2; ...] = W1[i1,j1] @ W2[i2,j2] @ ...
    
    Each site tensor has shape (chi_left, phys_in, phys_out, chi_right).
    
    Attributes:
        sites: List of tensors with shape (chi_l, d_in, d_out, chi_r)
        physical_dims_in: Input physical dimensions
        physical_dims_out: Output physical dimensions
    """
    sites: List[torch.Tensor]
    physical_dims_in: List[int] = field(default_factory=list)
    physical_dims_out: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.physical_dims_in and self.sites:
            self.physical_dims_in = [s.shape[1] for s in self.sites]
        if not self.physical_dims_out and self.sites:
            self.physical_dims_out = [s.shape[2] for s in self.sites]
    
    @property
    def num_sites(self) -> int:
        return len(self.sites)
    
    @classmethod
    def from_tensor_row(
        cls,
        tensors: List[torch.Tensor],
        connections: List[Tuple[int, int]],
    ) -> "TropicalMPO":
        """Build MPO from a row of tensors in the network.
        
        Args:
            tensors: List of factor tensors in this row
            connections: List of (site_idx, var_idx) pairs specifying
                        which variables connect to the row below
                        
        Returns:
            TropicalMPO representing the row operator
        """
        # Simplified implementation: treat each tensor as an MPO site
        sites = []
        for tensor in tensors:
            if tensor.ndim == 1:
                # Vector -> MPO site with trivial bond dims
                site = tensor.reshape(1, tensor.shape[0], 1, 1)
            elif tensor.ndim == 2:
                # Matrix -> MPO site
                site = tensor.reshape(1, tensor.shape[0], tensor.shape[1], 1)
            else:
                # Higher-order tensor: reshape appropriately
                # Assume first half dims are input, second half are output
                mid = tensor.ndim // 2
                in_shape = tensor.shape[:mid]
                out_shape = tensor.shape[mid:]
                in_dim = math.prod(in_shape) if in_shape else 1
                out_dim = math.prod(out_shape) if out_shape else 1
                site = tensor.reshape(1, in_dim, out_dim, 1)
            sites.append(site)
        
        return cls(sites=sites)


# =============================================================================
# Tropical SVD Approximation
# =============================================================================

def tropical_svd_approx(
    matrix: torch.Tensor,
    chi: Optional[int] = None,
    return_values: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD-like decomposition for tropical semiring.
    
    Since tropical semiring doesn't have a true SVD, we use an approximation:
    1. Treat log-values as regular values
    2. Apply standard SVD
    3. Truncate to top-chi singular values
    4. Return decomposition in a form suitable for tropical operations
    
    For tropical networks, this keeps the top-χ "most important" paths.
    
    Args:
        matrix: 2D tensor in log-domain (tropical values)
        chi: Maximum rank to keep. If None, keep all.
        return_values: If True, return singular values separately
        
    Returns:
        Tuple of (U, S, V) where:
        - U: (m, k) left singular vectors
        - S: (k,) singular values (in log domain)
        - V: (k, n) right singular vectors
        k = min(chi, rank) if chi is not None else rank
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")
    
    m, n = matrix.shape
    
    # Handle -inf values by replacing with very negative finite values
    neg_inf = float('-inf')
    finite_min = -1e10
    matrix_finite = torch.where(
        matrix == neg_inf,
        torch.full_like(matrix, finite_min),
        matrix
    )
    
    # Standard SVD on the log-domain values
    try:
        U, S, Vh = torch.linalg.svd(matrix_finite, full_matrices=False)
    except RuntimeError:
        # Fallback for numerical issues
        U = torch.eye(m, min(m, n), dtype=matrix.dtype, device=matrix.device)
        S = torch.ones(min(m, n), dtype=matrix.dtype, device=matrix.device)
        Vh = torch.eye(min(m, n), n, dtype=matrix.dtype, device=matrix.device)
    
    # Truncate to chi
    k = min(m, n)
    if chi is not None and chi < k:
        k = chi
        U = U[:, :k]
        S = S[:k]
        Vh = Vh[:k, :]
    
    # Convert S to log-domain scaling
    # In tropical semiring, we want to track the "weight" of each path
    # Use log(S) as the tropical singular value
    S_tropical = torch.log(S.abs() + 1e-10)
    
    return U, S_tropical, Vh


def tropical_truncate_bond(
    tensor: torch.Tensor,
    bond_dim: int,
    axis: int,
    return_info: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Truncate a tensor along a bond dimension by keeping top values.
    
    In tropical semiring, we keep the paths with highest values (most probable).
    
    Args:
        tensor: Input tensor
        bond_dim: Target dimension to truncate to
        axis: Which axis to truncate
        return_info: If True, return indices of kept values
        
    Returns:
        Truncated tensor (and optionally kept indices)
    """
    if tensor.shape[axis] <= bond_dim:
        return tensor, None
    
    # Move target axis to the end
    tensor = tensor.movedim(axis, -1)
    original_shape = tensor.shape[:-1]
    
    # Flatten all but last axis
    flat = tensor.reshape(-1, tensor.shape[-1])
    
    # For each row, find top-k values
    # Use max over other dimensions as importance metric
    importance = flat.max(dim=0).values
    _, top_indices = torch.topk(importance, bond_dim)
    top_indices = top_indices.sort().values
    
    # Select top columns
    truncated = flat[:, top_indices]
    
    # Reshape back
    result = truncated.reshape(*original_shape, bond_dim)
    result = result.movedim(-1, axis)
    
    if return_info:
        return result, top_indices
    return result, None


# =============================================================================
# Tropical Tensor Operations
# =============================================================================

def tropical_tensor_contract(
    a: torch.Tensor,
    b: torch.Tensor,
    contract_dims: Tuple[int, int] = (-1, 0),
) -> torch.Tensor:
    """Contract two tensors using tropical (max-plus) semiring.
    
    Performs contraction where:
    - Multiplication -> Addition (in log space)
    - Summation -> Maximum
    
    Args:
        a: First tensor
        b: Second tensor  
        contract_dims: Which dimensions to contract (default: last of a, first of b)
        
    Returns:
        Contracted tensor
    """
    dim_a, dim_b = contract_dims
    
    # Normalize negative indices
    if dim_a < 0:
        dim_a = a.ndim + dim_a
    if dim_b < 0:
        dim_b = b.ndim + dim_b
    
    # Check dimensions match
    if a.shape[dim_a] != b.shape[dim_b]:
        raise ValueError(
            f"Contraction dimension mismatch: {a.shape[dim_a]} vs {b.shape[dim_b]}"
        )
    
    contract_size = a.shape[dim_a]
    
    # Move contraction dims to standard positions
    a_perm = list(range(a.ndim))
    a_perm.remove(dim_a)
    a_perm.append(dim_a)
    a = a.permute(a_perm)
    
    b_perm = list(range(b.ndim))
    b_perm.remove(dim_b)
    b_perm.insert(0, dim_b)
    b = b.permute(b_perm)
    
    # Now a has shape (..., k) and b has shape (k, ...)
    # Result will have shape (a.shape[:-1], b.shape[1:])
    
    a_shape = a.shape[:-1]
    b_shape = b.shape[1:]
    
    # Reshape for broadcasting
    # a: (*a_shape, k, 1, 1, ...)
    # b: (1, 1, ..., k, *b_shape)
    a_broadcast = a.reshape(*a_shape, contract_size, *([1] * len(b_shape)))
    b_broadcast = b.reshape(*([1] * len(a_shape)), contract_size, *b_shape)
    
    # Tropical multiplication (addition in log space)
    combined = a_broadcast + b_broadcast
    
    # Tropical sum (max) over contracted dimension
    result = combined.max(dim=len(a_shape)).values
    
    return result


def tropical_mps_mpo_multiply(
    mps: TropicalMPS,
    mpo: TropicalMPO,
    chi: Optional[int] = None,
) -> TropicalMPS:
    """Multiply MPS by MPO with optional truncation.
    
    Computes: MPS' = MPO @ MPS
    
    This is the core operation for boundary contraction:
    - Apply one row of factors (MPO) to the boundary state (MPS)
    - Truncate to keep bond dimension bounded
    
    Args:
        mps: Input Matrix Product State
        mpo: Matrix Product Operator to apply
        chi: Maximum bond dimension after truncation
        
    Returns:
        New MPS representing MPO @ MPS
    """
    if mps.num_sites != mpo.num_sites:
        raise ValueError(
            f"MPS has {mps.num_sites} sites but MPO has {mpo.num_sites}"
        )
    
    if chi is None:
        chi = mps.chi
    
    new_sites = []
    
    for i in range(mps.num_sites):
        mps_site = mps.sites[i]  # (chi_l, d_mps, chi_r)
        mpo_site = mpo.sites[i]  # (kappa_l, d_in, d_out, kappa_r)
        
        chi_l, d_mps, chi_r = mps_site.shape
        kappa_l, d_in, d_out, kappa_r = mpo_site.shape
        
        if d_mps != d_in:
            raise ValueError(
                f"Dimension mismatch at site {i}: MPS has {d_mps}, MPO expects {d_in}"
            )
        
        # Contract over physical index d_in
        # MPS site: (chi_l, d_in, chi_r)
        # MPO site: (kappa_l, d_in, d_out, kappa_r)
        # Result: (chi_l, kappa_l, d_out, chi_r, kappa_r)
        
        # Reshape for broadcasting:
        # mps: (chi_l, 1, d_in, 1, chi_r, 1)
        # mpo: (1, kappa_l, d_in, d_out, 1, kappa_r)
        mps_exp = mps_site.reshape(chi_l, 1, d_in, 1, chi_r, 1)
        mpo_exp = mpo_site.reshape(1, kappa_l, d_in, d_out, 1, kappa_r)
        
        # Tropical product (add in log space) - broadcasts
        combined = mps_exp + mpo_exp  # (chi_l, kappa_l, d_in, d_out, chi_r, kappa_r)
        
        # Tropical sum (max) over contracted dimension d_in
        contracted = combined.max(dim=2).values  # (chi_l, kappa_l, d_out, chi_r, kappa_r)
        
        # Reshape to new MPS site: (chi_l * kappa_l, d_out, chi_r * kappa_r)
        new_chi_l = chi_l * kappa_l
        new_chi_r = chi_r * kappa_r
        new_site = contracted.reshape(new_chi_l, d_out, new_chi_r)
        new_sites.append(new_site)
    
    # Create new MPS
    new_mps = TropicalMPS(
        sites=new_sites,
        physical_dims=[s.shape[1] for s in new_sites],
        chi=chi
    )
    
    # Truncate if needed
    if chi is not None:
        new_mps = truncate_mps(new_mps, chi)
    
    return new_mps


def truncate_mps(mps: TropicalMPS, chi: int) -> TropicalMPS:
    """Truncate MPS bond dimensions to at most chi.
    
    Uses a sweep-based truncation that maintains MPS structure:
    1. Left-to-right sweep: truncate each bond using tropical SVD
    2. Absorb truncation into next site
    
    Args:
        mps: Input MPS to truncate
        chi: Maximum bond dimension
        
    Returns:
        New MPS with bond dimensions <= chi
    """
    if mps.num_sites <= 1:
        return mps.copy()
    
    sites = [s.clone() for s in mps.sites]
    
    # Left-to-right sweep
    for i in range(len(sites) - 1):
        site = sites[i]  # (chi_l, d, chi_r)
        chi_l, d, chi_r = site.shape
        
        if chi_r <= chi:
            continue
        
        # Reshape to matrix for SVD: (chi_l * d, chi_r)
        mat = site.reshape(chi_l * d, chi_r)
        
        # Tropical SVD approximation
        U, S, V = tropical_svd_approx(mat, chi)
        
        # U becomes truncated site
        new_chi_r = U.shape[1]
        sites[i] = U.reshape(chi_l, d, new_chi_r)
        
        # Absorb S @ V into next site
        # next_site: (chi_r, d_next, chi_r_next)
        next_site = sites[i + 1]
        _, d_next, chi_r_next = next_site.shape
        
        # S @ V: (new_chi_r, chi_r)
        sv = S.unsqueeze(1) + V  # tropical multiply
        
        # Contract with next site over chi_r
        # (new_chi_r, chi_r) @ (chi_r, d_next * chi_r_next)
        next_flat = next_site.reshape(chi_r, d_next * chi_r_next)
        
        # Tropical matmul
        contracted = tropical_tensor_contract(sv, next_flat, (-1, 0))
        sites[i + 1] = contracted.reshape(new_chi_r, d_next, chi_r_next)
    
    return TropicalMPS(
        sites=sites,
        physical_dims=mps.physical_dims.copy(),
        chi=chi
    )


# =============================================================================
# Backpointer for Approximate Contraction
# =============================================================================

@dataclass
class ApproximateBackpointer:
    """Stores information for recovering MPE assignment from approximate contraction.
    
    Unlike exact contraction which tracks all argmax values, approximate
    contraction only tracks information for the top-χ paths.
    
    Attributes:
        truncation_info: List of (site_idx, kept_indices) for each truncation
        path_values: Values (log-probabilities) of tracked paths
        path_assignments: Partial assignments for each tracked path
    """
    truncation_info: List[Tuple[int, torch.Tensor]] = field(default_factory=list)
    path_values: Optional[torch.Tensor] = None
    path_assignments: Optional[Dict[int, torch.Tensor]] = None
    
    def record_truncation(self, site_idx: int, kept_indices: torch.Tensor):
        """Record which indices were kept at a truncation step."""
        self.truncation_info.append((site_idx, kept_indices.clone()))
    
    def get_best_assignment(self) -> Dict[int, int]:
        """Recover the best (highest probability) assignment."""
        if self.path_values is None or self.path_assignments is None:
            return {}
        
        # Find path with maximum value
        best_idx = self.path_values.argmax().item()
        
        assignment = {}
        for var, values in self.path_assignments.items():
            assignment[var] = int(values[best_idx].item())
        
        return assignment


# =============================================================================
# Boundary Contraction Algorithm
# =============================================================================

@dataclass
class BoundaryContractionResult:
    """Result of boundary contraction.
    
    Attributes:
        value: Final scalar value (log-probability of MPE)
        mps: Final MPS (should be scalar after full contraction)
        backpointer: Information for recovering assignment
        chi_used: Maximum bond dimension actually used
    """
    value: float
    mps: TropicalMPS
    backpointer: ApproximateBackpointer
    chi_used: int


def _arrange_tensors_in_rows(
    tensors: List[torch.Tensor],
    vars_list: List[Tuple[int, ...]],
    var_order: Optional[List[int]] = None,
) -> List[List[Tuple[torch.Tensor, Tuple[int, ...]]]]:
    """Arrange tensors into rows for boundary contraction.
    
    Heuristically partitions tensors into rows based on variable ordering.
    
    Args:
        tensors: List of factor tensors
        vars_list: Variables for each tensor
        var_order: Variable ordering (determines row assignment)
        
    Returns:
        List of rows, each containing (tensor, vars) pairs
    """
    if var_order is None:
        # Default: use variable indices directly
        all_vars = set()
        for vars in vars_list:
            all_vars.update(vars)
        var_order = sorted(all_vars)
    
    # Assign each variable to a row
    var_to_row = {v: i // 2 for i, v in enumerate(var_order)}
    num_rows = max(var_to_row.values()) + 1 if var_to_row else 1
    
    rows: List[List[Tuple[torch.Tensor, Tuple[int, ...]]]] = [[] for _ in range(num_rows)]
    
    for tensor, vars in zip(tensors, vars_list):
        if not vars:
            rows[0].append((tensor, vars))
            continue
        # Assign to row of first variable
        row_idx = var_to_row.get(vars[0], 0)
        rows[row_idx].append((tensor, vars))
    
    # Filter empty rows
    rows = [r for r in rows if r]
    
    return rows


def boundary_contract(
    tensors: List[torch.Tensor],
    vars_list: List[Tuple[int, ...]],
    chi: int = 32,
    track_assignment: bool = True,
) -> BoundaryContractionResult:
    """Contract tensor network using MPS boundary method.
    
    This implements the MPS decoder approach from Bravyi et al.:
    1. Arrange tensors into rows
    2. Initialize boundary MPS from first row
    3. For each subsequent row:
       a. Build MPO from row tensors
       b. Apply MPO to MPS: MPS' = MPO @ MPS
       c. Truncate MPS to bond dimension chi
    4. Contract final MPS to scalar
    
    Args:
        tensors: List of factor tensors (in log domain)
        vars_list: Variables for each tensor
        chi: Maximum bond dimension
        track_assignment: Whether to track information for recovering assignment
        
    Returns:
        BoundaryContractionResult with final value and reconstruction info
    """
    if not tensors:
        return BoundaryContractionResult(
            value=0.0,
            mps=TropicalMPS(sites=[]),
            backpointer=ApproximateBackpointer(),
            chi_used=0
        )
    
    backpointer = ApproximateBackpointer()
    
    # Arrange tensors into rows
    rows = _arrange_tensors_in_rows(tensors, vars_list)
    
    if not rows:
        return BoundaryContractionResult(
            value=0.0,
            mps=TropicalMPS(sites=[]),
            backpointer=backpointer,
            chi_used=0
        )
    
    # Initialize MPS from first row
    # For simplicity, combine all tensors in first row
    first_row_tensors = [t for t, _ in rows[0]]
    if len(first_row_tensors) == 1:
        combined = first_row_tensors[0]
    else:
        # Tropical product of independent factors
        combined = first_row_tensors[0]
        for t in first_row_tensors[1:]:
            # Outer product with tropical multiplication
            combined = combined.unsqueeze(-1) + t.unsqueeze(0)
            combined = combined.reshape(-1)
    
    mps = TropicalMPS.from_tensor(combined.flatten(), chi=chi)
    max_chi_used = mps.max_bond_dim()
    
    # Process remaining rows
    for row_idx, row in enumerate(rows[1:], 1):
        row_tensors = [t for t, _ in row]
        
        if not row_tensors:
            continue
        
        # Build MPO from row
        mpo = TropicalMPO.from_tensor_row(row_tensors, [])
        
        # Apply MPO to MPS
        mps = tropical_mps_mpo_multiply(mps, mpo, chi=chi)
        
        max_chi_used = max(max_chi_used, mps.max_bond_dim())
    
    # Contract MPS to scalar
    final_value = mps.to_tensor()
    if final_value.numel() > 1:
        final_value = final_value.max()
    scalar_value = float(final_value.item())
    
    return BoundaryContractionResult(
        value=scalar_value,
        mps=mps,
        backpointer=backpointer,
        chi_used=max_chi_used
    )
