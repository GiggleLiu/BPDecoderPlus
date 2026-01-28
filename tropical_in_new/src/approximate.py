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
    
    Tracks both the truncation decisions and the variable assignments for each
    surviving path configuration.
    
    Attributes:
        truncation_info: List of (site_idx, kept_indices) for each truncation
        path_values: Values (log-probabilities) of tracked paths, shape (chi,)
        path_assignments: For each variable, the assignment for each tracked path
                         {var_id: tensor of shape (chi,) with assignments}
        var_dims: Dimension (cardinality) of each variable
    """
    truncation_info: List[Tuple[int, torch.Tensor]] = field(default_factory=list)
    path_values: Optional[torch.Tensor] = None
    path_assignments: Dict[int, torch.Tensor] = field(default_factory=dict)
    var_dims: Dict[int, int] = field(default_factory=dict)
    
    def record_truncation(
        self,
        site_idx: int,
        kept_indices: torch.Tensor,
        var_ids: Tuple[int, ...],
        var_dims: Tuple[int, ...],
    ):
        """Record truncation with variable assignment information.
        
        Args:
            site_idx: Index of the site being truncated
            kept_indices: Indices of configurations that were kept
            var_ids: Variable IDs involved in this site
            var_dims: Dimensions of each variable
        """
        self.truncation_info.append((site_idx, kept_indices.clone()))
        
        # Compute assignments for each kept configuration
        n_kept = kept_indices.numel()
        indices = kept_indices.clone()
        
        # Unravel indices to get assignment for each variable
        for var_id, dim in zip(reversed(var_ids), reversed(var_dims)):
            if var_id not in self.path_assignments:
                self.path_assignments[var_id] = indices % dim
                self.var_dims[var_id] = dim
            indices = indices // dim
    
    def record_values(self, values: torch.Tensor):
        """Record the values (log-probabilities) of tracked paths."""
        self.path_values = values.clone()
    
    def get_best_assignment(self) -> Dict[int, int]:
        """Recover the best (highest probability) assignment."""
        if self.path_values is None or len(self.path_assignments) == 0:
            return {}
        
        # Find path with maximum value
        best_idx = self.path_values.argmax().item()
        
        assignment = {}
        for var_id, values in self.path_assignments.items():
            if best_idx < values.numel():
                assignment[var_id] = int(values[best_idx].item())
            else:
                assignment[var_id] = 0  # Default
        
        return assignment
    
    def get_top_k_assignments(self, k: int = 5) -> List[Tuple[Dict[int, int], float]]:
        """Get the top-k most probable assignments.
        
        Args:
            k: Number of assignments to return
            
        Returns:
            List of (assignment, log_probability) tuples
        """
        if self.path_values is None or len(self.path_assignments) == 0:
            return []
        
        # Get top-k indices
        k = min(k, self.path_values.numel())
        top_values, top_indices = torch.topk(self.path_values, k)
        
        results = []
        for idx, val in zip(top_indices.tolist(), top_values.tolist()):
            assignment = {}
            for var_id, values in self.path_assignments.items():
                if idx < values.numel():
                    assignment[var_id] = int(values[idx].item())
            results.append((assignment, val))
        
        return results


# =============================================================================
# Syndrome Projection
# =============================================================================

def project_to_syndrome(
    assignment: Dict[int, int],
    H: "np.ndarray",
    syndrome: "np.ndarray",
    priors: "np.ndarray",
    max_flips: int = 100,
) -> Dict[int, int]:
    """Project assignment to satisfy syndrome constraint H @ x = s (mod 2).
    
    Uses greedy flipping based on prior probabilities to find a valid solution
    that satisfies the syndrome while staying close to the original assignment.
    
    Args:
        assignment: Current assignment {var_id: value}
        H: Parity check matrix (n_checks, n_vars)
        syndrome: Target syndrome (n_checks,)
        priors: Prior error probabilities (n_vars,)
        max_flips: Maximum number of bit flips allowed
        
    Returns:
        Modified assignment satisfying H @ x = s (mod 2)
    """
    import numpy as np
    
    n_vars = H.shape[1]
    
    # Convert assignment to numpy array (1-indexed to 0-indexed)
    x = np.zeros(n_vars, dtype=np.int32)
    for var_id, val in assignment.items():
        idx = var_id - 1  # Convert to 0-indexed
        if 0 <= idx < n_vars:
            x[idx] = val
    
    # Compute current syndrome
    current_syndrome = (H @ x) % 2
    
    # Check if already satisfied
    if np.array_equal(current_syndrome, syndrome.astype(np.int32)):
        return assignment
    
    # Find unsatisfied checks
    unsatisfied = np.where(current_syndrome != syndrome.astype(np.int32))[0]
    
    # Greedy flipping: flip variables that fix most unsatisfied checks
    # Prioritize flipping variables with higher prior (more likely to be errors)
    for _ in range(max_flips):
        if len(unsatisfied) == 0:
            break
        
        best_var = -1
        best_improvement = 0
        best_score = float('-inf')
        
        for var in range(n_vars):
            # How many unsatisfied checks would this flip fix?
            fixes = np.sum(H[unsatisfied, var])
            # How many satisfied checks would it break?
            satisfied = np.where(current_syndrome == syndrome.astype(np.int32))[0]
            breaks = np.sum(H[satisfied, var]) if len(satisfied) > 0 else 0
            
            improvement = fixes - breaks
            
            # Tie-break using prior probability
            if x[var] == 0:
                # Flipping 0->1: prefer higher prior
                score = np.log(priors[var] + 1e-10)
            else:
                # Flipping 1->0: prefer lower prior
                score = np.log(1 - priors[var] + 1e-10)
            
            if improvement > best_improvement or (improvement == best_improvement and score > best_score):
                best_var = var
                best_improvement = improvement
                best_score = score
        
        if best_var >= 0 and best_improvement > 0:
            x[best_var] = 1 - x[best_var]
            current_syndrome = (H @ x) % 2
            unsatisfied = np.where(current_syndrome != syndrome.astype(np.int32))[0]
        else:
            break
    
    # Convert back to assignment dict (0-indexed to 1-indexed)
    result = {}
    for i in range(n_vars):
        result[i + 1] = int(x[i])
    
    return result


# =============================================================================
# Simulated Annealing Refinement  
# =============================================================================

def refine_assignment_simulated_annealing(
    assignment: Dict[int, int],
    nodes: List,
    max_iterations: int = 1000,
    initial_temp: float = 1.0,
    final_temp: float = 0.01,
    var_dims: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[int, int], float]:
    """Refine assignment using simulated annealing.
    
    Unlike coordinate descent, this can escape local minima by accepting
    worse solutions with probability proportional to temperature.
    
    Args:
        assignment: Initial assignment
        nodes: List of TensorNode objects
        max_iterations: Number of iterations
        initial_temp: Starting temperature
        final_temp: Ending temperature
        var_dims: Variable dimensions
        
    Returns:
        Tuple of (refined_assignment, final_score)
    """
    import random
    import math
    
    if not assignment or not nodes:
        return assignment, float('-inf')
    
    if var_dims is None:
        var_dims = {v: 2 for v in assignment.keys()}
    
    current_assignment = assignment.copy()
    current_score = _compute_assignment_score(current_assignment, nodes)
    
    best_assignment = current_assignment.copy()
    best_score = current_score
    
    var_list = list(current_assignment.keys())
    n_vars = len(var_list)
    
    if n_vars == 0:
        return assignment, current_score
    
    # Temperature schedule
    temp_ratio = (final_temp / initial_temp) ** (1.0 / max_iterations)
    temp = initial_temp
    
    for iteration in range(max_iterations):
        # Pick random variable
        var_id = random.choice(var_list)
        dim = var_dims.get(var_id, 2)
        current_val = current_assignment[var_id]
        
        # Pick random new value
        if dim == 2:
            new_val = 1 - current_val
        else:
            new_val = random.randint(0, dim - 1)
            while new_val == current_val:
                new_val = random.randint(0, dim - 1)
        
        # Compute new score
        test_assignment = current_assignment.copy()
        test_assignment[var_id] = new_val
        test_score = _compute_assignment_score(test_assignment, nodes)
        
        # Accept or reject
        delta = test_score - current_score
        
        if delta > 0:
            # Always accept improvements
            accept = True
        else:
            # Accept worse with probability exp(delta/temp)
            accept_prob = math.exp(delta / temp) if temp > 0 else 0
            accept = random.random() < accept_prob
        
        if accept:
            current_assignment[var_id] = new_val
            current_score = test_score
            
            if current_score > best_score:
                best_assignment = current_assignment.copy()
                best_score = current_score
        
        # Cool down
        temp *= temp_ratio
    
    return best_assignment, best_score


# =============================================================================
# Iterative Refinement
# =============================================================================

def refine_assignment_local_search(
    assignment: Dict[int, int],
    nodes: List,
    max_iterations: int = 100,
    var_dims: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[int, int], float]:
    """Refine an approximate assignment using local search.
    
    Iteratively flips single variables to improve the log-probability.
    This is a greedy local search that can escape local minima by
    accepting moves that don't decrease the score.
    
    Args:
        assignment: Initial assignment to refine {var_id: value}
        nodes: List of TensorNode objects
        max_iterations: Maximum number of iterations
        var_dims: Variable dimensions (default: binary)
        
    Returns:
        Tuple of (refined_assignment, final_score)
    """
    if not assignment or not nodes:
        return assignment, float('-inf')
    
    if var_dims is None:
        var_dims = {v: 2 for v in assignment.keys()}
    
    # Compute initial score
    current_assignment = assignment.copy()
    current_score = _compute_assignment_score(current_assignment, nodes)
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try flipping each variable
        for var_id in list(current_assignment.keys()):
            dim = var_dims.get(var_id, 2)
            current_val = current_assignment[var_id]
            
            # Try each alternative value
            for new_val in range(dim):
                if new_val == current_val:
                    continue
                
                # Compute score with this variable flipped
                test_assignment = current_assignment.copy()
                test_assignment[var_id] = new_val
                test_score = _compute_assignment_score(test_assignment, nodes)
                
                if test_score > current_score:
                    current_assignment[var_id] = new_val
                    current_score = test_score
                    improved = True
                    break  # Move to next variable
    
    return current_assignment, current_score


def _compute_assignment_score(
    assignment: Dict[int, int],
    nodes: List,
) -> float:
    """Compute the log-probability score for an assignment.
    
    Args:
        assignment: Variable assignment {var_id: value}
        nodes: List of TensorNode objects
        
    Returns:
        Log-probability score (sum of log-factors)
    """
    total_score = 0.0
    
    for node in nodes:
        # Get indices into this factor's tensor
        indices = []
        for var in node.vars:
            if var in assignment:
                indices.append(assignment[var])
            else:
                # Variable not in assignment - use 0
                indices.append(0)
        
        if indices:
            try:
                value = node.values[tuple(indices)].item()
                total_score += value
            except (IndexError, RuntimeError):
                # Index out of bounds - assign very negative score
                total_score += -1e10
    
    return total_score


def refine_assignment_coordinate_descent(
    assignment: Dict[int, int],
    nodes: List,
    max_sweeps: int = 10,
    var_dims: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[int, int], float]:
    """Refine assignment using coordinate descent.
    
    For each variable, finds the optimal value given all other variables fixed.
    Sweeps through all variables until convergence or max_sweeps reached.
    
    Args:
        assignment: Initial assignment
        nodes: List of TensorNode objects  
        max_sweeps: Maximum number of full sweeps
        var_dims: Variable dimensions
        
    Returns:
        Tuple of (refined_assignment, final_score)
    """
    if not assignment or not nodes:
        return assignment, float('-inf')
    
    if var_dims is None:
        var_dims = {v: 2 for v in assignment.keys()}
    
    current_assignment = assignment.copy()
    var_list = list(current_assignment.keys())
    
    for sweep in range(max_sweeps):
        changed = False
        
        for var_id in var_list:
            dim = var_dims.get(var_id, 2)
            best_val = current_assignment[var_id]
            best_score = _compute_assignment_score(current_assignment, nodes)
            
            for val in range(dim):
                if val == current_assignment[var_id]:
                    continue
                    
                test_assignment = current_assignment.copy()
                test_assignment[var_id] = val
                score = _compute_assignment_score(test_assignment, nodes)
                
                if score > best_score:
                    best_val = val
                    best_score = score
            
            if best_val != current_assignment[var_id]:
                current_assignment[var_id] = best_val
                changed = True
        
        if not changed:
            break
    
    final_score = _compute_assignment_score(current_assignment, nodes)
    return current_assignment, final_score


# =============================================================================
# Assignment Tracking Utilities
# =============================================================================

def track_tensor_assignment(
    tensor: torch.Tensor,
    var_ids: Tuple[int, ...],
    chi: int,
    backpointer: ApproximateBackpointer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Track assignments for a tensor being incorporated into the boundary.
    
    Keeps top-chi configurations and records which variable assignments they correspond to.
    
    Args:
        tensor: Flattened tensor of log-probabilities, shape (prod(var_dims),)
        var_ids: Variable IDs for this tensor
        chi: Maximum number of configurations to keep
        backpointer: Backpointer to record assignment info
        
    Returns:
        Tuple of (truncated_values, kept_indices)
    """
    n = tensor.numel()
    
    if n <= chi:
        # Keep all configurations
        indices = torch.arange(n, device=tensor.device)
        return tensor, indices
    
    # Keep top-chi configurations
    values, indices = torch.topk(tensor, chi)
    
    # Record in backpointer
    var_dims = tuple(2 for _ in var_ids)  # Assuming binary variables
    backpointer.record_truncation(
        site_idx=len(backpointer.truncation_info),
        kept_indices=indices,
        var_ids=var_ids,
        var_dims=var_dims,
    )
    backpointer.record_values(values)
    
    return values, indices


def combine_tracked_tensors(
    values1: torch.Tensor,
    indices1: torch.Tensor,
    vars1: Tuple[int, ...],
    values2: torch.Tensor,
    indices2: torch.Tensor,
    vars2: Tuple[int, ...],
    chi: int,
    backpointer: ApproximateBackpointer,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """Combine two tracked tensors with outer product and truncation.
    
    Args:
        values1, indices1, vars1: First tensor's values, indices, and variables
        values2, indices2, vars2: Second tensor's values, indices, and variables
        chi: Maximum configurations to keep
        backpointer: Backpointer for recording
        
    Returns:
        Tuple of (combined_values, combined_indices, combined_vars)
    """
    # Outer product: combined[i,j] = values1[i] + values2[j]
    combined = values1.unsqueeze(1) + values2.unsqueeze(0)  # (n1, n2)
    combined_flat = combined.flatten()
    
    # Combined variables
    combined_vars = vars1 + tuple(v for v in vars2 if v not in vars1)
    
    n = combined_flat.numel()
    if n <= chi:
        combined_indices = torch.arange(n, device=combined_flat.device)
        return combined_flat, combined_indices, combined_vars
    
    # Top-chi
    top_values, top_flat_indices = torch.topk(combined_flat, chi)
    
    # Unravel to get which (i,j) pairs were kept
    n2 = values2.numel()
    kept_idx1 = top_flat_indices // n2
    kept_idx2 = top_flat_indices % n2
    
    # Map back to original indices
    orig_indices1 = indices1[kept_idx1]
    orig_indices2 = indices2[kept_idx2]
    
    # Record assignment information
    # For vars1, use orig_indices1; for vars2, use orig_indices2
    for var in vars1:
        if var not in backpointer.path_assignments:
            backpointer.path_assignments[var] = torch.zeros(chi, dtype=torch.long, device=combined.device)
            backpointer.var_dims[var] = 2
        # Compute assignment from index
        # This is simplified - assumes binary variables
        backpointer.path_assignments[var] = orig_indices1 % 2
    
    for var in vars2:
        if var not in vars1:
            if var not in backpointer.path_assignments:
                backpointer.path_assignments[var] = torch.zeros(chi, dtype=torch.long, device=combined.device)
                backpointer.var_dims[var] = 2
            backpointer.path_assignments[var] = orig_indices2 % 2
    
    backpointer.record_values(top_values)
    
    return top_values, top_flat_indices, combined_vars


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
    """Contract tensor network using boundary method with assignment tracking.
    
    This implements a simplified MPS-style boundary contraction:
    1. Process tensors in order, maintaining a boundary vector
    2. For each tensor, compute outer product with boundary
    3. Truncate to top-chi configurations
    4. Track assignments for the kept configurations
    
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
    
    # Sort tensors by first variable (simple ordering heuristic)
    indexed_tensors = list(enumerate(zip(tensors, vars_list)))
    indexed_tensors.sort(key=lambda x: min(x[1][1]) if x[1][1] else 0)
    
    # Initialize boundary from first tensor
    idx, (tensor, vars_tuple) = indexed_tensors[0]
    tensor_flat = tensor.flatten()
    
    if tensor_flat.numel() <= chi:
        boundary_values = tensor_flat
        boundary_indices = torch.arange(tensor_flat.numel(), device=tensor.device)
    else:
        boundary_values, boundary_indices = torch.topk(tensor_flat, chi)
    
    # Track assignments for first tensor
    if track_assignment:
        _record_assignments(backpointer, boundary_indices, vars_tuple, tensor.shape)
        backpointer.record_values(boundary_values)
    
    boundary_vars = set(vars_tuple)
    max_chi_used = boundary_values.numel()
    
    # Process remaining tensors
    for idx, (tensor, vars_tuple) in indexed_tensors[1:]:
        tensor_flat = tensor.flatten()
        tensor_vars = set(vars_tuple)
        
        # Outer product: combined[i,j] = boundary[i] + tensor[j]
        combined = boundary_values.unsqueeze(1) + tensor_flat.unsqueeze(0)
        combined_flat = combined.flatten()
        
        n_boundary = boundary_values.numel()
        n_tensor = tensor_flat.numel()
        
        # Truncate to top-chi
        if combined_flat.numel() <= chi:
            new_values = combined_flat
            new_flat_indices = torch.arange(combined_flat.numel(), device=tensor.device)
        else:
            new_values, new_flat_indices = torch.topk(combined_flat, chi)
        
        # Track which configurations were kept
        if track_assignment:
            boundary_idx = new_flat_indices // n_tensor
            tensor_idx = new_flat_indices % n_tensor
            
            # Update existing variable assignments
            for var in backpointer.path_assignments:
                old_vals = backpointer.path_assignments[var]
                if boundary_idx.max() < old_vals.numel():
                    backpointer.path_assignments[var] = old_vals[boundary_idx]
            
            # Add new variable assignments
            _record_assignments(backpointer, tensor_idx, vars_tuple, tensor.shape)
            backpointer.record_values(new_values)
        
        boundary_values = new_values
        boundary_vars.update(tensor_vars)
        max_chi_used = max(max_chi_used, boundary_values.numel())
    
    # Final value is the maximum
    final_value = float(boundary_values.max().item())
    
    # Create a simple MPS representation for compatibility
    mps = TropicalMPS(
        sites=[boundary_values.reshape(1, boundary_values.numel(), 1)],
        physical_dims=[boundary_values.numel()],
        chi=chi
    )
    
    return BoundaryContractionResult(
        value=final_value,
        mps=mps,
        backpointer=backpointer,
        chi_used=max_chi_used
    )


def _record_assignments(
    backpointer: ApproximateBackpointer,
    indices: torch.Tensor,
    vars_tuple: Tuple[int, ...],
    shape: torch.Size,
) -> None:
    """Record variable assignments for given indices.
    
    Args:
        backpointer: Backpointer to update
        indices: Flat indices into the tensor
        vars_tuple: Variables of the tensor
        shape: Shape of the tensor
    """
    if not vars_tuple:
        return
    
    n_configs = indices.numel()
    
    # Unravel flat indices to per-variable assignments
    remaining = indices.clone()
    
    for i, var in enumerate(reversed(vars_tuple)):
        dim = shape[len(vars_tuple) - 1 - i] if i < len(shape) else 2
        assignments = remaining % dim
        remaining = remaining // dim
        
        # Store in backpointer (may overwrite if variable already exists)
        backpointer.path_assignments[var] = assignments
        backpointer.var_dims[var] = dim
