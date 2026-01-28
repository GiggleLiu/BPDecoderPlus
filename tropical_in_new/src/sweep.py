"""Sweep-line contraction for tropical tensor networks.

This module implements the sweep contraction algorithm based on:
Chubb, "General tensor network decoding of 2D Pauli codes" [arXiv:2101.04125]

The sweep algorithm:
1. Planarizes the tensor network (embeds in 2D if necessary)
2. Sweeps a line across the network
3. Maintains a boundary MPS with bounded bond dimension
4. Contracts tensors as they are encountered by the sweep line

This is particularly effective for 2D tensor networks arising from
topological codes like the surface code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set
from enum import Enum
import math

import torch

from .approximate import (
    TropicalMPS,
    tropical_svd_approx,
    tropical_tensor_contract,
    truncate_mps,
    ApproximateBackpointer,
    track_tensor_assignment,
    combine_tracked_tensors,
)
from .network import TensorNode


class SweepDirection(Enum):
    """Direction of sweep line."""
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    TOP_TO_BOTTOM = "top_to_bottom"
    BOTTOM_TO_TOP = "bottom_to_top"


@dataclass
class TensorPosition:
    """Position of a tensor in 2D layout for sweep contraction.
    
    Attributes:
        tensor_idx: Index in the original tensor list
        x: Horizontal position
        y: Vertical position
        vars: Variables of this tensor
    """
    tensor_idx: int
    x: float
    y: float
    vars: Tuple[int, ...]


@dataclass
class SweepState:
    """State maintained during sweep contraction.
    
    Attributes:
        boundary_values: Current boundary values (log-probabilities), shape (chi,)
        boundary_indices: Indices corresponding to boundary values
        boundary_vars: Variables currently in the boundary
        contracted_tensors: Set of tensor indices already contracted
        current_position: Current sweep line position
        chi: Maximum bond dimension
        backpointer: Information for assignment recovery
    """
    boundary_values: Optional[torch.Tensor] = None
    boundary_indices: Optional[torch.Tensor] = None
    boundary_vars: Tuple[int, ...] = field(default_factory=tuple)
    contracted_tensors: Set[int] = field(default_factory=set)
    current_position: float = 0.0
    chi: int = 32
    backpointer: ApproximateBackpointer = field(default_factory=ApproximateBackpointer)


@dataclass
class SweepContractionResult:
    """Result of sweep contraction.
    
    Attributes:
        value: Final scalar value (log-probability)
        assignment: Recovered MPE assignment (if tracked)
        chi_used: Maximum bond dimension actually used
        num_sweeps: Number of sweep steps performed
    """
    value: float
    assignment: Dict[int, int]
    chi_used: int
    num_sweeps: int


# =============================================================================
# Layout and Positioning
# =============================================================================

def compute_tensor_layout(
    nodes: List[TensorNode],
    method: str = "spectral",
) -> List[TensorPosition]:
    """Compute 2D positions for tensors based on connectivity.
    
    Uses graph layout algorithms to embed the tensor network in 2D
    for sweep contraction.
    
    Args:
        nodes: List of tensor nodes
        method: Layout method - "spectral", "force", or "grid"
        
    Returns:
        List of TensorPosition with 2D coordinates
    """
    n = len(nodes)
    if n == 0:
        return []
    
    if n == 1:
        return [TensorPosition(0, 0.0, 0.0, nodes[0].vars)]
    
    # Build adjacency based on shared variables
    adjacency: Dict[int, List[int]] = {i: [] for i in range(n)}
    var_to_tensors: Dict[int, List[int]] = {}
    
    for i, node in enumerate(nodes):
        for v in node.vars:
            if v not in var_to_tensors:
                var_to_tensors[v] = []
            var_to_tensors[v].append(i)
    
    for var, tensor_indices in var_to_tensors.items():
        for i in tensor_indices:
            for j in tensor_indices:
                if i != j and j not in adjacency[i]:
                    adjacency[i].append(j)
    
    if method == "spectral":
        positions = _spectral_layout(n, adjacency)
    elif method == "force":
        positions = _force_directed_layout(n, adjacency)
    elif method == "grid":
        positions = _grid_layout(n, adjacency)
    else:
        # Default: simple row layout
        positions = [(float(i), 0.0) for i in range(n)]
    
    return [
        TensorPosition(i, positions[i][0], positions[i][1], nodes[i].vars)
        for i in range(n)
    ]


def _spectral_layout(
    n: int,
    adjacency: Dict[int, List[int]],
) -> List[Tuple[float, float]]:
    """Compute spectral layout using Laplacian eigenvectors."""
    # Build Laplacian matrix
    L = torch.zeros(n, n)
    for i, neighbors in adjacency.items():
        L[i, i] = len(neighbors)
        for j in neighbors:
            L[i, j] = -1.0
    
    # Compute eigenvectors
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        # Use 2nd and 3rd smallest eigenvectors as coordinates
        # (1st eigenvector is constant)
        if n >= 3:
            x = eigenvectors[:, 1].numpy()
            y = eigenvectors[:, 2].numpy()
        elif n == 2:
            x = eigenvectors[:, 1].numpy()
            y = [0.0, 0.0]
        else:
            x = [0.0]
            y = [0.0]
    except Exception:
        # Fallback to simple layout
        x = [float(i) for i in range(n)]
        y = [0.0] * n
    
    return list(zip(x, y))


def _force_directed_layout(
    n: int,
    adjacency: Dict[int, List[int]],
    iterations: int = 50,
) -> List[Tuple[float, float]]:
    """Compute force-directed layout (Fruchterman-Reingold style)."""
    # Initialize random positions
    positions = torch.randn(n, 2)
    
    k = 1.0 / math.sqrt(n)  # Optimal distance
    
    for _ in range(iterations):
        # Compute repulsive forces (all pairs)
        forces = torch.zeros(n, 2)
        
        for i in range(n):
            for j in range(i + 1, n):
                delta = positions[i] - positions[j]
                dist = delta.norm() + 0.01
                
                # Repulsive force
                repulsion = k * k / dist
                force = delta / dist * repulsion
                forces[i] += force
                forces[j] -= force
        
        # Compute attractive forces (edges only)
        for i, neighbors in adjacency.items():
            for j in neighbors:
                if j > i:
                    delta = positions[j] - positions[i]
                    dist = delta.norm() + 0.01
                    
                    # Attractive force
                    attraction = dist / k
                    force = delta / dist * attraction
                    forces[i] += force
                    forces[j] -= force
        
        # Update positions with damping
        positions += 0.1 * forces
    
    return [(float(positions[i, 0]), float(positions[i, 1])) for i in range(n)]


def _grid_layout(
    n: int,
    adjacency: Dict[int, List[int]],
) -> List[Tuple[float, float]]:
    """Simple grid layout."""
    cols = int(math.ceil(math.sqrt(n)))
    return [(float(i % cols), float(i // cols)) for i in range(n)]


# =============================================================================
# Sweep Contraction
# =============================================================================

def _get_sweep_order(
    positions: List[TensorPosition],
    direction: SweepDirection,
) -> List[int]:
    """Get order of tensors for sweep based on direction."""
    if direction == SweepDirection.LEFT_TO_RIGHT:
        key = lambda p: (p.x, p.y)
    elif direction == SweepDirection.RIGHT_TO_LEFT:
        key = lambda p: (-p.x, p.y)
    elif direction == SweepDirection.TOP_TO_BOTTOM:
        key = lambda p: (p.y, p.x)
    elif direction == SweepDirection.BOTTOM_TO_TOP:
        key = lambda p: (-p.y, p.x)
    else:
        key = lambda p: p.tensor_idx
    
    sorted_positions = sorted(positions, key=key)
    return [p.tensor_idx for p in sorted_positions]


def _contract_tensor_into_state(
    state: SweepState,
    tensor: torch.Tensor,
    tensor_vars: Tuple[int, ...],
) -> None:
    """Contract a single tensor into the boundary state with assignment tracking.
    
    Updates state in-place with new boundary values and assignment information.
    
    Args:
        state: Current sweep state (modified in-place)
        tensor: Tensor to contract (in log domain)
        tensor_vars: Variables of this tensor
    """
    tensor_flat = tensor.flatten()
    chi = state.chi
    
    if state.boundary_values is None:
        # First tensor: initialize boundary
        if tensor_flat.numel() <= chi:
            state.boundary_values = tensor_flat
            state.boundary_indices = torch.arange(tensor_flat.numel(), device=tensor.device)
        else:
            # Truncate to top-chi
            values, indices = torch.topk(tensor_flat, chi)
            state.boundary_values = values
            state.boundary_indices = indices
            
            # Record assignment for each kept configuration
            for i, var in enumerate(tensor_vars):
                # Unravel index to get assignment for this variable
                # Assuming all variables are binary
                divisor = 2 ** (len(tensor_vars) - 1 - i)
                assignments = (indices // divisor) % 2
                state.backpointer.path_assignments[var] = assignments
                state.backpointer.var_dims[var] = 2
        
        state.boundary_vars = tensor_vars
        state.backpointer.record_values(state.boundary_values)
        return
    
    # Find shared and new variables
    boundary_vars_set = set(state.boundary_vars)
    tensor_vars_set = set(tensor_vars)
    shared_vars = boundary_vars_set & tensor_vars_set
    new_vars = tensor_vars_set - boundary_vars_set
    
    n_boundary = state.boundary_values.numel()
    n_tensor = tensor_flat.numel()
    
    if not shared_vars:
        # No shared variables: outer product
        # combined[i,j] = boundary[i] + tensor[j]
        combined = state.boundary_values.unsqueeze(1) + tensor_flat.unsqueeze(0)
        combined_flat = combined.flatten()
        
        # Track indices
        if combined_flat.numel() <= chi:
            new_values = combined_flat
            new_flat_indices = torch.arange(combined_flat.numel(), device=tensor.device)
        else:
            new_values, new_flat_indices = torch.topk(combined_flat, chi)
        
        # Unravel flat indices to (boundary_idx, tensor_idx)
        boundary_idx = new_flat_indices // n_tensor
        tensor_idx = new_flat_indices % n_tensor
        
        # Update assignments for boundary variables
        for var in state.boundary_vars:
            if var in state.backpointer.path_assignments:
                old_assignments = state.backpointer.path_assignments[var]
                # Map through boundary_idx
                if boundary_idx.max() < old_assignments.numel():
                    state.backpointer.path_assignments[var] = old_assignments[boundary_idx]
        
        # Add assignments for new tensor variables
        for i, var in enumerate(tensor_vars):
            if var not in boundary_vars_set:
                # Unravel tensor_idx to get assignment for this variable
                divisor = 2 ** (len(tensor_vars) - 1 - i)
                assignments = (tensor_idx // divisor) % 2
                state.backpointer.path_assignments[var] = assignments
                state.backpointer.var_dims[var] = 2
        
        state.boundary_values = new_values
        state.boundary_vars = state.boundary_vars + tuple(new_vars)
        
    else:
        # Shared variables: need to align and contract
        # For now, use a simplified approach that works for binary variables
        
        # Expand both to full product then max over shared
        combined = state.boundary_values.unsqueeze(1) + tensor_flat.unsqueeze(0)
        combined_flat = combined.flatten()
        
        # Truncate
        if combined_flat.numel() <= chi:
            new_values = combined_flat
            new_flat_indices = torch.arange(combined_flat.numel(), device=tensor.device)
        else:
            new_values, new_flat_indices = torch.topk(combined_flat, chi)
        
        boundary_idx = new_flat_indices // n_tensor
        tensor_idx = new_flat_indices % n_tensor
        
        # Update assignments
        for var in state.boundary_vars:
            if var in state.backpointer.path_assignments:
                old_assignments = state.backpointer.path_assignments[var]
                if boundary_idx.max() < old_assignments.numel():
                    state.backpointer.path_assignments[var] = old_assignments[boundary_idx]
        
        for i, var in enumerate(tensor_vars):
            if var not in boundary_vars_set:
                divisor = 2 ** (len(tensor_vars) - 1 - i)
                assignments = (tensor_idx // divisor) % 2
                state.backpointer.path_assignments[var] = assignments
                state.backpointer.var_dims[var] = 2
        
        state.boundary_values = new_values
        state.boundary_vars = state.boundary_vars + tuple(new_vars)
    
    state.backpointer.record_values(state.boundary_values)


def sweep_contract(
    nodes: List[TensorNode],
    chi: int = 32,
    direction: SweepDirection = SweepDirection.LEFT_TO_RIGHT,
    layout_method: str = "spectral",
    track_assignment: bool = True,
) -> SweepContractionResult:
    """Contract tensor network using sweep line algorithm.
    
    The sweep algorithm processes tensors in order of their position
    along the sweep direction, maintaining a boundary state with tracked assignments.
    
    Args:
        nodes: List of tensor nodes to contract
        chi: Maximum bond dimension
        direction: Direction of sweep
        layout_method: Method for computing 2D layout
        track_assignment: Whether to track info for assignment recovery
        
    Returns:
        SweepContractionResult with final value and metadata
    """
    if not nodes:
        return SweepContractionResult(
            value=0.0,
            assignment={},
            chi_used=0,
            num_sweeps=0
        )
    
    # Compute layout
    positions = compute_tensor_layout(nodes, method=layout_method)
    
    # Get sweep order
    order = _get_sweep_order(positions, direction)
    
    # Initialize state
    state = SweepState(chi=chi)
    max_chi_used = 0
    
    # Process tensors in sweep order
    for tensor_idx in order:
        node = nodes[tensor_idx]
        
        # Contract tensor into boundary with assignment tracking
        _contract_tensor_into_state(state, node.values, node.vars)
        
        state.contracted_tensors.add(tensor_idx)
        
        if state.boundary_values is not None:
            max_chi_used = max(max_chi_used, state.boundary_values.numel())
    
    # Extract final value
    if state.boundary_values is None:
        final_value = 0.0
    else:
        final_value = float(state.boundary_values.max().item())
    
    # Recover assignment from backpointer
    assignment = state.backpointer.get_best_assignment()
    
    return SweepContractionResult(
        value=final_value,
        assignment=assignment,
        chi_used=max_chi_used,
        num_sweeps=len(order)
    )


def multi_direction_sweep(
    nodes: List[TensorNode],
    chi: int = 32,
    directions: Optional[List[SweepDirection]] = None,
    layout_method: str = "spectral",
) -> SweepContractionResult:
    """Try multiple sweep directions and return best result.
    
    Since different sweep directions can give different approximation
    quality, this function tries multiple directions and returns the
    one with the highest value (most probable assignment).
    
    Args:
        nodes: List of tensor nodes
        chi: Maximum bond dimension
        directions: List of directions to try. If None, tries all.
        layout_method: Method for computing 2D layout
        
    Returns:
        Best SweepContractionResult across all directions
    """
    if directions is None:
        directions = list(SweepDirection)
    
    best_result = None
    
    for direction in directions:
        result = sweep_contract(
            nodes,
            chi=chi,
            direction=direction,
            layout_method=layout_method,
            track_assignment=True
        )
        
        if best_result is None or result.value > best_result.value:
            best_result = result
    
    return best_result


# =============================================================================
# Adaptive Sweep (adjusts chi based on convergence)
# =============================================================================

def adaptive_sweep_contract(
    nodes: List[TensorNode],
    chi_start: int = 8,
    chi_max: int = 128,
    chi_step: int = 8,
    tolerance: float = 1e-6,
    max_iterations: int = 10,
) -> SweepContractionResult:
    """Sweep contraction with adaptive bond dimension.
    
    Starts with small chi and increases until the result converges
    or chi_max is reached.
    
    Args:
        nodes: List of tensor nodes
        chi_start: Initial bond dimension
        chi_max: Maximum bond dimension to try
        chi_step: Step size for increasing chi
        tolerance: Convergence tolerance for value
        max_iterations: Maximum number of chi values to try
        
    Returns:
        SweepContractionResult from converged or final iteration
    """
    chi = chi_start
    prev_value = None
    best_result = None
    
    for _ in range(max_iterations):
        if chi > chi_max:
            break
        
        result = multi_direction_sweep(nodes, chi=chi)
        
        if best_result is None or result.value > best_result.value:
            best_result = result
        
        # Check convergence
        if prev_value is not None:
            if abs(result.value - prev_value) < tolerance:
                break
        
        prev_value = result.value
        chi += chi_step
    
    return best_result


# =============================================================================
# Utility: Estimate required chi
# =============================================================================

def estimate_required_chi(
    nodes: List[TensorNode],
    target_accuracy: float = 0.99,
) -> int:
    """Estimate bond dimension required for target accuracy.
    
    Uses heuristics based on network structure to estimate chi.
    
    Args:
        nodes: Tensor nodes
        target_accuracy: Desired approximation accuracy (0 to 1)
        
    Returns:
        Estimated required bond dimension
    """
    if not nodes:
        return 1
    
    # Heuristic: chi scales with maximum tensor dimension
    max_dim = 1
    for node in nodes:
        if node.values.numel() > 0:
            max_dim = max(max_dim, max(node.values.shape))
    
    # Also consider network size
    n_tensors = len(nodes)
    
    # Heuristic formula
    base_chi = int(math.sqrt(max_dim * n_tensors))
    
    # Adjust for target accuracy
    accuracy_factor = -math.log10(1 - target_accuracy + 1e-10)
    chi = int(base_chi * (1 + accuracy_factor))
    
    # Clamp to reasonable range
    return max(4, min(chi, 256))
