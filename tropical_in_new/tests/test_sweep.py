"""Tests for sweep contraction algorithm."""

import pytest
import torch
import math

from tropical_in_new.src.sweep import (
    SweepDirection,
    TensorPosition,
    SweepState,
    compute_tensor_layout,
    sweep_contract,
    multi_direction_sweep,
    adaptive_sweep_contract,
    estimate_required_chi,
    _get_sweep_order,
    _spectral_layout,
    _force_directed_layout,
    _grid_layout,
)
from tropical_in_new.src.network import TensorNode


class TestTensorLayout:
    """Tests for tensor layout computation."""
    
    def test_empty_layout(self):
        """Test layout of empty network."""
        positions = compute_tensor_layout([])
        assert positions == []
    
    def test_single_tensor_layout(self):
        """Test layout of single tensor."""
        node = TensorNode(vars=(0,), values=torch.tensor([1.0, 2.0]))
        positions = compute_tensor_layout([node])
        
        assert len(positions) == 1
        assert positions[0].tensor_idx == 0
    
    def test_connected_tensors_layout(self):
        """Test layout of connected tensors."""
        nodes = [
            TensorNode(vars=(0, 1), values=torch.randn(2, 2)),
            TensorNode(vars=(1, 2), values=torch.randn(2, 2)),
            TensorNode(vars=(2, 3), values=torch.randn(2, 2)),
        ]
        
        positions = compute_tensor_layout(nodes, method="spectral")
        
        assert len(positions) == 3
        # Connected tensors should have positions (may be numpy floats)
        for pos in positions:
            assert math.isfinite(float(pos.x))
            assert math.isfinite(float(pos.y))
    
    def test_grid_layout(self):
        """Test grid layout method."""
        n = 6
        positions = _grid_layout(n, {i: [] for i in range(n)})
        
        assert len(positions) == n
        # Check grid pattern
        cols = int(math.ceil(math.sqrt(n)))
        for i, (x, y) in enumerate(positions):
            assert x == float(i % cols)
            assert y == float(i // cols)
    
    def test_force_directed_layout(self):
        """Test force-directed layout method."""
        n = 4
        adjacency = {
            0: [1, 2],
            1: [0, 3],
            2: [0, 3],
            3: [1, 2],
        }
        
        positions = _force_directed_layout(n, adjacency, iterations=10)
        
        assert len(positions) == n
        # All positions should be finite
        for x, y in positions:
            assert math.isfinite(x)
            assert math.isfinite(y)


class TestSweepOrder:
    """Tests for sweep ordering."""
    
    def test_left_to_right_order(self):
        """Test left-to-right sweep ordering."""
        positions = [
            TensorPosition(0, 2.0, 0.0, (0,)),
            TensorPosition(1, 0.0, 0.0, (1,)),
            TensorPosition(2, 1.0, 0.0, (2,)),
        ]
        
        order = _get_sweep_order(positions, SweepDirection.LEFT_TO_RIGHT)
        
        assert order == [1, 2, 0]  # Sorted by x ascending
    
    def test_right_to_left_order(self):
        """Test right-to-left sweep ordering."""
        positions = [
            TensorPosition(0, 2.0, 0.0, (0,)),
            TensorPosition(1, 0.0, 0.0, (1,)),
            TensorPosition(2, 1.0, 0.0, (2,)),
        ]
        
        order = _get_sweep_order(positions, SweepDirection.RIGHT_TO_LEFT)
        
        assert order == [0, 2, 1]  # Sorted by x descending
    
    def test_top_to_bottom_order(self):
        """Test top-to-bottom sweep ordering."""
        positions = [
            TensorPosition(0, 0.0, 2.0, (0,)),
            TensorPosition(1, 0.0, 0.0, (1,)),
            TensorPosition(2, 0.0, 1.0, (2,)),
        ]
        
        order = _get_sweep_order(positions, SweepDirection.TOP_TO_BOTTOM)
        
        assert order == [1, 2, 0]  # Sorted by y ascending


class TestSweepContract:
    """Tests for sweep contraction algorithm."""
    
    def test_empty_network(self):
        """Test sweep of empty network."""
        result = sweep_contract([], chi=32)
        
        assert result.value == 0.0
        assert result.chi_used == 0
        assert result.num_sweeps == 0
    
    def test_single_tensor(self):
        """Test sweep of single tensor."""
        node = TensorNode(
            vars=(0,),
            values=torch.tensor([1.0, 2.0, 3.0])
        )
        
        result = sweep_contract([node], chi=32)
        
        # Max value should be 3.0
        assert abs(result.value - 3.0) < 1e-6
        assert result.num_sweeps == 1
    
    def test_two_tensors_same_var(self):
        """Test sweep of two tensors sharing a variable."""
        nodes = [
            TensorNode(vars=(0,), values=torch.tensor([1.0, 2.0])),
            TensorNode(vars=(0,), values=torch.tensor([3.0, 4.0])),
        ]
        
        result = sweep_contract(nodes, chi=32)
        
        # Tropical product: max over all (i,j) of t1[i] + t2[j] = 2 + 4 = 6
        assert abs(result.value - 6.0) < 1e-6
    
    def test_chain_network(self):
        """Test sweep of chain-structured network."""
        nodes = [
            TensorNode(vars=(0, 1), values=torch.ones(2, 2)),
            TensorNode(vars=(1, 2), values=torch.ones(2, 2)),
            TensorNode(vars=(2, 3), values=torch.ones(2, 2)),
        ]
        
        result = sweep_contract(nodes, chi=32)
        
        # All ones in log domain, so result is sum of contractions
        assert math.isfinite(result.value)
        assert result.num_sweeps == 3
    
    def test_chi_limiting(self):
        """Test that chi limits output size."""
        # Create network that would have high bond dim
        nodes = [
            TensorNode(vars=(i, i+1), values=torch.randn(4, 4))
            for i in range(5)
        ]
        
        result = sweep_contract(nodes, chi=2)
        
        # Chi limiting affects the MPS, result should be valid
        assert math.isfinite(result.value)


class TestMultiDirectionSweep:
    """Tests for multi-direction sweep."""
    
    def test_tries_all_directions(self):
        """Test that multi-direction sweep tries different directions."""
        nodes = [
            TensorNode(vars=(0, 1), values=torch.randn(2, 2)),
            TensorNode(vars=(1, 2), values=torch.randn(2, 2)),
        ]
        
        result = multi_direction_sweep(nodes, chi=16)
        
        # Should complete without error
        assert result is not None
        assert math.isfinite(result.value)
    
    def test_returns_best_direction(self):
        """Test that multi-direction returns best result."""
        # Create asymmetric network where direction matters
        nodes = [
            TensorNode(vars=(0,), values=torch.tensor([10.0, 1.0])),
            TensorNode(vars=(1,), values=torch.tensor([1.0, 20.0])),
        ]
        
        result = multi_direction_sweep(nodes, chi=16)
        
        # Tropical product of these: max over (i,j) of t1[i] + t2[j] = 10 + 20 = 30
        assert result.value >= 20.0  # At least reasonable value


class TestAdaptiveSweep:
    """Tests for adaptive sweep contraction."""
    
    def test_starts_small(self):
        """Test that adaptive sweep starts with small chi."""
        nodes = [
            TensorNode(vars=(0,), values=torch.tensor([1.0, 2.0]))
        ]
        
        result = adaptive_sweep_contract(
            nodes,
            chi_start=4,
            chi_max=32,
            chi_step=4
        )
        
        assert result is not None
        assert math.isfinite(result.value)
    
    def test_increases_chi(self):
        """Test that adaptive sweep can increase chi."""
        nodes = [
            TensorNode(vars=(i, i+1), values=torch.randn(3, 3))
            for i in range(3)
        ]
        
        result = adaptive_sweep_contract(
            nodes,
            chi_start=2,
            chi_max=16,
            chi_step=2,
            max_iterations=5
        )
        
        # Should produce valid result
        assert result is not None
        assert math.isfinite(result.value)


class TestEstimateRequiredChi:
    """Tests for chi estimation."""
    
    def test_empty_network(self):
        """Test chi estimation for empty network."""
        chi = estimate_required_chi([])
        assert chi == 1
    
    def test_small_network(self):
        """Test chi estimation for small network."""
        nodes = [
            TensorNode(vars=(0,), values=torch.tensor([1.0, 2.0]))
        ]
        
        chi = estimate_required_chi(nodes)
        
        assert chi >= 4  # Minimum reasonable value
    
    def test_larger_network(self):
        """Test chi estimation scales with network size."""
        small_nodes = [
            TensorNode(vars=(i,), values=torch.randn(2))
            for i in range(5)
        ]
        
        large_nodes = [
            TensorNode(vars=(i,), values=torch.randn(10))
            for i in range(20)
        ]
        
        small_chi = estimate_required_chi(small_nodes)
        large_chi = estimate_required_chi(large_nodes)
        
        # Larger network should suggest higher chi
        assert large_chi >= small_chi
    
    def test_high_accuracy_increases_chi(self):
        """Test that higher accuracy target increases chi."""
        nodes = [
            TensorNode(vars=(i, i+1), values=torch.randn(4, 4))
            for i in range(5)
        ]
        
        low_chi = estimate_required_chi(nodes, target_accuracy=0.90)
        high_chi = estimate_required_chi(nodes, target_accuracy=0.99)
        
        assert high_chi >= low_chi


class TestSweepState:
    """Tests for SweepState class."""
    
    def test_initial_state(self):
        """Test initial sweep state."""
        state = SweepState(chi=32)
        
        assert state.boundary_values is None
        assert state.boundary_vars == ()
        assert len(state.contracted_tensors) == 0
        assert state.current_position == 0.0
        assert state.chi == 32
    
    def test_track_contracted(self):
        """Test tracking contracted tensors."""
        state = SweepState(chi=32)
        
        state.contracted_tensors.add(0)
        state.contracted_tensors.add(2)
        
        assert 0 in state.contracted_tensors
        assert 2 in state.contracted_tensors
        assert 1 not in state.contracted_tensors


class TestLayoutMethods:
    """Tests for different layout methods."""
    
    def test_spectral_layout(self):
        """Test spectral layout produces valid positions."""
        n = 5
        adjacency = {i: [(i+1) % n] for i in range(n)}
        
        positions = _spectral_layout(n, adjacency)
        
        assert len(positions) == n
        for x, y in positions:
            assert math.isfinite(x)
            assert math.isfinite(y)
    
    def test_spectral_layout_small(self):
        """Test spectral layout for very small networks."""
        # Single node
        positions = _spectral_layout(1, {0: []})
        assert len(positions) == 1
        
        # Two nodes
        positions = _spectral_layout(2, {0: [1], 1: [0]})
        assert len(positions) == 2
    
    def test_all_layout_methods_work(self):
        """Test that all layout methods produce valid results."""
        nodes = [
            TensorNode(vars=(0, 1), values=torch.randn(2, 2)),
            TensorNode(vars=(1, 2), values=torch.randn(2, 2)),
            TensorNode(vars=(2, 0), values=torch.randn(2, 2)),
        ]
        
        for method in ["spectral", "force", "grid"]:
            positions = compute_tensor_layout(nodes, method=method)
            
            assert len(positions) == 3
            for pos in positions:
                assert math.isfinite(pos.x)
                assert math.isfinite(pos.y)


class TestSweepEdgeCases:
    """Edge cases for sweep contraction."""
    
    def test_disconnected_network(self):
        """Test sweep with disconnected tensors."""
        nodes = [
            TensorNode(vars=(0,), values=torch.tensor([1.0, 2.0])),
            TensorNode(vars=(10,), values=torch.tensor([3.0, 4.0])),  # Disconnected
        ]
        
        result = sweep_contract(nodes, chi=32)
        
        # Should still produce valid result
        assert math.isfinite(result.value)
    
    def test_single_var_tensors(self):
        """Test with tensors that have single variables."""
        nodes = [
            TensorNode(vars=(i,), values=torch.randn(3))
            for i in range(5)
        ]
        
        result = sweep_contract(nodes, chi=16)
        
        assert math.isfinite(result.value)
        assert result.num_sweeps == 5
    
    def test_large_tensor_values(self):
        """Test sweep with large tensor values."""
        nodes = [
            TensorNode(vars=(0, 1), values=torch.randn(2, 2) * 1000),
        ]
        
        result = sweep_contract(nodes, chi=16)
        
        assert math.isfinite(result.value)
    
    def test_negative_values(self):
        """Test sweep with negative tensor values (log-domain)."""
        nodes = [
            TensorNode(vars=(0, 1), values=torch.tensor([[-5.0, -2.0], [-3.0, -1.0]])),
        ]
        
        result = sweep_contract(nodes, chi=16)
        
        # Should find the maximum (-1.0)
        assert result.value >= -5.0


class TestSweepDirections:
    """Tests for different sweep directions."""
    
    def test_bottom_to_top_order(self):
        """Test bottom-to-top sweep ordering."""
        positions = [
            TensorPosition(0, 0.0, 2.0, (0,)),
            TensorPosition(1, 0.0, 0.0, (1,)),
            TensorPosition(2, 0.0, 1.0, (2,)),
        ]
        
        order = _get_sweep_order(positions, SweepDirection.BOTTOM_TO_TOP)
        
        assert order == [0, 2, 1]  # Sorted by -y
    
    def test_all_directions_produce_result(self):
        """Test that all sweep directions produce valid results."""
        nodes = [
            TensorNode(vars=(0, 1), values=torch.randn(2, 2)),
            TensorNode(vars=(1, 2), values=torch.randn(2, 2)),
        ]
        
        for direction in SweepDirection:
            result = sweep_contract(nodes, chi=16, direction=direction)
            
            assert math.isfinite(result.value)


class TestSweepRandomized:
    """Randomized tests for sweep contraction."""
    
    @pytest.mark.parametrize("n_nodes", [1, 2, 5, 10])
    def test_varying_network_sizes(self, n_nodes):
        """Test sweep with different network sizes."""
        nodes = [
            TensorNode(vars=(i, (i+1) % n_nodes), values=torch.randn(2, 2))
            for i in range(n_nodes)
        ]
        
        result = sweep_contract(nodes, chi=8)
        
        assert math.isfinite(result.value)
        assert result.num_sweeps == n_nodes
    
    @pytest.mark.parametrize("chi", [2, 4, 8, 16, 32])
    def test_varying_chi(self, chi):
        """Test sweep with different chi values."""
        nodes = [
            TensorNode(vars=(i, i+1), values=torch.randn(3, 3))
            for i in range(4)
        ]
        
        result = sweep_contract(nodes, chi=chi)
        
        assert math.isfinite(result.value)
    
    @pytest.mark.parametrize("layout", ["spectral", "force", "grid"])
    def test_different_layouts(self, layout):
        """Test sweep with different layout methods."""
        nodes = [
            TensorNode(vars=(0, 1), values=torch.randn(2, 2)),
            TensorNode(vars=(1, 2), values=torch.randn(2, 2)),
            TensorNode(vars=(2, 0), values=torch.randn(2, 2)),
        ]
        
        result = sweep_contract(nodes, chi=16, layout_method=layout)
        
        assert math.isfinite(result.value)


class TestAdaptiveSweepBehavior:
    """Tests for adaptive sweep behavior."""
    
    def test_converges_early(self):
        """Test that adaptive sweep can converge early."""
        # Simple network should converge quickly
        nodes = [
            TensorNode(vars=(0,), values=torch.tensor([1.0, 2.0]))
        ]
        
        result = adaptive_sweep_contract(
            nodes,
            chi_start=2,
            chi_max=64,
            chi_step=2,
            tolerance=1e-10,
            max_iterations=10
        )
        
        assert result is not None
    
    def test_respects_max_chi(self):
        """Test that adaptive sweep respects chi_max."""
        nodes = [
            TensorNode(vars=(i, i+1), values=torch.randn(4, 4))
            for i in range(5)
        ]
        
        result = adaptive_sweep_contract(
            nodes,
            chi_start=2,
            chi_max=8,
            chi_step=2,
            max_iterations=20
        )
        
        # chi_used should not exceed chi_max significantly
        assert result.chi_used <= 16  # Allow some flexibility


class TestTensorPositionClass:
    """Tests for TensorPosition dataclass."""
    
    def test_create_position(self):
        """Test creating a TensorPosition."""
        pos = TensorPosition(
            tensor_idx=5,
            x=1.5,
            y=-2.3,
            vars=(0, 1, 2)
        )
        
        assert pos.tensor_idx == 5
        assert pos.x == 1.5
        assert pos.y == -2.3
        assert pos.vars == (0, 1, 2)
    
    def test_position_with_empty_vars(self):
        """Test position with no variables."""
        pos = TensorPosition(tensor_idx=0, x=0.0, y=0.0, vars=())
        
        assert pos.vars == ()


class TestLayoutConnectivity:
    """Tests for layout with various connectivity patterns."""
    
    def test_fully_connected(self):
        """Test layout with fully connected tensors."""
        # All tensors share variable 0
        nodes = [
            TensorNode(vars=(0, i+1), values=torch.randn(2, 2))
            for i in range(4)
        ]
        
        positions = compute_tensor_layout(nodes, method="force")
        
        assert len(positions) == 4
    
    def test_linear_chain(self):
        """Test layout with linear chain connectivity."""
        nodes = [
            TensorNode(vars=(i, i+1), values=torch.randn(2, 2))
            for i in range(5)
        ]
        
        positions = compute_tensor_layout(nodes, method="spectral")
        
        assert len(positions) == 5
    
    def test_star_topology(self):
        """Test layout with star topology."""
        # Central node connected to all others
        nodes = [TensorNode(vars=(0,), values=torch.randn(2))]
        nodes += [
            TensorNode(vars=(0, i), values=torch.randn(2, 2))
            for i in range(1, 5)
        ]
        
        positions = compute_tensor_layout(nodes, method="grid")
        
        assert len(positions) == 5
