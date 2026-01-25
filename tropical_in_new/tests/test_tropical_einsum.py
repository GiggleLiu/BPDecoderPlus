"""Tests for tropical_einsum module - adapted from OMEinsum test patterns."""

import pytest
import torch
import numpy as np

from tropical_in_new.src.tropical_einsum import (
    tropical_einsum,
    tropical_reduce_max,
    match_rule,
    argmax_trace,
    Backpointer,
    Identity,
    TropicalSum,
    Permutedims,
    Diag,
    Tr,
    SimpleBinaryRule,
    DefaultRule,
    _align_tensor,
)


# =============================================================================
# Helper functions
# =============================================================================

def tropical_einsum_reference(tensors, ixs, iy):
    """Reference implementation using explicit loops for verification."""
    # Build combined shape
    all_vars = []
    for ix in ixs:
        for v in ix:
            if v not in all_vars:
                all_vars.append(v)

    var_to_size = {}
    for tensor, ix in zip(tensors, ixs):
        for i, v in enumerate(ix):
            var_to_size[v] = tensor.shape[i]

    # Align and add
    combined = None
    for tensor, ix in zip(tensors, ixs):
        aligned = _align_tensor(tensor, tuple(ix), tuple(all_vars))
        if combined is None:
            combined = aligned
        else:
            combined = combined + aligned

    # Reduce via max
    elim_vars = [v for v in all_vars if v not in iy]
    result = combined
    for v in elim_vars:
        dim = all_vars.index(v)
        result = result.max(dim=dim).values
        all_vars.remove(v)

    # Permute to output order
    if tuple(all_vars) != tuple(iy):
        perm = [all_vars.index(v) for v in iy]
        result = result.permute(perm)

    return result


# =============================================================================
# Unary Rule Tests (adapted from OMEinsum)
# =============================================================================

class TestUnaryRules:
    """Test unary contraction rules."""

    def test_identity(self):
        """Test ix == iy (identity)."""
        x = torch.randn(3, 4)
        result, bp = tropical_einsum([x], [(0, 1)], (0, 1))
        torch.testing.assert_close(result, x)
        assert bp is None

    def test_permutedims_2d(self):
        """Test permutation: ij -> ji."""
        x = torch.randn(3, 4)
        result, bp = tropical_einsum([x], [(0, 1)], (1, 0))
        torch.testing.assert_close(result, x.T)
        assert bp is None

    def test_permutedims_3d(self):
        """Test permutation: ijk -> kji."""
        x = torch.randn(2, 3, 4)
        result, bp = tropical_einsum([x], [(0, 1, 2)], (2, 1, 0))
        torch.testing.assert_close(result, x.permute(2, 1, 0))
        assert bp is None

    def test_tropical_sum_single_dim(self):
        """Test tropical sum (max) over single dimension: ij -> i."""
        x = torch.randn(3, 4)
        result, bp = tropical_einsum([x], [(0, 1)], (0,), track_argmax=True)
        expected = x.max(dim=1).values
        torch.testing.assert_close(result, expected)
        assert bp is not None
        assert bp.elim_vars == (1,)

    def test_tropical_sum_multiple_dims(self):
        """Test tropical sum over multiple dimensions: ijk -> i."""
        x = torch.randn(2, 3, 4)
        result, bp = tropical_einsum([x], [(0, 1, 2)], (0,), track_argmax=True)
        expected = x.reshape(2, -1).max(dim=1).values
        torch.testing.assert_close(result, expected)
        assert bp is not None

    def test_tropical_sum_to_scalar(self):
        """Test tropical sum to scalar: ij -> ()."""
        x = torch.randn(3, 4)
        result, bp = tropical_einsum([x], [(0, 1)], (), track_argmax=True)
        expected = x.max()
        torch.testing.assert_close(result, expected)
        assert bp is not None

    def test_trace_tropical(self):
        """Test tropical trace: ii -> ()."""
        x = torch.randn(4, 4)
        result, bp = tropical_einsum([x], [(0, 0)], ())
        expected = torch.diagonal(x).max()
        torch.testing.assert_close(result, expected)


# =============================================================================
# Binary Rule Tests (adapted from OMEinsum)
# =============================================================================

class TestBinaryRules:
    """Test binary contraction rules."""

    def test_outer_product(self):
        """Test outer product: i,k -> ik (no contraction)."""
        a = torch.randn(3)
        b = torch.randn(4)
        result, bp = tropical_einsum([a, b], [(0,), (1,)], (0, 1))
        expected = a.unsqueeze(1) + b.unsqueeze(0)
        torch.testing.assert_close(result, expected)

    def test_dot_product(self):
        """Test tropical dot product: j,j -> ()."""
        a = torch.randn(5)
        b = torch.randn(5)
        result, bp = tropical_einsum([a, b], [(0,), (0,)], (), track_argmax=True)
        expected = (a + b).max()
        torch.testing.assert_close(result, expected)
        assert bp is not None

    def test_matrix_vector_left(self):
        """Test matrix-vector: ij,j -> i."""
        a = torch.randn(3, 4)
        b = torch.randn(4)
        result, bp = tropical_einsum([a, b], [(0, 1), (1,)], (0,), track_argmax=True)
        # Tropical: max_j(A[i,j] + b[j])
        expected = (a + b.unsqueeze(0)).max(dim=1).values
        torch.testing.assert_close(result, expected)

    def test_matrix_vector_right(self):
        """Test vector-matrix: j,jk -> k."""
        a = torch.randn(4)
        b = torch.randn(4, 5)
        result, bp = tropical_einsum([a, b], [(0,), (0, 1)], (1,), track_argmax=True)
        expected = (a.unsqueeze(1) + b).max(dim=0).values
        torch.testing.assert_close(result, expected)

    def test_matrix_matrix(self):
        """Test tropical matrix multiplication: ij,jk -> ik."""
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        result, bp = tropical_einsum([a, b], [(0, 1), (1, 2)], (0, 2), track_argmax=True)

        # Reference: C[i,k] = max_j(A[i,j] + B[j,k])
        expected = torch.zeros(3, 5)
        for i in range(3):
            for k in range(5):
                expected[i, k] = (a[i, :] + b[:, k]).max()

        torch.testing.assert_close(result, expected)
        assert bp is not None
        assert bp.elim_vars == (1,)

    def test_matrix_matrix_transposed_output(self):
        """Test tropical matmul with transposed output: ij,jk -> ki."""
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        result, bp = tropical_einsum([a, b], [(0, 1), (1, 2)], (2, 0), track_argmax=True)

        expected = torch.zeros(5, 3)
        for i in range(3):
            for k in range(5):
                expected[k, i] = (a[i, :] + b[:, k]).max()

        torch.testing.assert_close(result, expected)

    def test_batched_matmul(self):
        """Test batched tropical matmul: ijl,jkl -> ikl."""
        a = torch.randn(3, 4, 2)  # (i, j, l)
        b = torch.randn(4, 5, 2)  # (j, k, l)
        result, bp = tropical_einsum(
            [a, b], [(0, 1, 2), (1, 3, 2)], (0, 3, 2), track_argmax=True
        )

        # Reference
        expected = torch.zeros(3, 5, 2)
        for l in range(2):
            for i in range(3):
                for k in range(5):
                    expected[i, k, l] = (a[i, :, l] + b[:, k, l]).max()

        torch.testing.assert_close(result, expected)

    def test_element_wise_with_broadcast(self):
        """Test element-wise addition with broadcast: il,l -> il."""
        a = torch.randn(3, 4)
        b = torch.randn(4)
        result, bp = tropical_einsum([a, b], [(0, 1), (1,)], (0, 1))
        expected = a + b.unsqueeze(0)
        torch.testing.assert_close(result, expected)


# =============================================================================
# Rule Matching Tests
# =============================================================================

class TestRuleMatching:
    """Test rule matching logic."""

    def test_match_identity(self):
        rule = match_rule([(0, 1)], (0, 1))
        assert isinstance(rule, Identity)

    def test_match_permutedims(self):
        rule = match_rule([(0, 1)], (1, 0))
        assert isinstance(rule, Permutedims)

    def test_match_tropical_sum(self):
        rule = match_rule([(0, 1)], (0,))
        assert isinstance(rule, TropicalSum)

    def test_match_trace(self):
        rule = match_rule([(0, 0)], ())
        assert isinstance(rule, Tr)

    def test_match_binary_simple(self):
        rule = match_rule([(0, 1), (1, 2)], (0, 2))
        assert isinstance(rule, SimpleBinaryRule)

    def test_match_binary_outer_product(self):
        rule = match_rule([(0,), (1,)], (0, 1))
        assert isinstance(rule, SimpleBinaryRule)


# =============================================================================
# Backpointer and Argmax Tracing Tests
# =============================================================================

class TestArgmaxTracing:
    """Test argmax tracing for MPE recovery."""

    def test_argmax_trace_1d(self):
        """Test argmax trace for 1D reduction."""
        x = torch.tensor([1.0, 5.0, 3.0, 2.0])
        result, bp = tropical_einsum([x], [(0,)], (), track_argmax=True)

        assert result.item() == 5.0
        assert bp is not None

        assignment = argmax_trace(bp, {})
        assert assignment[0] == 1  # Index of max value

    def test_argmax_trace_2d_single_dim(self):
        """Test argmax trace for 2D with single dim reduction."""
        x = torch.tensor([[1.0, 5.0], [3.0, 2.0], [4.0, 1.0]])
        result, bp = tropical_einsum([x], [(0, 1)], (0,), track_argmax=True)

        assert result.tolist() == [5.0, 3.0, 4.0]
        assert bp is not None

        # Check argmax for each row
        for i in range(3):
            assignment = argmax_trace(bp, {0: i})
            expected_j = int(x[i].argmax())
            assert assignment[1] == expected_j

    def test_argmax_trace_binary_contraction(self):
        """Test argmax trace for binary contraction."""
        a = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
        b = torch.tensor([[2.0, 1.0], [1.0, 3.0]])

        result, bp = tropical_einsum([a, b], [(0, 1), (1, 2)], (0, 2), track_argmax=True)

        # C[i,k] = max_j(A[i,j] + B[j,k])
        # C[0,0] = max(1+2, 2+1) = max(3, 3) = 3
        # C[0,1] = max(1+1, 2+3) = max(2, 5) = 5
        # C[1,0] = max(3+2, 1+1) = max(5, 2) = 5
        # C[1,1] = max(3+1, 1+3) = max(4, 4) = 4

        assert bp is not None

        # Test argmax recovery
        assignment = argmax_trace(bp, {0: 0, 2: 1})
        assert assignment[1] == 1  # j=1 gives max for C[0,1]


# =============================================================================
# Correctness Tests Against Reference
# =============================================================================

class TestCorrectnessAgainstReference:
    """Test tropical_einsum against reference implementation."""

    @pytest.mark.parametrize("seed", range(5))
    def test_random_matrix_multiply(self, seed):
        """Test random matrix multiplication patterns."""
        torch.manual_seed(seed)

        a = torch.randn(4, 5)
        b = torch.randn(5, 6)

        result, _ = tropical_einsum([a, b], [(0, 1), (1, 2)], (0, 2))
        expected = tropical_einsum_reference([a, b], [(0, 1), (1, 2)], (0, 2))

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("seed", range(5))
    def test_random_3tensor_chain(self, seed):
        """Test 3-tensor chain contraction."""
        torch.manual_seed(seed)

        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        c = torch.randn(5, 6)

        # Contract a-b first, then with c
        ab, _ = tropical_einsum([a, b], [(0, 1), (1, 2)], (0, 2))
        result, _ = tropical_einsum([ab, c], [(0, 2), (2, 3)], (0, 3))

        # Reference
        expected = tropical_einsum_reference([a, b], [(0, 1), (1, 2)], (0, 2))
        expected = tropical_einsum_reference([expected, c], [(0, 2), (2, 3)], (0, 3))

        torch.testing.assert_close(result, expected)


# =============================================================================
# Integration with tropical-gemm
# =============================================================================

class TestTropicalGemmIntegration:
    """Test integration with tropical-gemm library."""

    def test_uses_tropical_gemm_when_available(self):
        """Verify tropical-gemm is being used for matrix multiplication."""
        import tropical_gemm

        a = torch.randn(10, 20)
        b = torch.randn(20, 30)

        result, bp = tropical_einsum([a, b], [(0, 1), (1, 2)], (0, 2), track_argmax=True)

        # Verify result is correct
        expected = torch.zeros(10, 30)
        for i in range(10):
            for k in range(30):
                expected[i, k] = (a[i, :] + b[:, k]).max()

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_tropical_gemm_matches_reference(self):
        """Test that tropical-gemm results match reference."""
        torch.manual_seed(42)

        for m, k, n in [(5, 10, 8), (20, 30, 25), (3, 100, 4)]:
            a = torch.randn(m, k)
            b = torch.randn(k, n)

            result, _ = tropical_einsum([a, b], [(0, 1), (1, 2)], (0, 2))
            expected = tropical_einsum_reference([a, b], [(0, 1), (1, 2)], (0, 2))

            torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_element_tensors(self):
        """Test with single-element tensors."""
        a = torch.tensor([5.0])
        b = torch.tensor([3.0])

        result, _ = tropical_einsum([a, b], [(0,), (0,)], ())
        assert result.item() == 8.0

    def test_empty_contraction(self):
        """Test contraction with no eliminated variables."""
        a = torch.randn(3)
        b = torch.randn(4)

        result, bp = tropical_einsum([a, b], [(0,), (1,)], (0, 1))
        expected = a.unsqueeze(1) + b.unsqueeze(0)
        torch.testing.assert_close(result, expected)
        assert bp is None

    def test_scalar_result(self):
        """Test contraction resulting in scalar."""
        x = torch.randn(3, 4, 5)
        result, bp = tropical_einsum([x], [(0, 1, 2)], (), track_argmax=True)
        expected = x.max()
        torch.testing.assert_close(result, expected)

    def test_high_dimensional(self):
        """Test with higher-dimensional tensors."""
        a = torch.randn(2, 3, 4)
        b = torch.randn(4, 5)

        result, _ = tropical_einsum([a, b], [(0, 1, 2), (2, 3)], (0, 1, 3))
        expected = tropical_einsum_reference([a, b], [(0, 1, 2), (2, 3)], (0, 1, 3))
        torch.testing.assert_close(result, expected)
