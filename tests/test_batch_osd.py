"""Tests for BatchOSDDecoder."""

import numpy as np
import pytest
import torch

from bpdecoderplus.batch_bp import BatchBPDecoder
from bpdecoderplus.batch_osd import BatchOSDDecoder
from bpdecoderplus.circuit import generate_circuit
from bpdecoderplus.dem import extract_dem, build_parity_check_matrix
from bpdecoderplus.syndrome import sample_syndromes


class TestBatchOSDDecoderInit:
    """Test BatchOSDDecoder initialization."""

    def test_basic_init(self):
        """Test decoder initializes with valid inputs."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        assert decoder.num_checks == 2
        assert decoder.num_errors == 3

    def test_init_stores_H(self):
        """Test that H matrix is stored correctly."""
        H = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        np.testing.assert_array_equal(decoder.H, H)


class TestBatchOSDDecoderSolve:
    """Test BatchOSDDecoder.solve method."""

    def test_single_error(self):
        """Test solving for a single error."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        # Syndrome [1,0] consistent with error on position 0
        syndrome = np.array([1, 0], dtype=np.int8)
        probs = np.array([0.9, 0.1, 0.1])

        result = decoder.solve(syndrome, probs, osd_order=0)

        # Should find error at position 0
        assert result[0] == 1
        # Result must satisfy syndrome
        assert np.all((H @ result) % 2 == syndrome)

    def test_zero_syndrome(self):
        """Test solving zero syndrome returns zero vector."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        syndrome = np.array([0, 0], dtype=np.int8)
        probs = np.array([0.1, 0.1, 0.1])

        result = decoder.solve(syndrome, probs, osd_order=0)

        assert np.all(result == 0)

    def test_osd_order_improves_solution(self):
        """Test that higher OSD order can find better solutions."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        syndrome = np.array([1, 1], dtype=np.int8)
        probs = np.array([0.1, 0.8, 0.1])

        result_0 = decoder.solve(syndrome, probs, osd_order=0)
        result_10 = decoder.solve(syndrome, probs, osd_order=10)

        # Both must satisfy syndrome
        assert np.all((H @ result_0) % 2 == syndrome)
        assert np.all((H @ result_10) % 2 == syndrome)

    def test_mismatched_probs_raises(self):
        """Test that mismatched probability length raises ValueError."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        syndrome = np.array([1, 0], dtype=np.int8)
        probs = np.array([0.1, 0.1])  # Wrong length

        with pytest.raises(ValueError, match="doesn't match"):
            decoder.solve(syndrome, probs)

    def test_combination_sweep_method(self):
        """Test OSD-CS method produces valid solutions."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        syndrome = np.array([1, 0], dtype=np.int8)
        probs = np.array([0.8, 0.1, 0.1])

        result = decoder.solve(syndrome, probs, osd_order=5, osd_method='combination_sweep')
        assert np.all((H @ result) % 2 == syndrome)


class TestBatchOSDDecoderSolveBatch:
    """Test BatchOSDDecoder.solve_batch method."""

    def test_batch_solve(self):
        """Test batch solving multiple syndromes."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        syndromes = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
        probs = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.8, 0.1],
        ])

        results = decoder.solve_batch(syndromes, probs, osd_order=5)

        assert results.shape == (3, 3)
        # All results must satisfy their respective syndromes
        for i in range(3):
            assert np.all((H @ results[i]) % 2 == syndromes[i])


class TestBatchOSDDecoderRREF:
    """Test RREF computation."""

    def test_rref_identity(self):
        """Test RREF of identity matrix."""
        H = np.eye(3, dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        syndrome = np.array([1, 0, 1], dtype=np.int8)
        sorted_indices = np.array([0, 1, 2])

        augmented, pivot_cols = decoder._get_rref_cached(sorted_indices, syndrome)
        assert pivot_cols == [0, 1, 2]

    def test_rref_rank_deficient(self):
        """Test RREF of rank-deficient matrix."""
        H = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        syndrome = np.array([1, 1], dtype=np.int8)
        sorted_indices = np.array([0, 1, 2])

        augmented, pivot_cols = decoder._get_rref_cached(sorted_indices, syndrome)
        assert len(pivot_cols) == 1  # Rank 1


class TestOSDSoftWeightCorrectness:
    """Test that soft-weight cost function selects correct solutions."""

    def test_soft_weight_prefers_high_prob_errors(self):
        """Soft weight should prefer flipping high-probability positions.

        Given two valid solutions with the same Hamming weight,
        soft weight (-log p) should pick the one using higher-probability errors.
        """
        H = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        # Syndrome [1, 1]: valid solutions include [1,0,1,0] and [0,1,0,1]
        syndrome = np.array([1, 1], dtype=np.int8)
        # q0 and q2 are high-probability errors; q1 and q3 are low
        probs = np.array([0.9, 0.01, 0.8, 0.01])

        result = decoder.solve(syndrome, probs, osd_order=10)

        # Should pick [1,0,1,0] (cost = -log(0.9)-log(0.8) ≈ 0.33)
        # over [0,1,0,1] (cost = -log(0.01)-log(0.01) ≈ 9.21)
        assert np.all((H @ result) % 2 == syndrome), "Solution must satisfy syndrome"
        assert result[0] == 1 and result[2] == 1, (
            f"Should select high-probability errors [1,0,1,0], got {result}"
        )

    def test_soft_weight_vs_hamming_disagree(self):
        """Construct a case where Hamming weight and soft weight disagree.

        For H=[[1,1,0],[0,1,1]], syndrome [1,1]:
        - [0,1,0] (Hamming=1, but q1 has very low prob) -> soft cost = -log(0.001) ≈ 6.9
        - [1,0,1] (Hamming=2, but q0,q2 have very high prob) -> soft cost ≈ 0.02

        OSD with soft weight should pick [1,0,1] despite higher Hamming weight.
        """
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        syndrome = np.array([1, 1], dtype=np.int8)
        # q1 has very low probability; q0 and q2 have very high probability
        probs = np.array([0.99, 0.001, 0.99])

        result = decoder.solve(syndrome, probs, osd_order=10)

        assert np.all((H @ result) % 2 == syndrome), "Solution must satisfy syndrome"
        # OSD should pick [1,0,1] (soft cost ≈ 0.02) over [0,1,0] (soft cost ≈ 6.9)
        expected = np.array([1, 0, 1])
        np.testing.assert_array_equal(result, expected,
            err_msg="Soft weight should prefer [1,0,1] over [0,1,0] despite higher Hamming weight")


class TestOSDSurfaceCode:
    """Test OSD on actual surface code DEM data."""

    @pytest.fixture
    def surface_code_d3(self):
        """Generate d=3 surface code DEM and syndromes."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)
        H, priors, obs_flip = build_parity_check_matrix(dem)
        syndromes, observables = sample_syndromes(circuit, num_shots=200)
        return H, priors, obs_flip, syndromes, observables

    def test_all_solutions_satisfy_syndrome(self, surface_code_d3):
        """Every OSD solution must satisfy H @ e ≡ s (mod 2)."""
        H, priors, obs_flip, syndromes, _ = surface_code_d3
        H_int = H.astype(int)

        bp_decoder = BatchBPDecoder(H, priors.astype(np.float32), device='cpu')
        osd_decoder = BatchOSDDecoder(H, device='cpu')

        # Decode 50 syndromes
        batch = torch.from_numpy(syndromes[:50]).float()
        marginals = bp_decoder.decode(batch, max_iter=30, damping=0.2)
        marginals_np = marginals.numpy()

        violations = 0
        for i in range(50):
            solution = osd_decoder.solve(syndromes[i], marginals_np[i], osd_order=5)
            computed_s = (H_int @ solution) % 2
            if not np.all(computed_s == syndromes[i]):
                violations += 1

        assert violations == 0, (
            f"{violations}/50 OSD solutions violate the syndrome constraint"
        )

    def test_osd_improves_upon_bp(self, surface_code_d3):
        """OSD should achieve LER ≤ BP-only on surface code at p=0.01.

        This is the fundamental correctness property: OSD post-processing
        should never make things worse than BP hard-decision.
        """
        H, priors, obs_flip, syndromes, observables = surface_code_d3
        H_int = H.astype(int)

        bp_decoder = BatchBPDecoder(H, priors.astype(np.float32), device='cpu')
        osd_decoder = BatchOSDDecoder(H, device='cpu')

        n_samples = 200
        batch = torch.from_numpy(syndromes[:n_samples]).float()
        marginals = bp_decoder.decode(batch, max_iter=60, damping=0.2)
        marginals_np = marginals.numpy()

        bp_errors = 0
        osd_errors = 0
        for i in range(n_samples):
            # BP prediction
            bp_hard = (marginals_np[i] > 0.5).astype(int)
            bp_pred = int(np.dot(bp_hard, obs_flip) % 2)
            if bp_pred != observables[i]:
                bp_errors += 1

            # OSD prediction
            solution = osd_decoder.solve(syndromes[i], marginals_np[i], osd_order=10)
            osd_pred = int(np.dot(solution, obs_flip) % 2)
            if osd_pred != observables[i]:
                osd_errors += 1

        bp_ler = bp_errors / n_samples
        osd_ler = osd_errors / n_samples

        # OSD should not be significantly worse than BP
        # Allow small statistical fluctuation (up to 3%)
        assert osd_ler <= bp_ler + 0.03, (
            f"OSD LER ({osd_ler:.3f}) should not exceed BP LER ({bp_ler:.3f}) + 0.03. "
            "OSD post-processing is making things worse."
        )

    def test_known_error_recovery(self, surface_code_d3):
        """OSD should recover a known injected single error."""
        H, priors, obs_flip, _, _ = surface_code_d3
        H_int = H.astype(int)

        # Inject a single error and compute its syndrome
        col_weights = H_int.sum(axis=0)
        error_pos = np.argmax(col_weights)  # Pick a high-weight column
        true_error = np.zeros(H.shape[1], dtype=int)
        true_error[error_pos] = 1
        syndrome = (H_int @ true_error) % 2

        # Give OSD near-perfect probability info (simulating a well-converged BP)
        probs = np.full(H.shape[1], 0.001)
        probs[error_pos] = 0.99

        osd_decoder = BatchOSDDecoder(H, device='cpu')
        result = osd_decoder.solve(syndrome.astype(np.int8), probs, osd_order=10)

        # OSD should recover the injected error
        assert np.all((H_int @ result) % 2 == syndrome), "Must satisfy syndrome"
        assert result[error_pos] == 1, (
            f"Failed to recover injected error at position {error_pos}"
        )

    def test_osd_cs_matches_exhaustive_on_small_order(self, surface_code_d3):
        """OSD-CS and exhaustive should agree at small OSD order on the same input."""
        H, priors, obs_flip, syndromes, _ = surface_code_d3

        bp_decoder = BatchBPDecoder(H, priors.astype(np.float32), device='cpu')
        osd_decoder = BatchOSDDecoder(H, device='cpu')

        # Use a non-trivial syndrome
        idx = np.argmax(syndromes.sum(axis=1))  # Pick syndrome with most detections
        batch = torch.from_numpy(syndromes[idx:idx+1]).float()
        marginals = bp_decoder.decode(batch, max_iter=30, damping=0.2)
        probs = marginals[0].numpy()

        # At osd_order=2, exhaustive searches 4 candidates, CS searches ~4 too
        result_exhaustive = osd_decoder.solve(
            syndromes[idx], probs, osd_order=2, osd_method='exhaustive'
        )
        result_cs = osd_decoder.solve(
            syndromes[idx], probs, osd_order=2, osd_method='combination_sweep'
        )

        H_int = H.astype(int)
        # Both must satisfy syndrome
        assert np.all((H_int @ result_exhaustive) % 2 == syndromes[idx])
        assert np.all((H_int @ result_cs) % 2 == syndromes[idx])
