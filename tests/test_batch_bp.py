"""Tests for BatchBPDecoder."""

import numpy as np
import pytest
import stim
import torch

from bpdecoderplus.batch_bp import BatchBPDecoder
from bpdecoderplus.dem import extract_dem, build_parity_check_matrix
from bpdecoderplus.syndrome import sample_syndromes
from bpdecoderplus.circuit import generate_circuit


class TestBatchBPDecoderInit:
    """Test BatchBPDecoder initialization."""

    def test_basic_init(self):
        """Test decoder initializes with valid inputs."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        priors = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        assert decoder.num_checks == 2
        assert decoder.num_qubits == 3
        assert decoder.num_edges == 4  # 4 ones in H

    def test_edge_structure(self):
        """Test edge lists are built correctly."""
        H = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.float32)
        priors = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        assert decoder.num_edges == 4


class TestBatchBPDecoderDecode:
    """Test BatchBPDecoder decoding."""

    def test_zero_syndrome(self):
        """Test decoding zero syndrome returns low error probabilities."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        priors = np.array([0.01, 0.01, 0.01], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        syndromes = torch.zeros(1, 2)
        marginals = decoder.decode(syndromes, max_iter=20)

        assert marginals.shape == (1, 3)
        # With zero syndrome and low priors, marginals should be low
        assert (marginals < 0.5).all()

    def test_batch_decoding(self):
        """Test batch decoding processes multiple syndromes."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        priors = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        syndromes = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.float32)
        marginals = decoder.decode(syndromes, max_iter=20)

        assert marginals.shape == (3, 3)

    def test_min_sum_method(self):
        """Test min-sum decoding method."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        priors = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        syndromes = torch.tensor([[1, 0]], dtype=torch.float32)
        marginals = decoder.decode(syndromes, max_iter=20, method='min-sum')

        assert marginals.shape == (1, 3)
        assert (marginals >= 0).all() and (marginals <= 1).all()

    def test_sum_product_method(self):
        """Test sum-product decoding method."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        priors = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        syndromes = torch.tensor([[1, 0]], dtype=torch.float32)
        marginals = decoder.decode(syndromes, max_iter=20, method='sum-product')

        assert marginals.shape == (1, 3)
        assert (marginals >= 0).all() and (marginals <= 1).all()

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        priors = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        syndromes = torch.tensor([[1, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="Unknown method"):
            decoder.decode(syndromes, method='invalid')

    def test_damping_effect(self):
        """Test that damping produces valid marginals."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        priors = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        syndromes = torch.tensor([[1, 1]], dtype=torch.float32)

        m1 = decoder.decode(syndromes, max_iter=50, damping=0.0)
        m2 = decoder.decode(syndromes, max_iter=50, damping=0.5)

        # Both should produce valid probabilities
        assert (m1 >= 0).all() and (m1 <= 1).all()
        assert (m2 >= 0).all() and (m2 <= 1).all()

    def test_repetition_code(self):
        """Test BP on simple repetition code."""
        # [3,1,3] repetition code
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        priors = np.array([0.1, 0.01, 0.01], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        # Syndrome [1,0] consistent with error on qubit 0
        syndromes = torch.tensor([[1, 0]], dtype=torch.float32)
        marginals = decoder.decode(syndromes, max_iter=50)

        # Qubit 0 should have highest error probability
        assert marginals[0, 0] > marginals[0, 1]
        assert marginals[0, 0] > marginals[0, 2]


class TestBPExactMarginalsOnTree:
    """Sum-product BP gives exact posteriors on tree-structured factor graphs."""

    def test_chain_code_exact_posteriors(self):
        """On a chain (tree) code, sum-product should give exact posteriors.

        Code: x0-c0-x1-c1-x2 (chain graph, no loops)
        H = [[1,1,0],[0,1,1]]
        Prior: p0=0.3, p1=0.1, p2=0.05
        Syndrome: [1, 0]

        Exact posterior for x0 given s=[1,0]:
          P(x0=1|s) ∝ P(s|x0=1)*P(x0=1) = P(c0=1|x0=1)*P(c1=0|x0=1)*P(x0=1)
        We compute these exactly and compare with BP.
        """
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        p = np.array([0.3, 0.1, 0.05], dtype=np.float32)
        decoder = BatchBPDecoder(H, p, device='cpu')

        syndromes = torch.tensor([[1, 0]], dtype=torch.float32)
        marginals = decoder.decode(syndromes, max_iter=100, damping=0.0, method='sum-product')

        # Exact calculation by enumeration over all 2^3 = 8 codewords
        # P(e|s) ∝ P(s|e) * P(e), where P(s|e) = 1 if He=s mod 2, else 0
        s = np.array([1, 0])
        posteriors = np.zeros(3)
        total = 0.0
        for e0 in range(2):
            for e1 in range(2):
                for e2 in range(2):
                    e = np.array([e0, e1, e2])
                    if np.all((H.astype(int) @ e) % 2 == s):
                        prob = 1.0
                        for i in range(3):
                            prob *= p[i] if e[i] == 1 else (1 - p[i])
                        posteriors += prob * e
                        total += prob
        exact_marginals = posteriors / total

        # Sum-product on tree should match exact marginals within numerical tolerance
        np.testing.assert_allclose(
            marginals[0].numpy(), exact_marginals, atol=0.02,
            err_msg="Sum-product on tree code should approximate exact posteriors"
        )

    def test_longer_chain_exact(self):
        """Test exact marginals on a longer chain (5-bit repetition code)."""
        # H for [5,1,5] repetition code (tree-structured)
        H = np.array([
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
        ], dtype=np.float32)
        p = np.array([0.2, 0.05, 0.1, 0.05, 0.15], dtype=np.float32)
        decoder = BatchBPDecoder(H, p, device='cpu')

        # Syndrome consistent with error on qubit 0: s=[1,0,0,0]
        syndromes = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
        marginals = decoder.decode(syndromes, max_iter=100, damping=0.0, method='sum-product')

        # Exact enumeration
        s = np.array([1, 0, 0, 0])
        posteriors = np.zeros(5)
        total = 0.0
        for bits in range(32):
            e = np.array([(bits >> i) & 1 for i in range(5)])
            if np.all((H.astype(int) @ e) % 2 == s):
                prob = 1.0
                for i in range(5):
                    prob *= p[i] if e[i] == 1 else (1 - p[i])
                posteriors += prob * e
                total += prob
        exact_marginals = posteriors / total

        np.testing.assert_allclose(
            marginals[0].numpy(), exact_marginals, atol=0.02,
            err_msg="Sum-product on 5-bit chain should match exact posteriors"
        )


class TestBPSurfaceCode:
    """Test BP on actual surface code DEM data."""

    @pytest.fixture
    def surface_code_d3(self):
        """Generate d=3 surface code DEM and syndromes."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)
        H, priors, obs_flip = build_parity_check_matrix(dem)
        syndromes, observables = sample_syndromes(circuit, num_shots=200)
        return H, priors, obs_flip, syndromes, observables

    def test_marginals_in_valid_range(self, surface_code_d3):
        """All BP marginals must be in [0, 1]."""
        H, priors, obs_flip, syndromes, _ = surface_code_d3
        decoder = BatchBPDecoder(H, priors.astype(np.float32), device='cpu')

        batch = torch.from_numpy(syndromes[:50]).float()
        marginals = decoder.decode(batch, max_iter=30, damping=0.2)

        assert (marginals >= 0).all(), "Marginals must be non-negative"
        assert (marginals <= 1).all(), "Marginals must be at most 1"

    def test_syndrome_satisfaction_rate(self, surface_code_d3):
        """BP hard decisions should satisfy the syndrome for a meaningful fraction of samples.

        At p=0.01 (well below threshold), BP should converge for most syndromes.
        """
        H, priors, obs_flip, syndromes, _ = surface_code_d3
        decoder = BatchBPDecoder(H, priors.astype(np.float32), device='cpu')

        batch = torch.from_numpy(syndromes[:100]).float()
        marginals = decoder.decode(batch, max_iter=60, damping=0.2)

        # Check syndrome satisfaction: H @ e ≡ s (mod 2)
        hard_decisions = (marginals > 0.5).int().numpy()
        H_int = H.astype(int)
        satisfied = 0
        for i in range(len(batch)):
            computed_s = (H_int @ hard_decisions[i]) % 2
            if np.all(computed_s == syndromes[i]):
                satisfied += 1

        # At p=0.01, BP should satisfy syndrome for at least 50% of samples
        assert satisfied >= 50, (
            f"Only {satisfied}/100 syndromes satisfied. "
            "BP should converge for most samples at p=0.01"
        )

    def test_low_error_rate_posteriors_are_low(self, surface_code_d3):
        """At low physical error rate, most marginals should be well below 0.5."""
        H, priors, obs_flip, syndromes, _ = surface_code_d3
        decoder = BatchBPDecoder(H, priors.astype(np.float32), device='cpu')

        # Use zero syndromes (no errors detected)
        batch = torch.zeros(10, H.shape[0])
        marginals = decoder.decode(batch, max_iter=30, damping=0.2)

        # With no detected errors and p=0.01, average posterior should be very low
        avg_marginal = marginals.mean().item()
        assert avg_marginal < 0.1, (
            f"Average marginal {avg_marginal:.4f} too high for zero syndrome at p=0.01"
        )

    def test_known_single_error_detection(self, surface_code_d3):
        """Injecting a known single error: BP should rank it among top candidates.

        At low p, the posterior won't be high in absolute terms (prior dominates),
        but the error position should be ranked higher than average.
        """
        H, priors, obs_flip, syndromes, _ = surface_code_d3
        H_int = H.astype(int)

        # Find a column with weight >= 2 (most error mechanisms trigger >=2 detectors)
        col_weights = H_int.sum(axis=0)
        good_col = np.argmax(col_weights)

        # Compute syndrome for a single error at that position
        single_error = np.zeros(H.shape[1], dtype=int)
        single_error[good_col] = 1
        syndrome = (H_int @ single_error) % 2

        decoder = BatchBPDecoder(H, priors.astype(np.float32), device='cpu')
        batch = torch.from_numpy(syndrome[np.newaxis, :]).float()
        marginals = decoder.decode(batch, max_iter=60, damping=0.2)

        # The error position should rank in the top 10% of marginals
        m = marginals[0].numpy()
        rank = (m >= m[good_col]).sum()  # How many are >= this marginal
        percentile = rank / len(m)
        assert percentile <= 0.2, (
            f"Injected error at position {good_col} has rank {rank}/{len(m)} "
            f"(top {percentile*100:.0f}%). Expected top 20%."
        )
