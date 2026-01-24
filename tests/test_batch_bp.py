"""Tests for BatchBPDecoder."""

import numpy as np
import pytest
import torch

from bpdecoderplus.batch_bp import BatchBPDecoder


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
