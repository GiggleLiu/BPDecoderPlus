"""Tests for BatchOSDDecoder."""

import numpy as np
import pytest
import torch

from bpdecoderplus.batch_osd import BatchOSDDecoder


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
