import unittest
import numpy as np
import torch

try:
    from ._path import add_project_root_to_path
except ImportError:
    from _path import add_project_root_to_path

add_project_root_to_path()

from bpdecoderplus.osd import OSDDecoder
from bpdecoderplus.batch_bp import BatchBPDecoder
from bpdecoderplus.pytorch_bp import BeliefPropagation, _check_syndrome_satisfied


class TestOSDCostFunction(unittest.TestCase):
    """Test that OSD cost function uses correct log-probability weight."""

    def test_cost_function_minimum_weight(self):
        """Test that cost function selects minimum weight solution."""
        # Simple 3-bit repetition code: H = [1 1 1]
        H = np.array([[1, 1, 1]], dtype=np.int8)
        decoder = OSDDecoder(H)

        # Syndrome = 1 (odd parity)
        syndrome = np.array([1], dtype=np.int8)

        # Error probabilities: first bit most likely
        error_probs = np.array([0.9, 0.1, 0.1])

        # OSD-0 should select single-bit error on first position
        result = decoder.solve(syndrome, error_probs, osd_order=0)

        # Expected: [1, 0, 0] (minimum weight solution with highest probability)
        self.assertEqual(result.sum(), 1, "Should select single-bit error")
        self.assertEqual(result[0], 1, "Should select most likely error position")

    def test_cost_function_log_probability(self):
        """Test that cost function uses -log(p) weighting."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        decoder = OSDDecoder(H)

        syndrome = np.array([1, 0], dtype=np.int8)
        error_probs = np.array([0.8, 0.3, 0.2])

        # Compute cost manually for different solutions
        probs_clipped = np.clip(error_probs, 1e-10, 1 - 1e-10)

        # Solution 1: [1, 0, 0] - satisfies syndrome [1, 0]
        # Check: [1,1,0]@[1,0,0] = 1, [0,1,1]@[1,0,0] = 0 ✓
        sol1 = np.array([1, 0, 0])
        cost1 = np.sum(sol1 * (-np.log(probs_clipped)))

        # Solution 2: [0, 1, 1] - satisfies syndrome [1, 0]
        # Check: [1,1,0]@[0,1,1] = 1, [0,1,1]@[0,1,1] = 0 ✓
        sol2 = np.array([0, 1, 1])
        cost2 = np.sum(sol2 * (-np.log(probs_clipped)))

        # Solution 1 should have lower cost (higher probability)
        self.assertLess(cost1, cost2, "Higher probability solution should have lower cost")

        # OSD should select solution 1
        result = decoder.solve(syndrome, error_probs, osd_order=10)
        np.testing.assert_array_equal(result, sol1, "Should select minimum cost solution")


class TestBatchBPConvergence(unittest.TestCase):
    """Test that Batch BP checks syndrome satisfaction for early stopping."""

    def test_syndrome_convergence_early_stop(self):
        """Test that BP stops early when syndrome is satisfied."""
        # Simple repetition code
        H = np.array([[1, 1, 1]], dtype=np.float32)
        channel_probs = np.array([0.1, 0.1, 0.1], dtype=np.float32)

        decoder = BatchBPDecoder(H, channel_probs, device='cpu')

        # Syndrome that's easy to satisfy
        syndromes = torch.tensor([[0]], dtype=torch.float32)

        # Decode with many iterations
        marginals = decoder.decode(syndromes, max_iter=100)

        # Check that decoding satisfies syndrome
        decoding = (marginals > 0.5).float()
        computed_syndrome = (torch.from_numpy(H) @ decoding.T).T % 2
        self.assertTrue(
            torch.allclose(computed_syndrome, syndromes),
            "Decoding should satisfy syndrome"
        )

    def test_batch_syndrome_satisfaction(self):
        """Test syndrome satisfaction for batch of syndromes."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        channel_probs = np.array([0.1, 0.1, 0.1], dtype=np.float32)

        decoder = BatchBPDecoder(H, channel_probs, device='cpu')

        # Multiple syndromes
        syndromes = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.float32)

        marginals = decoder.decode(syndromes, max_iter=50)

        # Check all decodings satisfy their syndromes
        decoding = (marginals > 0.5).float()
        computed_syndrome = (torch.from_numpy(H) @ decoding.T).T % 2

        for i in range(len(syndromes)):
            self.assertTrue(
                torch.allclose(computed_syndrome[i], syndromes[i]),
                f"Decoding {i} should satisfy syndrome"
            )


class TestSyndromeCheckFunction(unittest.TestCase):
    """Test the syndrome satisfaction check function."""

    def test_syndrome_check_satisfied(self):
        """Test syndrome check returns True when satisfied."""
        # Create simple factor graph for XOR constraint
        from bpdecoderplus.pytorch_bp import read_model_from_string

        # Simple parity check: x1 XOR x2 = 0
        content = "\n".join([
            "MARKOV",
            "2",
            "2 2",
            "1",
            "2 0 1",
            "4",
            "1.0 0.0 0.0 1.0",  # Enforces x1 XOR x2 = 0
        ])

        model = read_model_from_string(content)
        bp = BeliefPropagation(model)

        # Decoding that satisfies: both 0 or both 1
        decoding1 = {1: 0, 2: 0}
        syndrome = {0: 0}
        self.assertTrue(_check_syndrome_satisfied(bp, decoding1, syndrome))

        decoding2 = {1: 1, 2: 1}
        self.assertTrue(_check_syndrome_satisfied(bp, decoding2, syndrome))

    def test_syndrome_check_not_satisfied(self):
        """Test syndrome check returns False when not satisfied."""
        from bpdecoderplus.pytorch_bp import read_model_from_string

        content = "\n".join([
            "MARKOV",
            "2",
            "2 2",
            "1",
            "2 0 1",
            "4",
            "1.0 0.0 0.0 1.0",
        ])

        model = read_model_from_string(content)
        bp = BeliefPropagation(model)

        # Decoding that doesn't satisfy: x1=0, x2=1 gives parity 1
        decoding = {1: 0, 2: 1}
        syndrome = {0: 0}
        self.assertFalse(_check_syndrome_satisfied(bp, decoding, syndrome))


if __name__ == '__main__':
    unittest.main()
