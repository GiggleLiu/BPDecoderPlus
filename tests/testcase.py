import unittest
import torch

try:
    from ._path import add_project_root_to_path
except ImportError:
    from _path import add_project_root_to_path

add_project_root_to_path()

from bpdecoderplus.pytorch_bp import (
    read_model_from_string,
    BeliefPropagation,
    belief_propagate,
    compute_marginals,
    initial_state,
    collect_message,
    process_message,
    apply_evidence,
)


from tests.test_utils import exact_marginals


class TestBPAdditionalCases(unittest.TestCase):
    def test_unary_factor_marginal(self):
        content = "\n".join(
            [
                "MARKOV",
                "1",
                "3",
                "1",
                "1 0",
                "3",
                "0.2 0.3 0.5",
            ]
        )
        model = read_model_from_string(content)
        bp = BeliefPropagation(model)
        state, info = belief_propagate(bp, max_iter=20, tol=1e-10, damping=0.0)
        self.assertTrue(info.converged)
        marginals = compute_marginals(state, bp)
        expected = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        self.assertTrue(torch.allclose(marginals[1], expected, atol=1e-6))

    def test_chain_three_vars_exact(self):
        content = "\n".join(
            [
                "MARKOV",
                "3",
                "2 2 2",
                "2",
                "2 0 1",
                "2 1 2",
                "4",
                "0.9 0.1 0.2 0.8",
                "4",
                "0.3 0.7 0.6 0.4",
            ]
        )
        model = read_model_from_string(content)
        bp = BeliefPropagation(model)
        state, info = belief_propagate(bp, max_iter=50, tol=1e-10, damping=0.0)
        self.assertTrue(info.converged)
        marginals = compute_marginals(state, bp)
        exact = exact_marginals(model)
        for var_idx in marginals:
            self.assertTrue(torch.allclose(marginals[var_idx], exact[var_idx], atol=1e-6))

    def test_message_normalization(self):
        content = "\n".join(
            [
                "MARKOV",
                "2",
                "2 2",
                "1",
                "2 0 1",
                "4",
                "0.9 0.1 0.2 0.8",
            ]
        )
        model = read_model_from_string(content)
        bp = BeliefPropagation(model)
        state = initial_state(bp)
        collect_message(bp, state, normalize=True)
        process_message(bp, state, normalize=True, damping=0.0)
        for var_msgs in state.message_in:
            for msg in var_msgs:
                self.assertAlmostEqual(float(msg.sum()), 1.0, places=6)
        for var_msgs in state.message_out:
            for msg in var_msgs:
                self.assertAlmostEqual(float(msg.sum()), 1.0, places=6)

    def test_zero_message_handling(self):
        content = "\n".join(
            [
                "MARKOV",
                "1",
                "2",
                "2",
                "1 0",
                "1 0",
                "2",
                "0.0 0.0",
                "2",
                "0.7 0.3",
            ]
        )
        model = read_model_from_string(content)
        bp = BeliefPropagation(model)
        state = initial_state(bp)
        collect_message(bp, state, normalize=True)
        process_message(bp, state, normalize=True, damping=0.0)
        self.assertAlmostEqual(float(state.message_in[0][0].sum()), 0.0, places=6)
        self.assertAlmostEqual(float(state.message_out[0][1].sum()), 0.0, places=6)

    def test_evidence_out_of_range_zeros_factor(self):
        content = "\n".join(
            [
                "MARKOV",
                "1",
                "2",
                "1",
                "1 0",
                "2",
                "0.4 0.6",
            ]
        )
        model = read_model_from_string(content)
        bp = apply_evidence(BeliefPropagation(model), {1: 5})
        self.assertAlmostEqual(float(bp.factors[0].values.sum()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
