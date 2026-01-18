import unittest

from pytorch_bp import (
    read_model_file,
    read_evidence_file,
    BeliefPropagation,
    belief_propagate,
    compute_marginals,
    apply_evidence,
)


class TestIntegration(unittest.TestCase):
    def test_example_file_runs(self):
        model = read_model_file("examples/simple_model.uai")
        bp = BeliefPropagation(model)
        state, info = belief_propagate(bp, max_iter=30, tol=1e-8)
        self.assertTrue(info.iterations > 0)
        marginals = compute_marginals(state, bp)
        self.assertEqual(set(marginals.keys()), {1, 2})

    def test_example_with_evidence(self):
        model = read_model_file("examples/simple_model.uai")
        evidence = read_evidence_file("examples/simple_model.evid")
        bp = apply_evidence(BeliefPropagation(model), evidence)
        state, info = belief_propagate(bp, max_iter=30, tol=1e-8)
        self.assertTrue(info.iterations > 0)
        marginals = compute_marginals(state, bp)
        self.assertEqual(set(marginals.keys()), {1, 2})


if __name__ == "__main__":
    unittest.main()
