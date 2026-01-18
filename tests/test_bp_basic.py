import unittest
import itertools
import torch

from pytorch_bp import (
    read_model_from_string,
    BeliefPropagation,
    belief_propagate,
    compute_marginals,
    apply_evidence,
)


def exact_marginals(model, evidence=None):
    evidence = evidence or {}
    assignments = list(itertools.product(*[range(c) for c in model.cards]))
    weights = []
    for assignment in assignments:
        if any(assignment[var_idx - 1] != val for var_idx, val in evidence.items()):
            weights.append(0.0)
            continue
        weight = 1.0
        for factor in model.factors:
            idx = tuple(assignment[v - 1] for v in factor.vars)
            weight *= float(factor.values[idx])
        weights.append(weight)
    total = sum(weights)
    marginals = {}
    for var_idx, card in enumerate(model.cards):
        values = []
        for value in range(card):
            mass = 0.0
            for assignment, weight in zip(assignments, weights):
                if assignment[var_idx] == value:
                    mass += weight
            values.append(mass / total if total > 0 else 0.0)
        marginals[var_idx + 1] = torch.tensor(values, dtype=torch.float64)
    return marginals


class TestBeliefPropagationBasic(unittest.TestCase):
    def setUp(self):
        self.content = "\n".join(
            [
                "MARKOV",
                "2",
                "2 2",
                "2",
                "1 0",
                "2 0 1",
                "2",
                "0.6 0.4",
                "4",
                "0.9 0.1 0.2 0.8",
            ]
        )

    def test_bp_matches_exact_tree(self):
        model = read_model_from_string(self.content)
        bp = BeliefPropagation(model)
        state, info = belief_propagate(bp, max_iter=50, tol=1e-10, damping=0.0)
        self.assertTrue(info.converged)
        marginals = compute_marginals(state, bp)
        exact = exact_marginals(model)

        for var_idx in marginals:
            self.assertTrue(torch.allclose(marginals[var_idx], exact[var_idx], atol=1e-6))

    def test_apply_evidence(self):
        model = read_model_from_string(self.content)
        evidence = {1: 1}
        bp = apply_evidence(BeliefPropagation(model), evidence)
        state, info = belief_propagate(bp, max_iter=50, tol=1e-10, damping=0.0)
        self.assertTrue(info.converged)
        marginals = compute_marginals(state, bp)

        exact = exact_marginals(read_model_from_string(self.content), evidence=evidence)
        self.assertTrue(
            torch.allclose(
                marginals[1], torch.tensor([0.0, 1.0], dtype=torch.float64)
            )
        )
        self.assertAlmostEqual(float(marginals[2].sum()), 1.0, places=6)
        self.assertTrue(torch.allclose(marginals[2], exact[2], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
