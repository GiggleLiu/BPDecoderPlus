"""Test MPE implementation against standard benchmarks.

The Asia network (also known as the "lung cancer" network) is a classic
Bayesian network benchmark created by Lauritzen & Spiegelhalter (1988).

Variables:
0: Asia (visit to Asia)
1: Smoking
2: Tuberculosis
3: Lung Cancer
4: Tuberculosis or Cancer
5: Positive X-ray
6: Dyspnea
7: Bronchitis

The MPE should find the most likely configuration given the network structure.
"""

import itertools
import os
import pytest
import torch

from tropical_in_new.src.mpe import mpe_tropical
from tropical_in_new.src.utils import read_model_file


BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "benchmarks")


def brute_force_mpe(model):
    """Compute MPE by exhaustive enumeration."""
    best_score = float("-inf")
    best_assignment = None

    for combo in itertools.product(*(range(c) for c in model.cards)):
        score = 0.0
        for factor in model.factors:
            idx = tuple(combo[v - 1] for v in factor.vars)
            val = factor.values[idx].item()
            if val <= 0:
                score = float("-inf")
                break
            score += torch.log(torch.tensor(val)).item()

        if score > best_score:
            best_score = score
            best_assignment = {i + 1: combo[i] for i in range(len(model.cards))}

    return best_assignment, best_score


class TestAsiaBenchmark:
    """Test MPE on the Asia (lung cancer) network benchmark."""

    @pytest.fixture
    def asia_model(self):
        """Load the Asia benchmark model."""
        uai_path = os.path.join(BENCHMARK_DIR, "asia.uai")
        return read_model_file(uai_path, factor_eltype=torch.float64)

    def test_asia_mpe_matches_brute_force(self, asia_model):
        """Test that tropical MPE matches brute-force enumeration."""
        # Compute MPE using our implementation
        assignment, score, info = mpe_tropical(asia_model)

        # Compute MPE using brute force
        brute_assignment, brute_score = brute_force_mpe(asia_model)

        # Verify they match
        assert assignment == brute_assignment, (
            f"MPE assignment mismatch:\n"
            f"  tropical: {assignment}\n"
            f"  brute:    {brute_assignment}"
        )
        assert abs(score - brute_score) < 1e-6, (
            f"MPE score mismatch: tropical={score}, brute={brute_score}"
        )

    def test_asia_mpe_with_evidence(self, asia_model):
        """Test MPE with evidence on Asia network."""
        # Evidence: person visited Asia (var 1 = 0) and has dyspnea (var 7 = 0)
        evidence = {1: 0, 7: 0}

        assignment, score, _ = mpe_tropical(asia_model, evidence=evidence)

        # Verify evidence is respected
        assert assignment[1] == 0, "Evidence not respected for Asia variable"
        assert assignment[7] == 0, "Evidence not respected for Dyspnea variable"

        # Verify against brute force with evidence
        best_score = float("-inf")
        best_assignment = None
        for combo in itertools.product(*(range(c) for c in asia_model.cards)):
            # Skip if doesn't match evidence
            if combo[0] != 0 or combo[6] != 0:
                continue

            score_bf = 0.0
            for factor in asia_model.factors:
                idx = tuple(combo[v - 1] for v in factor.vars)
                val = factor.values[idx].item()
                if val <= 0:
                    score_bf = float("-inf")
                    break
                score_bf += torch.log(torch.tensor(val)).item()

            if score_bf > best_score:
                best_score = score_bf
                best_assignment = {i + 1: combo[i] for i in range(len(asia_model.cards))}

        assert assignment == best_assignment

    def test_asia_mpe_result_reasonable(self, asia_model):
        """Test that MPE result is reasonable for Asia network."""
        assignment, score, info = mpe_tropical(asia_model)

        # The most likely configuration should have:
        # - No visit to Asia (0.99 probability)
        # - Smoker status depends on prior (0.5)
        # - No tuberculosis (given no Asia visit, 0.99)
        # - No lung cancer (given smoking status)
        # - No bronchitis (given smoking status)

        # Basic sanity checks
        assert len(assignment) == 8, "Should have assignments for all 8 variables"
        assert all(1 <= k <= 8 for k in assignment.keys()), "Variables should be 1-indexed"
        assert all(v in [0, 1] for v in assignment.values()), "All variables are binary"

        # Score should be negative (log probability)
        assert score < 0, "Log probability should be negative"

        # The most likely state should have no Asia visit (high prior probability)
        assert assignment[1] == 1, "Most likely: no visit to Asia"


class TestMPECorrectness:
    """Additional MPE correctness tests."""

    def test_mpe_simple_chain(self):
        """Test MPE on a simple chain model."""
        # A -> B -> C with deterministic relationships
        uai_content = """MARKOV
3
2 2 2
2
2 0 1
2 1 2

4
1.0 0.0
0.0 1.0

4
1.0 0.0
0.0 1.0
"""
        from tropical_in_new.src.utils import read_model_from_string
        model = read_model_from_string(uai_content, factor_eltype=torch.float64)

        assignment, score, _ = mpe_tropical(model)

        # With identity factors, both 0-0-0 and 1-1-1 have the same probability
        # The MPE should be one of them
        assert (assignment == {1: 0, 2: 0, 3: 0} or
                assignment == {1: 1, 2: 1, 3: 1})

    def test_mpe_with_strong_evidence(self):
        """Test MPE where evidence strongly determines the solution."""
        uai_content = """MARKOV
2
2 2
1
2 0 1

4
0.9 0.1
0.1 0.9
"""
        from tropical_in_new.src.utils import read_model_from_string
        model = read_model_from_string(uai_content, factor_eltype=torch.float64)

        # Without evidence: both (0,0) and (1,1) have prob 0.9
        assignment1, _, _ = mpe_tropical(model)
        assert assignment1[1] == assignment1[2]  # Should match

        # With evidence on var 1 = 0
        assignment2, _, _ = mpe_tropical(model, evidence={1: 0})
        assert assignment2 == {1: 0, 2: 0}

        # With evidence on var 1 = 1
        assignment3, _, _ = mpe_tropical(model, evidence={1: 1})
        assert assignment3 == {1: 1, 2: 1}
