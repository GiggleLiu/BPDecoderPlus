import torch

from src.mpe import mpe_tropical
from src.utils import read_model_from_string


def _brute_force_mpe(cards, factors):
    best_score = float("-inf")
    best_assignment = None
    for x0 in range(cards[0]):
        for x1 in range(cards[1]):
            score = 0.0
            for vars, values in factors:
                idx = []
                for v in vars:
                    idx.append(x0 if v == 1 else x1)
                score += torch.log(values[tuple(idx)]).item()
            if score > best_score:
                best_score = score
                best_assignment = {1: x0, 2: x1}
    return best_assignment, best_score


def test_mpe_matches_bruteforce():
    uai = "\n".join(
        [
            "MARKOV",
            "2",
            "2 2",
            "3",
            "1 0",
            "1 1",
            "2 0 1",
            "2",
            "0.6 0.4",
            "2",
            "0.3 0.7",
            "4",
            "1.2 0.2 0.2 1.2",
        ]
    )
    model = read_model_from_string(uai, factor_eltype=torch.float64)
    assignment, score, _ = mpe_tropical(model)
    brute_assignment, brute_score = _brute_force_mpe(
        model.cards, [(f.vars, f.values) for f in model.factors]
    )
    assert assignment == brute_assignment
    assert abs(score - brute_score) < 1e-8


def test_mpe_with_evidence():
    uai = "\n".join(
        [
            "MARKOV",
            "2",
            "2 2",
            "2",
            "1 0",
            "2 0 1",
            "2",
            "0.8 0.2",
            "4",
            "1.0 0.1 0.1 1.0",
        ]
    )
    model = read_model_from_string(uai, factor_eltype=torch.float64)
    assignment, score, _ = mpe_tropical(model, evidence={1: 1})
    assert assignment[1] == 1
    assert isinstance(score, float)
