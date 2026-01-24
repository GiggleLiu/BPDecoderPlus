import itertools

import pytest
import torch

from tropical_in_new.src.mpe import mpe_tropical, recover_mpe_assignment
from tropical_in_new.src.network import TensorNode
from tropical_in_new.src.utils import read_model_from_string


def _brute_force_mpe(cards, factors):
    """General brute-force MPE over any number of variables."""
    best_score = float("-inf")
    best_assignment = None
    for combo in itertools.product(*(range(c) for c in cards)):
        score = 0.0
        for vars, values in factors:
            idx = tuple(combo[v - 1] for v in vars)
            score += torch.log(values[idx]).item()
        if score > best_score:
            best_score = score
            best_assignment = {i + 1: combo[i] for i in range(len(cards))}
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


def test_mpe_three_variables():
    """Test MPE on a 3-variable model to exercise general brute-force."""
    uai = "\n".join(
        [
            "MARKOV",
            "3",
            "2 2 2",
            "3",
            "2 0 1",
            "2 1 2",
            "1 2",
            "4",
            "1.0 0.2 0.3 0.9",
            "4",
            "0.8 0.1 0.2 0.7",
            "2",
            "0.4 0.6",
        ]
    )
    model = read_model_from_string(uai, factor_eltype=torch.float64)
    assignment, score, _ = mpe_tropical(model)
    brute_assignment, brute_score = _brute_force_mpe(
        model.cards, [(f.vars, f.values) for f in model.factors]
    )
    assert assignment == brute_assignment
    assert abs(score - brute_score) < 1e-8


def test_mpe_partial_order():
    """Test MPE with a partial elimination order (remaining vars reduced at end)."""
    uai = "\n".join(
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
            "1.2 0.2 0.2 1.2",
        ]
    )
    model = read_model_from_string(uai, factor_eltype=torch.float64)
    # Only eliminate var 1, leaving var 2 for final reduce
    assignment, score, _ = mpe_tropical(model, order=[1])
    brute_assignment, brute_score = _brute_force_mpe(
        model.cards, [(f.vars, f.values) for f in model.factors]
    )
    assert assignment == brute_assignment
    assert abs(score - brute_score) < 1e-8


def test_recover_mpe_assignment_tensor_node():
    """Test recover_mpe_assignment directly on a TensorNode with vars."""
    node = TensorNode(
        vars=(1, 2),
        values=torch.tensor([[0.1, 0.9], [0.3, 0.2]]),
    )
    assignment = recover_mpe_assignment(node)
    # The max is at (0, 1) → var 1=0, var 2=1
    assert assignment == {1: 0, 2: 1}


def test_recover_mpe_assignment_bad_node():
    """Test recover_mpe_assignment raises on missing variables."""
    from tropical_in_new.src.contraction import ReduceNode
    from tropical_in_new.src.primitives import Backpointer

    child = TensorNode(vars=(1, 2), values=torch.tensor([[0.1, 0.9], [0.3, 0.2]]))
    bp = Backpointer(
        elim_vars=(2,), elim_shape=(2,), out_vars=(99,),  # bad out_vars
        argmax_flat=torch.tensor([1, 0])
    )
    root = ReduceNode(vars=(), values=torch.tensor(0.9), child=child, elim_vars=(2,), backpointer=bp)
    with pytest.raises(KeyError, match="Missing assignment"):
        recover_mpe_assignment(root)


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
