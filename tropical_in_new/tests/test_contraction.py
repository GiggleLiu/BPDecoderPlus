import torch

from src.contraction import build_contraction_tree, choose_order, contract_tree
from src.network import TensorNode
from src.primitives import tropical_einsum


def test_choose_order_returns_all_vars():
    nodes = [
        TensorNode(vars=(1, 2), values=torch.zeros((2, 2))),
        TensorNode(vars=(2, 3), values=torch.zeros((2, 2))),
    ]
    order = choose_order(nodes, heuristic="min_fill")
    assert set(order) == {1, 2, 3}


def test_contract_tree_reduces_to_scalar():
    nodes = [
        TensorNode(vars=(1,), values=torch.tensor([0.1, 0.0])),
        TensorNode(vars=(1, 2), values=torch.tensor([[0.2, 0.3], [0.4, 0.1]])),
    ]
    order = [1, 2]
    tree = build_contraction_tree(order, nodes)
    root = contract_tree(tree, tropical_einsum)
    assert root.values.numel() == 1
