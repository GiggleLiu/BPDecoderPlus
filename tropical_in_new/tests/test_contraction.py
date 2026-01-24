import pytest
import torch

from tropical_in_new.src.contraction import (
    _elim_order_from_tree_dict,
    _extract_leaf_index,
    _infer_var_sizes,
    build_contraction_tree,
    choose_order,
    contract_tree,
)
from tropical_in_new.src.network import TensorNode
from tropical_in_new.src.primitives import tropical_einsum


def test_choose_order_returns_all_vars():
    nodes = [
        TensorNode(vars=(1, 2), values=torch.zeros((2, 2))),
        TensorNode(vars=(2, 3), values=torch.zeros((2, 2))),
    ]
    order = choose_order(nodes, heuristic="omeco")
    assert set(order) == {1, 2, 3}


def test_choose_order_invalid_heuristic():
    nodes = [TensorNode(vars=(1,), values=torch.zeros((2,)))]
    with pytest.raises(ValueError, match="Only the 'omeco' heuristic"):
        choose_order(nodes, heuristic="invalid")


def test_contract_tree_reduces_to_scalar():
    nodes = [
        TensorNode(vars=(1,), values=torch.tensor([0.1, 0.0])),
        TensorNode(vars=(1, 2), values=torch.tensor([[0.2, 0.3], [0.4, 0.1]])),
    ]
    order = [1, 2]
    tree = build_contraction_tree(order, nodes)
    root = contract_tree(tree, tropical_einsum)
    assert root.values.numel() == 1


def test_contract_tree_three_nodes_shared_var():
    """Test contraction with 3 nodes sharing a variable (uses einsum with elimination)."""
    nodes = [
        TensorNode(vars=(1, 2), values=torch.tensor([[0.1, 0.2], [0.3, 0.4]])),
        TensorNode(vars=(2, 3), values=torch.tensor([[0.5, 0.6], [0.7, 0.8]])),
        TensorNode(vars=(2,), values=torch.tensor([0.1, 0.9])),
    ]
    order = [2, 1, 3]
    tree = build_contraction_tree(order, nodes)
    root = contract_tree(tree, tropical_einsum)
    assert root.values.numel() == 1


def test_contract_tree_partial_order():
    """Test contraction where order doesn't cover all vars (remaining merged)."""
    nodes = [
        TensorNode(vars=(1, 2), values=torch.tensor([[0.1, 0.2], [0.3, 0.4]])),
        TensorNode(vars=(3,), values=torch.tensor([0.5, 0.6])),
    ]
    order = [1]
    tree = build_contraction_tree(order, nodes)
    root = contract_tree(tree, tropical_einsum)
    # var 2 and 3 remain
    assert 2 in root.vars or 3 in root.vars


def test_infer_var_sizes_inconsistent():
    nodes = [
        TensorNode(vars=(1,), values=torch.zeros((2,))),
        TensorNode(vars=(1,), values=torch.zeros((3,))),
    ]
    with pytest.raises(ValueError, match="inconsistent sizes"):
        _infer_var_sizes(nodes)


def test_extract_leaf_index():
    assert _extract_leaf_index({"leaf": 0}) == 0
    assert _extract_leaf_index({"leaf_index": 2}) == 2
    assert _extract_leaf_index({"index": 1}) == 1
    assert _extract_leaf_index({"tensor": 3}) == 3
    assert _extract_leaf_index({"other": "abc"}) is None
    assert _extract_leaf_index({"leaf": "not_int"}) is None


def test_elim_order_from_tree_dict():
    tree_dict = {
        "children": [
            {"leaf": 0},
            {"leaf": 1},
        ]
    }
    ixs = [[1, 2], [2, 3]]
    order = _elim_order_from_tree_dict(tree_dict, ixs)
    assert set(order) == {1, 2, 3}


def test_elim_order_from_tree_dict_no_children():
    tree_dict = {"other_key": "value"}
    ixs = [[1, 2]]
    order = _elim_order_from_tree_dict(tree_dict, ixs)
    # No children → remaining vars appended
    assert set(order) == {1, 2}
