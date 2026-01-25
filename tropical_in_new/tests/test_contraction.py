import pytest
import torch

from tropical_in_new.src.contraction import (
    _elim_order_from_tree_dict,
    _infer_var_sizes,
    build_contraction_tree,
    choose_order,
    contract_tree,
    contract_omeco_tree,
    get_omeco_tree,
)
from tropical_in_new.src.network import TensorNode


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
    root = contract_tree(tree)
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
    root = contract_tree(tree)
    assert root.values.numel() == 1


def test_contract_tree_partial_order():
    """Test contraction where order doesn't cover all vars (remaining merged)."""
    nodes = [
        TensorNode(vars=(1, 2), values=torch.tensor([[0.1, 0.2], [0.3, 0.4]])),
        TensorNode(vars=(3,), values=torch.tensor([0.5, 0.6])),
    ]
    order = [1]
    tree = build_contraction_tree(order, nodes)
    root = contract_tree(tree)
    # var 2 and 3 remain
    assert 2 in root.vars or 3 in root.vars


def test_infer_var_sizes_inconsistent():
    nodes = [
        TensorNode(vars=(1,), values=torch.zeros((2,))),
        TensorNode(vars=(1,), values=torch.zeros((3,))),
    ]
    with pytest.raises(ValueError, match="inconsistent sizes"):
        _infer_var_sizes(nodes)


def test_elim_order_from_tree_dict():
    tree_dict = {
        "children": [
            {"tensor_index": 0},
            {"tensor_index": 1},
        ]
    }
    ixs = [[1, 2], [2, 3]]
    order = _elim_order_from_tree_dict(tree_dict, ixs)
    assert set(order) == {1, 2, 3}


def test_elim_order_from_tree_dict_omeco_format():
    """Test elimination order extraction with omeco format (args instead of children)."""
    tree_dict = {
        "args": [
            {"tensor_index": 0},
            {"tensor_index": 1},
        ],
        "eins": {"ixs": [[1, 2], [2, 3]], "iy": [1, 3]}
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


def test_get_omeco_tree():
    """Test getting optimized tree from omeco."""
    nodes = [
        TensorNode(vars=(1, 2), values=torch.zeros((2, 2))),
        TensorNode(vars=(2, 3), values=torch.zeros((2, 2))),
    ]
    tree_dict = get_omeco_tree(nodes)
    assert isinstance(tree_dict, dict)
    # Should have either tensor_index (leaf) or args (internal node)
    assert "tensor_index" in tree_dict or "args" in tree_dict


def test_contract_omeco_tree_basic():
    """Test contract_omeco_tree basic functionality."""
    nodes = [
        TensorNode(vars=(1, 2), values=torch.tensor([[0.1, 0.2], [0.3, 0.4]])),
        TensorNode(vars=(2, 3), values=torch.tensor([[0.5, 0.6], [0.7, 0.8]])),
    ]
    tree_dict = get_omeco_tree(nodes)
    root = contract_omeco_tree(tree_dict, nodes)
    # Should contract to a 2x2 tensor (vars 1 and 3)
    assert root.values.shape == (2, 2) or root.values.numel() == 4


def test_contract_omeco_tree_matches_legacy():
    """Test that contract_omeco_tree produces correct results."""
    torch.manual_seed(42)
    nodes = [
        TensorNode(vars=(1, 2), values=torch.randn(3, 4)),
        TensorNode(vars=(2, 3), values=torch.randn(4, 5)),
        TensorNode(vars=(3, 4), values=torch.randn(5, 6)),
    ]

    # New approach using omeco tree
    tree_dict = get_omeco_tree(nodes)
    new_root = contract_omeco_tree(tree_dict, nodes, track_argmax=False)

    # Verify result shape and existence
    assert new_root.values is not None
    assert new_root.values.numel() > 0

    # Verify correctness: compute reference result manually
    # Chain contraction: (1,2) x (2,3) x (3,4) -> (1,4) after eliminating 2,3
    a, b, c = [n.values for n in nodes]
    # Tropical matmul: C[i,k] = max_j(A[i,j] + B[j,k])
    ab = torch.zeros(3, 5)
    for i in range(3):
        for k in range(5):
            ab[i, k] = (a[i, :] + b[:, k]).max()
    expected = torch.zeros(3, 6)
    for i in range(3):
        for k in range(6):
            expected[i, k] = (ab[i, :] + c[:, k]).max()

    # The final result should match the expected tropical chain contraction
    torch.testing.assert_close(
        new_root.values.reshape(expected.shape), expected, atol=1e-5, rtol=1e-5
    )
