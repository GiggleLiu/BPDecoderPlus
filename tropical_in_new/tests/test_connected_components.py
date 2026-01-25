"""Tests for connected components handling in contraction tree generation.

These tests verify that the tropical TN correctly handles factor graphs with
disconnected components, which was a bug fix for Issue #68.
"""

import pytest
import torch

from tropical_in_new.src.contraction import (
    _find_connected_components,
    get_omeco_tree,
)
from tropical_in_new.src.network import TensorNode


class TestFindConnectedComponents:
    """Tests for the _find_connected_components function."""

    def test_single_component_connected(self):
        """All factors share variables - single component."""
        ixs = [[1], [2], [1, 2]]
        components = _find_connected_components(ixs)
        assert len(components) == 1
        assert sorted(components[0]) == [0, 1, 2]

    def test_two_disconnected_components(self):
        """Two separate groups of factors."""
        ixs = [[1], [2], [1, 2], [3], [4], [3, 4]]
        components = _find_connected_components(ixs)
        assert len(components) == 2
        # Check that each component contains the right factors
        comp_sets = [set(c) for c in components]
        assert {0, 1, 2} in comp_sets
        assert {3, 4, 5} in comp_sets

    def test_all_independent_factors(self):
        """Each factor is its own component."""
        ixs = [[1], [2], [3], [4], [5]]
        components = _find_connected_components(ixs)
        assert len(components) == 5
        all_indices = set()
        for c in components:
            all_indices.update(c)
        assert all_indices == {0, 1, 2, 3, 4}

    def test_chain_connected(self):
        """Factors connected in a chain."""
        ixs = [[1, 2], [2, 3], [3, 4], [4, 5]]
        components = _find_connected_components(ixs)
        assert len(components) == 1
        assert sorted(components[0]) == [0, 1, 2, 3]

    def test_empty_factors(self):
        """Empty factor list."""
        components = _find_connected_components([])
        assert components == []

    def test_single_factor(self):
        """Single factor."""
        ixs = [[1, 2, 3]]
        components = _find_connected_components(ixs)
        assert len(components) == 1
        assert components[0] == [0]


class TestGetOmecoTreeConnectedComponents:
    """Tests for get_omeco_tree handling of disconnected components."""

    def _count_leaves(self, tree):
        """Count leaf nodes in a contraction tree."""
        if "tensor_index" in tree:
            return 1, [tree["tensor_index"]]
        args = tree.get("args", tree.get("children", []))
        total = 0
        indices = []
        for a in args:
            c, i = self._count_leaves(a)
            total += c
            indices.extend(i)
        return total, indices

    def test_connected_factors_all_included(self):
        """All factors in a connected graph should be in the tree."""
        nodes = [
            TensorNode(vars=(1,), values=torch.rand(2)),
            TensorNode(vars=(2,), values=torch.rand(2)),
            TensorNode(vars=(1, 2), values=torch.rand(2, 2)),
        ]
        tree = get_omeco_tree(nodes)
        num_leaves, indices = self._count_leaves(tree)
        assert num_leaves == 3
        assert sorted(indices) == [0, 1, 2]

    def test_disconnected_factors_all_included(self):
        """All factors in disconnected components should be in the tree."""
        nodes = [
            TensorNode(vars=(1,), values=torch.rand(2)),
            TensorNode(vars=(2,), values=torch.rand(2)),
            TensorNode(vars=(1, 2), values=torch.rand(2, 2)),
            TensorNode(vars=(3,), values=torch.rand(2)),
            TensorNode(vars=(4,), values=torch.rand(2)),
            TensorNode(vars=(3, 4), values=torch.rand(2, 2)),
        ]
        tree = get_omeco_tree(nodes)
        num_leaves, indices = self._count_leaves(tree)
        assert num_leaves == 6
        assert sorted(indices) == [0, 1, 2, 3, 4, 5]

    def test_independent_factors_all_included(self):
        """Independent single-variable factors should all be included."""
        nodes = [
            TensorNode(vars=(1,), values=torch.rand(2)),
            TensorNode(vars=(2,), values=torch.rand(2)),
            TensorNode(vars=(3,), values=torch.rand(2)),
            TensorNode(vars=(4,), values=torch.rand(2)),
            TensorNode(vars=(5,), values=torch.rand(2)),
        ]
        tree = get_omeco_tree(nodes)
        num_leaves, indices = self._count_leaves(tree)
        assert num_leaves == 5
        assert sorted(indices) == [0, 1, 2, 3, 4]

    def test_mixed_structure_all_included(self):
        """Mix of connected and disconnected factors."""
        # Simulates surface code structure: prior factors + constraint factors
        nodes = [
            # Prior factors (single variable each)
            TensorNode(vars=(1,), values=torch.rand(2)),
            TensorNode(vars=(2,), values=torch.rand(2)),
            TensorNode(vars=(3,), values=torch.rand(2)),
            TensorNode(vars=(4,), values=torch.rand(2)),
            # Constraint factors (multiple variables)
            TensorNode(vars=(1, 2), values=torch.rand(2, 2)),
            TensorNode(vars=(2, 3), values=torch.rand(2, 2)),
            TensorNode(vars=(3, 4), values=torch.rand(2, 2)),
        ]
        tree = get_omeco_tree(nodes)
        num_leaves, indices = self._count_leaves(tree)
        assert num_leaves == 7
        assert sorted(indices) == [0, 1, 2, 3, 4, 5, 6]

    def test_surface_code_like_structure(self):
        """Structure similar to surface code decoding problem."""
        # 10 variables, 10 prior factors + 5 constraint factors
        nodes = []
        # Prior factors
        for i in range(1, 11):
            nodes.append(TensorNode(vars=(i,), values=torch.rand(2)))
        # Constraint factors (each touching 2-3 variables)
        nodes.append(TensorNode(vars=(1, 2), values=torch.rand(2, 2)))
        nodes.append(TensorNode(vars=(2, 3), values=torch.rand(2, 2)))
        nodes.append(TensorNode(vars=(3, 4, 5), values=torch.rand(2, 2, 2)))
        nodes.append(TensorNode(vars=(5, 6, 7), values=torch.rand(2, 2, 2)))
        nodes.append(TensorNode(vars=(7, 8, 9, 10), values=torch.rand(2, 2, 2, 2)))

        tree = get_omeco_tree(nodes)
        num_leaves, indices = self._count_leaves(tree)
        assert num_leaves == 15
        assert sorted(indices) == list(range(15))
