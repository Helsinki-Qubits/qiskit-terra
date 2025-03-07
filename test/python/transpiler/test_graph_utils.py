"""Test graph utils"""

import unittest
import numpy as np
import rustworkx as rx

from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.graph_utils import (
    noncutting_vertices,
    pydigraph_to_pygraph,
    postorder_traversal,
    preorder_traversal,
)
from qiskit.transpiler import CouplingMap


class TestGraphUtils(QiskitTestCase):
    """Test graph utils"""

    def test_noncutting_vertices_returns_np_ndarray(self):
        """Test the output type of noncutting_vertices"""
        graph = pydigraph_to_pygraph(CouplingMap().graph)

        instance = noncutting_vertices(graph)

        self.assertIsInstance(instance, np.ndarray)

    def test_noncutting_vertices_returns_an_ndarray_with_noncutting_vertices(self):
        """Test noncutting_vertices method for correctness"""
        coupling_list = [[0, 2], [1, 2], [2, 3], [2, 4], [3, 6], [4, 5], [4, 6]]
        graph = pydigraph_to_pygraph(CouplingMap(couplinglist=coupling_list).graph)

        instance = noncutting_vertices(graph)
        expected = np.array([0, 1, 3, 5, 6])

        self.assertCountEqual(instance, expected)

    def test_pydigraph_to_pygraph_returns_pygraph(self):
        """Test the output type of pydigraph_to_pygraph"""
        coupling = CouplingMap()

        instance = pydigraph_to_pygraph(coupling.graph)

        self.assertIsInstance(instance, rx.PyGraph)

    def test_postorder_traversal_returns_correct_edges(self):
        """Test that postorder_traversal returns correct edge list"""
        tree = rx.PyGraph()
        tree.add_nodes_from([0, 1, 2, 3, 4])
        tree.add_edges_from([(0, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1)])

        result = postorder_traversal(tree, 0)
        expected = [(1, 2), (1, 3), (1, 4), (0, 1)]

        self.assertEqual(result, expected)

    def test_postorder_traversal_returns_empty_list_if_root_node_not_in_tree(self):
        """Test that postorder_traversal returns empty edge list if the given
        node is not found in the tree"""
        tree = rx.PyGraph()
        tree.add_nodes_from([0, 1, 2, 3])
        tree.add_edges_from([(0, 1, 1), (1, 2, 1), (1, 3, 1)])

        result = postorder_traversal(tree, 5)

        self.assertEqual(result, [])

    def test_postorder_traversal_returns_empty_list_for_tree_with_no_edges(self):
        """Test that postorder_traversal returns empty edge list if the given
        tree doesn't have any edges"""
        tree = rx.PyGraph()
        tree.add_nodes_from([0])

        result = postorder_traversal(tree, 0)

        self.assertEqual(result, [])

    def test_postorder_traversal_returns_correct_edges_if_visited_list_is_not_empty(self):
        """Test that postorder_traversal returns correct edge list when some nodes have already been visited"""
        tree = rx.PyGraph()
        tree.add_nodes_from([0, 1, 2, 3, 4, 5, 6])
        tree.add_edges_from(
            [(0, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1), (2, 5, 1), (4, 6, 1), (2, 3, 1)]
        )
        visited = [0, 1]

        result = postorder_traversal(tree, 2, visited)
        expected = [(2, 3), (2, 5)]

        self.assertEqual(result, expected)

    def test_postorder_traversal_returns_an_empty_list_if_root_is_empty(self):
        """"""
        tree = rx.PyGraph()

        result = postorder_traversal(tree)
        expected = []

        self.assertEqual(result, expected)

    def test_preorder_traversal_returns_correct_edges(self):
        """Test that preorder_traversal returns correct edge list"""
        tree = rx.PyGraph()
        tree.add_nodes_from([0, 1, 2, 3, 4])
        tree.add_edges_from([(0, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1)])
        expected = [(0, 1), (1, 2), (1, 3), (1, 4)]

        result = preorder_traversal(tree, 0)

        self.assertEqual(result, expected)

    def test_preorder_traversal_returns_empty_list_if_root_node_not_in_tree(self):
        """Test that preorder_traversal returns empty edge list if the given
        node is not found in the tree"""
        tree = rx.PyGraph()
        tree.add_nodes_from([0, 1, 2, 3])
        tree.add_edges_from([(0, 1, 1), (1, 2, 1), (1, 3, 1)])

        result = preorder_traversal(tree, 5)

        self.assertEqual(result, [])

    def test_preorder_traversal_returns_empty_list_for_tree_with_no_edges(self):
        """Test that preorder_traversal returns empty edge list if the given
        tree doesn't have any edges"""
        tree = rx.PyGraph()
        tree.add_nodes_from([0])

        result = preorder_traversal(tree, 0)

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
