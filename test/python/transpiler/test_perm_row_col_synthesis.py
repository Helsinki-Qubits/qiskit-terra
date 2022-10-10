"""Test PermRowColSynthesis"""

import unittest
import numpy as np
import retworkx as rx
from qiskit.test import QiskitTestCase
from qiskit.transpiler.passes.synthesis.perm_row_col_synthesis import PermRowColSynthesis
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap


class TestPermRowColSynthesis(QiskitTestCase):
    """Test PermRowColSynthesis"""

    def test_run_returns_a_dag(self):
        """Test the output type of run"""
        coupling = CouplingMap()
        circuit = QuantumCircuit()
        dag = circuit_to_dag(circuit)
        synthesis = PermRowColSynthesis(coupling)

        instance = synthesis.run(dag)

        self.assertIsInstance(instance, DAGCircuit)

    def test_perm_row_col_returns_circuit(self):
        """Test the output type of perm_row_col"""
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)
        parity_mat = np.identity(3)

        instance = synthesis.perm_row_col(parity_mat, coupling)

        self.assertIsInstance(instance, QuantumCircuit)

    def test_choose_row_returns_int(self):
        """Test the output type of choose_row"""
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)
        parity_mat = np.array([[1, 0], [0, 1]])
        vertices = np.array([0, 1])

        instance = synthesis.choose_row(vertices, parity_mat)

        self.assertIsInstance(instance, np.int64)

    def test_choose_row_returns_correct_index(self):
        """
        Test method to test the correctness of the choose_row method
        """
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)
        parity_mat = np.array(
            [
                [0, 1, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1, 0],
            ]
        )
        vertices = np.array([1, 3, 4, 5, 6, 8])

        index = synthesis.choose_row(vertices, parity_mat)

        self.assertEqual(index, 6)

    def test_choose_column_returns_int(self):
        """Test the output type of choose_column"""
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)
        parity_mat = np.array([[1, 0], [0, 1]])
        cols = np.array([0, 1])

        instance = synthesis.choose_column(parity_mat, cols, 0)

        self.assertIsInstance(instance, np.int64)

    def test_choose_column_returns_correct_index(self):
        """Test choose_colum method for correctness"""
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)

        parity_mat = np.array(
            [
                [0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1, 0],
            ]
        )
        vertices = np.array([1, 3, 4, 5, 6, 7])

        index = synthesis.choose_column(parity_mat, vertices, 4)

        self.assertEqual(index, 3)

    def test_choose_column_returns_correct_index_with_similar_col_sums(self):
        """Test choose_column method for correctness in case of col_sums having same integers"""
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)

        parity_mat = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])

        vertices = np.array([0, 1, 2])

        index = synthesis.choose_column(parity_mat, vertices, 2)

        self.assertEqual(index, 2)

    def test_eliminate_column_returns_int(self):
        """Test the output type of eliminate_column"""
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)
        parity_mat = np.ndarray(0)
        terminals = np.ndarray(0)

        instance = synthesis.eliminate_column(parity_mat, coupling, 0, terminals)

        self.assertIsInstance(instance, np.ndarray)

    def test_eliminate_row_returns_int(self):
        """Test the output type of eliminate_row"""
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)
        parity_mat = np.ndarray(0)
        terminals = np.ndarray(0)

        instance = synthesis.eliminate_row(parity_mat, coupling, 0, terminals)

        self.assertIsInstance(instance, np.ndarray)


    def test_noncutting_vertices_returns_np_ndarray(self):
        """Test the output type of _noncutting_vertices"""
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)

        instance = synthesis._noncutting_vertices(coupling)

        self.assertIsInstance(instance, np.ndarray)

    def test_noncutting_vertices_returns_an_ndarray_with_noncutting_vertices(self):
        """Test _noncutting_vertices method for correctness"""
        coupling_list = [[0, 2], [1, 2], [2, 3], [2, 4], [3, 6], [4, 5], [4, 6]]
        coupling = CouplingMap(couplinglist=coupling_list)
        synthesis = PermRowColSynthesis(coupling)

        instance = synthesis._noncutting_vertices(coupling)
        expected = np.array([0, 1, 3, 5, 6])

        self.assertCountEqual(instance, expected)

    def test_pydigraph_to_pygraph_returns_pygraph(self):
        """Test the output type of _pydigraph_to_pygraph"""
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)

        instance = synthesis._pydigraph_to_pygraph(coupling.graph)

        self.assertIsInstance(instance, rx.PyGraph)



if __name__ == "__main__":
    unittest.main()
