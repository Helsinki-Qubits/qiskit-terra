# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Permrowcol-algorithm functionality implementation"""

import numpy as np
import retworkx as rx

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import LinearFunction
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.synthesis.graph_utils import (
    postorder_traversal,
    preorder_traversal,
    pydigraph_to_pygraph,
    noncutting_vertices,
)
from qiskit.circuit.library.generalized_gates.permutation import Permutation
from qiskit.circuit.exceptions import CircuitError
from qiskit.synthesis.linear.linear_circuits_utils import calc_inverse_matrix
from qiskit.providers.fake_provider import FakeManilaV2


class PermRowCol:
    """Permrowcol algorithm"""

    def __init__(self, coupling_map: CouplingMap):
        self._coupling_map = coupling_map
        self._graph = pydigraph_to_pygraph(self._coupling_map.graph)
        self.add_hadamarts = False

    def perm_row_col(self, parity_mat: np.ndarray) -> QuantumCircuit:

        """Run permrowcol algorithm on the given parity matrix

        Args:
            parity_mat (np.ndarray): parity matrix representing a circuit

        Returns:
            QuantumCircuit: synthesized circuit
        """
        num_qubits = len(self._graph.node_indexes())
        qubit_alloc = [-1] * num_qubits

        circuit = QuantumCircuit(num_qubits)

        while len(self._graph.node_indexes()) > 1:
            n_vertices = noncutting_vertices(self._graph)
            row = self.choose_row(n_vertices, parity_mat)

            cols = self._return_columns(qubit_alloc)
            column = self.choose_column(parity_mat, cols, row)
            nodes = self._get_nodes(parity_mat, column)

            self._eliminate_column(circuit, parity_mat, row, column, nodes)

            # for edge in self._eliminate_column(parity_mat, row, column, nodes):
            #     circuit.cx(edge[0], edge[1])
            if sum(parity_mat[row]) > 1:
                nodes = self._get_nodes_for_eliminate_row(parity_mat, column, row)
                self._eliminate_row(circuit, parity_mat, row, nodes)


            # if sum(parity_mat[row]) > 1:
            #     nodes = self._get_nodes_for_eliminate_row(parity_mat, column, row)

            #     for edge in self._eliminate_row(parity_mat, row, nodes):
            #         circuit.cx(edge[0], edge[1])  # Adds a CNOT to the circuit

            qubit_alloc[column] = row

            self._reduce_graph(row)

        if len(qubit_alloc) != 0:
            qubit_alloc[qubit_alloc.index(-1)] = self._graph.node_indexes()[0]

        try:
            perm = Permutation(num_qubits, qubit_alloc)
        except CircuitError:
            raise RuntimeError(
                f"Formed qubit allocation vector is not a valid permutation pattern: {qubit_alloc}"
            )
        return circuit.inverse(), perm
        # return circuit, perm

    def _reduce_graph(self, node: int):
        """Removes a node from pydigraph

        Args:
            node (int): index of node to remove
        """
        self._graph.remove_node(node)

    def _get_nodes(self, parity_mat: np.ndarray, column: int) -> list:
        """Returns a list of nodes that have 1s in the chosen column in the parity matrix

        Args:
            parity_mat (np.ndarray): parity matrix
            column (int): column index

        Returns:
            list: list of nodes

        """
        return [node for node in self._graph.node_indexes() if parity_mat[node, column] == 1]

    def _return_columns(self, qubit_alloc: list) -> list:
        """Returns list of indices of not yet processed columns in parity matrix

        Args:
            qubit_alloc (list): qubit allocation list

        Returns:
            list: list of indices of yet to be processed columns in the parity matrix

        """
        return [i for i in range(len(qubit_alloc)) if qubit_alloc[i] == -1]

    def choose_row(self, vertices: np.ndarray, parity_mat: np.ndarray) -> np.int64:
        """Choose row to eliminate and return the index.

        Args:
            vertices (np.ndarray): vertices (corresponding to rows) to choose from
            parity_mat (np.ndarray): parity matrix

        Returns:
            int: vertex/row index
        """
        return vertices[np.argmin([sum(parity_mat[i]) for i in vertices])]

    def choose_column(self, parity_mat: np.ndarray, cols: np.ndarray, chosen_row: int) -> np.int64:
        """Choose column to eliminate and return the index.

        Args:
            parity_mat (np.ndarray): parity matrix
            cols (np.ndarray): column indices to choose from
            chosen_row (int): row index that has been eliminated

        Returns:
            int: column index
        """
        col_sum = [
            sum(parity_mat[:, i]) if parity_mat[chosen_row][i] == 1 else len(parity_mat) + 1
            for i in cols
        ]
        return cols[np.argmin(col_sum)]

    def _eliminate_column(
        self,
        circuit: QuantumCircuit,
        parity_mat: np.ndarray,
        root: int,
        col: int,
        terminals: np.ndarray
    ):
        """Eliminates the selected column from the parity matrix and returns the operations.

        Args:
            circuit (QuantumCircuit): the circuit we are synthesizing
            parity_mat (np.ndarray): parity matrix
            coupling (CouplingMap): topology
            root (int): root of the steiner tree
            terminals (np.ndarray): terminals of the steiner tree

        """
        tree = rx.steiner_tree(self._graph, terminals, weight_fn=lambda x: 1)
        post_edges = postorder_traversal(tree, root)

        for edge in post_edges:
            if parity_mat[edge[0], col] == 0:
                self._add_cnot(circuit, parity_mat, edge[0], edge[1])

        for edge in post_edges:
            self._add_cnot(circuit, parity_mat, edge[1], edge[0])


    # def _eliminate_column(
    #     self,
    #     parity_mat: np.ndarray,
    #     root: int,
    #     col: int,
    #     terminals: np.ndarray,
    # ) -> list:
    #     """Eliminates the selected column from the parity matrix and returns the operations.

    #     Args:
    #         parity_mat (np.ndarray): parity matrix
    #         coupling (CouplingMap): topology
    #         root (int): root of the steiner tree
    #         terminals (np.ndarray): terminals of the steiner tree

    #     Returns:
    #         list: list of tuples represents control and target qubits with a cnot gate between them.
    #     """
    #     C = []
    #     tree = rx.steiner_tree(self._graph, terminals, weight_fn=lambda x: 1)
    #     post_edges = postorder_traversal(tree, root)

    #     for edge in post_edges:
    #         if parity_mat[edge[0], col] == 0:
    #             C.append((edge[0], edge[1]))
    #             parity_mat[edge[0], :] = (parity_mat[edge[0], :] + parity_mat[edge[1], :]) % 2

    #     for edge in post_edges:
    #         C.append((edge[1], edge[0]))
    #         parity_mat[edge[1], :] = (parity_mat[edge[0], :] + parity_mat[edge[1], :]) % 2

    #     return C

    def _eliminate_row(self, circuit: QuantumCircuit, parity_mat: np.ndarray, root: int, terminals: np.ndarray):
        """Eliminates the selected row from the parity matrix and returns the operations as a list of tuples.

        Args:
            circuit (QuantumCircuit): the circuit we are synthesizing
            parity_mat (np.ndarray): parity matrix
            coupling (CouplingMap): topology
            root (int): root of the steiner tree
            terminals (np.ndarray): terminals of the steiner tree
        """
        tree = rx.steiner_tree(self._graph, terminals, weight_fn=lambda x: 1)

        pre_edges = preorder_traversal(tree, root)
        post_edges = postorder_traversal(tree, root)

        for edge in pre_edges:

            if edge[1] not in terminals:
                self._add_cnot(circuit, parity_mat, edge[0], edge[1])

        for edge in post_edges:
            self._add_cnot(circuit, parity_mat, edge[0], edge[1])

    # def _eliminate_row(self, parity_mat: np.ndarray, root: int, terminals: np.ndarray) -> list:
    #     """Eliminates the selected row from the parity matrix and returns the operations as a list of tuples.

    #     Args:
    #         parity_mat (np.ndarray): parity matrix
    #         coupling (CouplingMap): topology
    #         root (int): root of the steiner tree
    #         terminals (np.ndarray): terminals of the steiner tree

    #     Returns:
    #         list of tuples represents control and target qubits with a cnot gate between them.
    #     """
    #     C = []
    #     tree = rx.steiner_tree(self._graph, terminals, weight_fn=lambda x: 1)

    #     pre_edges = preorder_traversal(tree, root)
    #     post_edges = postorder_traversal(tree, root)

    #     for edge in pre_edges:

    #         if edge[1] not in terminals:
    #             C.append((edge[0], edge[1]))
    #             parity_mat[edge[0], :] = (parity_mat[edge[0], :] + parity_mat[edge[1], :]) % 2

    #     for edge in post_edges:
    #         C.append((edge[0], edge[1]))
    #         parity_mat[edge[0], :] = (parity_mat[edge[0], :] + parity_mat[edge[1], :]) % 2

    #     return C

    def _add_cnot(self, circuit: QuantumCircuit, parity_mat: np.ndarray, control: int, target: int):
        """" Adds a CX between `control` and `target` qubits to the given QuantumCircuit `circuit` and updates the matrix `parity_mat` accordingly.
        If the CX direction is not allowed by the CouplingMap, it will be conjugated with Hadamards and reversed.

        Args:
            circuit (QuantumCircuit): the circuit being synthesized
            parity_mat (np.ndarray): the parity matrix
            control (int): the control qubit for the CNOT
            target (int) : the target qubit for the CNOT

        """
        if self.add_hadamarts==True:
            if (control, target) not in self._coupling_map:
                circuit.h(control)
                circuit.h(target)
                circuit.cx(target, control)
                circuit.h(control)
                circuit.h(target)
            else:
                circuit.cx(control, target)
        else:
                circuit.cx(control, target)
        parity_mat[control, :] = (parity_mat[control, :] + parity_mat[target, :]) % 2
        #Bellow is Ariannes code but the upper line gives right results
        #parity_mat[target, :] = (parity_mat[control, :] + parity_mat[target, :]) % 2


    def _get_nodes_for_eliminate_row(
        self, parity_mat: np.ndarray, chosen_column: int, chosen_row: int
    ) -> list:

        """Find terminals for steiner_tree in eliminate_row method as a list of rows making linear combination of row chosen_row

        Args:
            parity_mat (np.ndarray): parity matrix representing a circuit
            chosen_column (int): index of the column to be eliminated
            chosen_row (int): index of the row to be eliminated


        Returns:
            List: list of terminals for steiner_tree in eliminate_row method.

        """

        A = np.delete(np.delete(parity_mat.copy(), chosen_row, 0), chosen_column, 1).astype(
            int
        )  # Parity_mat without chosen_column and chosen_row
        B = np.delete(parity_mat[chosen_row], chosen_column)  # Chosen_row without chosen_column

        inv_A = calc_inverse_matrix(A)
        # inv_A = LinearFunction(
        #     LinearFunction(A).synthesize().reverse_ops()
        # ).linear  # Creates inverse of parity_mat

        X = np.insert((np.matmul(B, inv_A) % 2), chosen_row, 1)  # Calculates B*inv_A

        nodes = [
            i for i in self._graph.node_indices() if i == chosen_row or X[i] == 1
        ]  # Finds indexes of rows that are added to chosen_row

        return nodes

    def test_perm_row_col_returns_valid_output_with_a_common_case(self):
        """Test the output of perm_row_col for correctness"""
        backend = FakeManilaV2()
        coupling_map = backend.coupling_map
        coupling = CouplingMap(coupling_map)
        permrowcol = PermRowCol(coupling)
        parity_mat = build_random_parity_matrix(42, 5, 60)
        original_parity_map = parity_mat.copy()
        circuit, perm = permrowcol.perm_row_col(parity_mat)
        circuit_matrix = LinearFunction(circuit).linear.astype(int)
        t_circuit_matrix = np.transpose(circuit_matrix)
        instance = np.matmul(circuit_matrix, parity_mat)
        self.assertEqual(np.array_equal(instance, original_parity_map), True)
