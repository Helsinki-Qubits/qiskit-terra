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
from qiskit.transpiler.synthesis.graph_utils import (postorder_traversal,
                                                     preorder_traversal,
                                                     pydigraph_to_pygraph)


class PermRowCol:
    """Permrowcol algorithm"""

    def __init__(self, coupling_map: CouplingMap):
        self._coupling_map = coupling_map
        self._graph = coupling_map.graph

    def perm_row_col(self, parity_mat: np.ndarray, coupling_map: CouplingMap) -> QuantumCircuit:
        """Run permrowcol algorithm on the given parity matrix

        Args:
            parity_mat (np.ndarray): parity matrix representing a circuit
            coupling_map (CouplingMap): topology constraint

        Returns:
            QuantumCircuit: synthesized circuit
        """
        # TODO
        circuit = QuantumCircuit(QuantumRegister(0))
        return circuit

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

    def eliminate_column(
        self,
        parity_mat: np.ndarray,
        root: int,
        col: int,
        terminals: np.ndarray,
    ) -> list:
        """Eliminates the selected column from the parity matrix and returns the operations.

        Args:
            parity_mat (np.ndarray): parity matrix
            coupling (CouplingMap): topology
            root (int): root of the steiner tree
            terminals (np.ndarray): terminals of the steiner tree

        Returns:
            list: list of tuples represents control and target qubits with a cnot gate between them.
        """
        C = []
        tree = rx.steiner_tree(pydigraph_to_pygraph(self._graph), terminals, weight_fn=lambda x: 1)
        post_edges = []
        postorder_traversal(tree, root, post_edges)

        for edge in post_edges:
            if parity_mat[edge[0], col] == 0:
                C.append((edge[0], edge[1]))
                parity_mat[edge[0], :] = (parity_mat[edge[0], :] + parity_mat[edge[1], :]) % 2

        for edge in post_edges:
            C.append((edge[1], edge[0]))
            parity_mat[edge[1], :] = (parity_mat[edge[0], :] + parity_mat[edge[1], :]) % 2

        return C

    def eliminate_row(self, parity_mat: np.ndarray, root: int, terminals: np.ndarray) -> list:
        """Eliminates the selected row from the parity matrix and returns the operations as a list of tuples.

        Args:
            parity_mat (np.ndarray): parity matrix
            coupling (CouplingMap): topology
            root (int): root of the steiner tree
            terminals (np.ndarray): terminals of the steiner tree

        Returns:
            list of tuples represents control and target qubits with a cnot gate between them.
        """
        C = []
        tree = rx.steiner_tree(pydigraph_to_pygraph(self._graph), terminals, weight_fn=lambda x: 1)

        pre_edges = []
        preorder_traversal(tree, root, pre_edges)
        post_edges = []
        postorder_traversal(tree, root, post_edges)

        for edge in pre_edges:

            if edge[1] not in terminals:
                C.append((edge[0], edge[1]))
                parity_mat[edge[0], :] = (parity_mat[edge[0], :] + parity_mat[edge[1], :]) % 2

        for edge in post_edges:
            C.append((edge[0], edge[1]))
            parity_mat[edge[0], :] = (parity_mat[edge[0], :] + parity_mat[edge[1], :]) % 2

        return C

    def matrix_edit(self, parity_mat: np.ndarray, vertices: np.ndarray, chosen_column: int, chosen_row: int, circuit: QuantumCircuit) -> QuantumCircuit:
        """ Checks if the sum of the chosen row is bigger than one. If the sum is bigger than one, the function performs row elimination.
        
        Args:
            parity_mat: np.ndarray
            vertices (np.ndarray): vertices (corresponding to rows) to choose from
            chosen_column: int
            chosen_row: int
            circuit: QuantumCircuit

        
        Returns:
            quantumCircuit eliminated row.
        
        """

        row = self.choose_row(self, vertices, parity_mat)
        column = self.choose_column(self, parity_mat, chosen_column, row)

        

        if sum(parity_mat[row]) > 1:

            C = self.parity_mat
            C_copy = []

            A_first_step = (
                # Creates a new matrix without the chosen row
                C_copy.append(C[rows nuber][halutun vaihdettavan tunnut, esim. jos halutaan ensimmäinen niin luku on 0 jne.] = Nollarivi)
                )
            
            A = (
                # Creates a new matrix without the chosen row and column
                C_copy.append(C[][] = Nollakolumni (nollattu arvi kohdasta x))
                )

            B = (
                # M[r] (chosen row) without chosen column

                )

            inv_A = (
                # Creates inverse of the matrix
                LinearFunction(LinearFunction(C).synthesize().reverse_ops()).linear
                )

            X = (
                # Creates A^-1B
                np.matmul(inv_A, X)
                )

            return C(pitää palauttaa circuit)


        elif sum(parity_mat[r_materials]) <= 1:
            pass

        return True

        for i in range(len(self.parity_mat)):
            C_copy = []
            for j in range(hauttu rivi):
                x = 0
                C_copy.append(x)
            C_copy.append(C_copy)
