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

"""Helper functions for hadling graphs"""

import numpy as np
import retworkx as rx

from qiskit.transpiler import CouplingMap


def pydigraph_to_pygraph(pydigraph: rx.PyDiGraph) -> rx.PyGraph:
    """Changes directed Pydigraph into an undirected Pygraph"""
    return pydigraph.to_undirected()


def noncutting_vertices(pygraph: rx.PyGraph) -> np.ndarray:
    """Extracts noncutting vertices from a given coupling map graph. Direction is not taken into account.

    Args:
        pygraph (rx.PyGraph): undirected graph of topology

    Returns:
        np.ndarray: array of non-cutting node indices
    """
    cutting_vertices = rx.articulation_points(pygraph)
    vertices = set(pygraph.node_indices())
    noncutting = np.array(list(vertices - cutting_vertices))

    return noncutting


def postorder_traversal(tree: rx.PyGraph, root: int = None, visited: list = None) -> list:
    """Traverse the given tree in postorder. Traversed edges are saved as tuples.
    The first element is the parent and second the child.
    Children are visited in increasing order.

    Args:
        tree (rx.PyGraph): tree to traverse
        root (int): root node
        visited (list): list of visited nodes

    Returns:
        edges (list): edge list
    """
    edges = []
    if visited == None:
        visited = []
    if root != None:
        visited.append(root)
        for neighbor in sorted(tree.neighbors(root)):
            if neighbor not in visited:
                # Travels to the next node before adding the edge.
                edges.extend(postorder_traversal(tree, neighbor, visited))
                edges.append((root, neighbor))
    return edges


def preorder_traversal(tree: rx.PyGraph, root: int = None, visited: list = None) -> list:
    """Preorder traversal of the edges of the given tree. Traversed edges are saved as tuples,
    where the first element is the parent and second the child. Children are visited in
    increasing order.

    Args:
        tree (rx.PyGraph): tree to traverse
        root (int): root node
        visited (list): list of visited nodes

    Returns:
        edges (list): edge list
    """
    edges = []
    if visited == None:
        visited = []
    if root != None:
        visited.append(root)
        for neighbor in sorted(tree.neighbors(root)):
            if neighbor not in visited:
                # Adds edge before traveling to the next node.
                edges.append((root, neighbor))
                edges.extend(postorder_traversal(tree, neighbor, visited))
    return edges
