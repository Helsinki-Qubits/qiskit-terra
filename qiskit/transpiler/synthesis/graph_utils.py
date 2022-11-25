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


# Susanna's code:

# def postorder_traversal(tree: rx.PyGraph, node: int, edges: list, parent: int = None):
#    """Traverse the given tree in postorder. Traversed edges are saved as tuples.
#    The first element is the parent and second the child.
#    Children are visited in increasing order.
#
#    Args:
#        tree (rx.PyGraph): tree to traverse
#        node (int): root node
#        edges (list): edge list
#        parent (int, optional): parent node. Defaults to None.
#    """
#    if node == None:
#        return
#    for n in sorted(tree.neighbors(node)):
#        if n == parent:
#            continue
#        postorder_traversal(tree, n, edges, node)
#    if parent != None:
#        edges.append((parent, node))

# Ariannes suggestion:


def postorder_traversal(tree: rx.PyGraph, root: int, visited: list = []) -> list:
    edges = []
    visited.append(
        root
    )  # Visited is only to avoid back links. You can also give a subgraph, but that was more complicated in pyGraph.
    for neighbor in sorted(tree.neighbors(root)):
        if neighbor not in visited:
            # Note that because the tree is asumed to be a tree, you don't need to give the other neighbors as visited.
            edges.extend(postorder_traversal(tree, neighbor, visited))
            edges.append((root, neighbor))
    return edges


def preorder_traversal(tree: rx.PyGraph, node: int, edges: list, parent: int = None):
    """Preorder traversal of the edges of the given tree. Traversed edges are saved as tuples,
    where the first element is the parent and second the child. Children are visited in
    increasing order.

    Args:
        tree (rx.PyGraph): tree to traverse
        node (int): root node
        edges (list): list of edges
        parent (int, optional): parent node. Defaults to None.
    """
    if node == None:
        return
    if parent != None:
        edges.append((parent, node))
    for n in sorted(tree.neighbors(node)):
        if n == parent:
            continue
        preorder_traversal(tree, n, edges, node)
