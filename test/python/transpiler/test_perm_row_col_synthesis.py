"""Test PermRowColSynthesis"""

from builtins import issubclass
from qiskit.circuit import quantumcircuit
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from qiskit.circuit.quantumregister import QuantumRegister

from qiskit.test import QiskitTestCase
from qiskit.transpiler.passes.synthesis.perm_row_col_synthesis import PermRowColSynthesis
from qiskit.transpiler.passes.synthesis.linear_functions_synthesis import LinearFunctionsSynthesis
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap, coupling
from qiskit.circuit.library.generalized_gates.linear_function import LinearFunction
from qiskit.circuit.library.generalized_gates.permutation import Permutation
from qiskit.transpiler.passes.optimization.collect_linear_functions import CollectLinearFunctions


class TestPermRowColSynthesis(QiskitTestCase):
    """Test PermRowColSynthesis"""

    def create_dag(self, circuits=[]):
        dag = DAGCircuit()
        for circuit in circuits:
            if isinstance(circuit, QuantumCircuit):
                qargs = range(len(circuit.qubits))
            else:
                qargs = range(len(circuit))

            dag.apply_operation_back(LinearFunction(circuit), qargs)

        return dag

    def test_is_a_subclass_of_linear_function_synthesis(self):
        """Test that the permrowcolsynthesis instance is a subclass of
        linear function synthesis"""
        self.assertTrue(issubclass(PermRowColSynthesis, LinearFunctionsSynthesis))

    def test_run_returns_a_dag(self):
        """Test the output type of run"""
        coupling = CouplingMap()
        circuit = QuantumCircuit()
        dag = circuit_to_dag(circuit)
        synthesis = PermRowColSynthesis(coupling)

        instance = synthesis.run(dag)

        self.assertIsInstance(instance, DAGCircuit)

    @patch("qiskit.transpiler.passes.synthesis.perm_row_col_synthesis.PermRowCol.perm_row_col")
    def test_run(self, mock_perm_row_col):
        """Test run method"""
        empty = QuantumCircuit(6)
        empty.cx(0, 1)
        empty.cx(1, 5)
        empty.cx(3, 1)
        empty.cx(1, 4)
        empty.cx(1, 3)
        empty.cx(3, 5)
        empty.cx(2, 1)
        collec = CollectLinearFunctions()
        e_dag = circuit_to_dag(empty)
        dag = collec.run(e_dag)

        coupling = CouplingMap()
        circ = QuantumCircuit(6)
        circ.cx(1, 0)
        circ.cx(0, 1)
        circ.cx(0, 3)
        circ.cx(3, 4)
        circ.cx(5, 4)
        circ.cx(4, 1)
        circ.cx(2, 5)
        circ.cx(1, 2)
        circ.cx(1, 4)
        circ.cx(5, 2)
        circ.cx(2, 5)
        circ.cx(5, 4)
        circ.cx(5, 2)

        perm = Permutation(6, [5, 3, 1, 0, 4, 2])

        mock_perm_row_col.return_value = (circ, perm)

        synthesis = PermRowColSynthesis(coupling)

        instance = synthesis.run(dag)

        ret_circ = dag_to_circuit(instance)

        cnots = []
        for d in ret_circ.data:
            if d.operation.name == "cx":
                cnots.append(d)

        ret_cnots = QuantumCircuit.from_instructions(cnots)

        self.assertTrue(mock_perm_row_col.called)


if __name__ == "__main__":
    unittest.main()
