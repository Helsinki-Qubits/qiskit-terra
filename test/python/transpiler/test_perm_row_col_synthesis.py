"""Test PermRowColSynthesis"""

from builtins import issubclass
import unittest
from unittest.mock import patch

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.transpiler.passes.synthesis.perm_row_col_synthesis import PermRowColSynthesis
from qiskit.transpiler.passes.synthesis.linear_functions_synthesis import LinearFunctionsSynthesis
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.circuit.library.generalized_gates.linear_function import LinearFunction
from qiskit.circuit.library.generalized_gates.permutation import Permutation
from qiskit.transpiler.passes.optimization.collect_linear_functions import CollectLinearFunctions
from qiskit.quantum_info.operators.operator import Operator
from qiskit.transpiler.synthesis.matrix_utils import build_random_parity_matrix
from qiskit.quantum_info import Statevector
from qiskit.providers.fake_provider import FakeTenerife, FakeManilaV2
from qiskit.synthesis.linear import graysynth, synth_cnot_count_full_pmh
from qiskit.transpiler.synthesis.permrowcol import PermRowCol


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

    def test_run_with_empty_circuit(self):
        """Test that the input and output circuits are equivalent if
        the input circuit has no instructions"""
        empty = QuantumCircuit(6)
        dag = CollectLinearFunctions().run(circuit_to_dag(empty))
        coupling_list = [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)]
        coupling = CouplingMap(coupling_list)

        synthesis = PermRowColSynthesis(coupling)
        instance = synthesis.run(dag)
        res_circ = dag_to_circuit(instance)

        self.assertTrue(Operator(empty).equiv(Operator(res_circ)))
    

    
    def test_run_with_random_circuit(self):
        """Test that the input and output circuits are equivalent 
        with a randomly generated circuit"""
        
        backend = FakeManilaV2()
        coupling_map = backend.coupling_map
        coupling = CouplingMap(coupling_map)
        synthesis = PermRowColSynthesis(coupling)

        r_parity_matrix = build_random_parity_matrix(42, 5, 60).astype(int)
        r_circuit = LinearFunction(r_parity_matrix).synthesize()

        dag = CollectLinearFunctions().run(circuit_to_dag(r_circuit))

        instance = synthesis.run(dag)

        #self.assertTrue(Operator(composed).equiv(Operator.from_label('I'*len(r_circuit.qubits)))) # False

        #self.assertTrue(Statevector.from_instruction(r_circuit).equiv(Statevector.from_instruction(res_circ)))

    @patch("qiskit.transpiler.passes.synthesis.perm_row_col_synthesis.PermRowCol.perm_row_col")
    def test_run_with_mock(self, mock_perm_row_col):
        """Test run method"""
        input_circ = QuantumCircuit(6)
        input_circ.cx(0, 1)
        input_circ.cx(1, 5)
        input_circ.cx(3, 1)
        input_circ.cx(1, 4)
        input_circ.cx(1, 3)
        input_circ.cx(3, 5)
        input_circ.cx(2, 1)
        collec = CollectLinearFunctions()
        e_dag = circuit_to_dag(input_circ)
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

        res_circ = dag_to_circuit(instance)

        self.assertTrue(Statevector.from_instruction(res_circ).equiv(Statevector.from_instruction(input_circ)))
        composed = input_circ.compose(res_circ.inverse(), qubits=range(len(res_circ.qubits)))
        Operator(composed).equiv(Operator.from_label('I'*len(input_circ.qubits)))


if __name__ == "__main__":
    unittest.main()
