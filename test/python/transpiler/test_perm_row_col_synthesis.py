"""Test PermRowColSynthesis"""

from builtins import issubclass
import unittest

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.transpiler.passes.synthesis.perm_row_col_synthesis import PermRowColSynthesis
from qiskit.transpiler.passes.synthesis.linear_functions_synthesis import LinearFunctionsSynthesis
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit

from qiskit.transpiler.synthesis.matrix_utils import build_random_parity_matrix
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.circuit.library.generalized_gates.linear_function import LinearFunction
from qiskit.providers.fake_provider import FakeTenerife, FakeManilaV2


class TestPermRowColSynthesis(QiskitTestCase):
    """Test PermRowColSynthesis"""

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

    def test_sanity_check(self):
        backend = FakeTenerife()
        data = backend.properties().to_dict()["gates"]
        coupling_list = [tuple(item["qubits"]) for item in data if item["gate"] == "cx"]
        coupling = CouplingMap(coupling_list)
#        print("coupling map:")
#        print(coupling)

        random_parity_matrix = build_random_parity_matrix(13, 5, 60).astype(int)
#        print("random parity matrix:")
#        print(random_parity_matrix)
        n = 5

        random_circuit = LinearFunction(random_parity_matrix).synthesize()
        random_circuit_copy = random_circuit.copy()
#        print("random circuit")
#        print(random_circuit)

        dag_permrowcol = circuit_to_dag(random_circuit)
        dag_linearfunction = circuit_to_dag(random_circuit_copy)

        synthesis_permrowcol = PermRowColSynthesis(coupling)
        synthesis_linearfunction = LinearFunctionsSynthesis()

        dag_permrowcol = synthesis_permrowcol.run(dag_permrowcol)
        dag_linearfunction = synthesis_linearfunction.run(dag_linearfunction)

        circuit_permrowcol = dag_to_circuit(dag_permrowcol)
        circuit_linearfunction = dag_to_circuit(dag_linearfunction)
#        print("circuits:")
#        print("permrowcol:")
#        print(circuit_permrowcol)
#        print("linearfunction:")
#        print(circuit_linearfunction)

        parity_mat_permrowcol = LinearFunction(circuit_permrowcol).linear.astype(int)
        parity_mat_linearfunction = LinearFunction(circuit_linearfunction).linear.astype(int)

#        print("parity matrices:")
#        print("permrowcol:")
#        print(parity_mat_permrowcol)
#        print("linearfunction")
#        print(parity_mat_linearfunction)

        self.assertTrue(np.array_equal(parity_mat_permrowcol, parity_mat_linearfunction))


if __name__ == "__main__":
    unittest.main()
