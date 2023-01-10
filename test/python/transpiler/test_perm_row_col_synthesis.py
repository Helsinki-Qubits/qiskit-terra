"""Test PermRowColSynthesis"""

from builtins import issubclass
import unittest
from unittest.mock import patch
import numpy as np

from qiskit.circuit.library.standard_gates.x import CXGate
from qiskit.circuit.quantumregister import QuantumRegister

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


class TestPermRowColSynthesis(QiskitTestCase):
    """Test PermRowColSynthesis"""

    def setUp(self):
        super().setUp()
        self.seed = 1234
        self.rng = np.random.default_rng(self.seed)
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

    def test_sanity_check(self):
        backend = FakeTenerife()
        data = backend.properties().to_dict()["gates"]
        coupling_list = [tuple(item["qubits"]) for item in data if item["gate"] == "cx"]
        coupling = CouplingMap(coupling_list)
        # print("coupling map:")
        # print(coupling)

        random_parity_matrix = build_random_parity_matrix(13, 5, 60).astype(int)
        # print("random parity matrix:")
        # print(random_parity_matrix)
        n = 5

        random_circuit = LinearFunction(random_parity_matrix).synthesize()
        random_circuit_copy = random_circuit.copy()
        # print("random circuit")
        # print(random_circuit)

        dag_permrowcol = circuit_to_dag(random_circuit)
        dag_linearfunction = circuit_to_dag(random_circuit_copy)

        synthesis_permrowcol = PermRowColSynthesis(coupling)
        synthesis_linearfunction = LinearFunctionsSynthesis()

        dag_permrowcol = synthesis_permrowcol.run(dag_permrowcol)
        dag_linearfunction = synthesis_linearfunction.run(dag_linearfunction)

        circuit_permrowcol = dag_to_circuit(dag_permrowcol)
        circuit_linearfunction = dag_to_circuit(dag_linearfunction)
        # print("circuits:")
        # print("permrowcol:")
        # print(circuit_permrowcol)
        # print("linearfunction:")
        # print(circuit_linearfunction)

        parity_mat_permrowcol = LinearFunction(circuit_permrowcol).linear.astype(int)
        parity_mat_linearfunction = LinearFunction(circuit_linearfunction).linear.astype(int)

        # print("parity matrices:")
        # print("permrowcol:")
        # print(parity_mat_permrowcol)
        # print("linearfunction")
        # print(parity_mat_linearfunction)

        # example random
        qc = self._random_cnot_circuit(5, 4)
        print(qc)

        self.assertTrue(np.array_equal(parity_mat_permrowcol, parity_mat_linearfunction))

    def _random_cnot_circuit(self, num_qubits, depth, seed=None):
        qr = QuantumRegister(num_qubits, "q")
        qc = QuantumCircuit(num_qubits)

        # if seed is None:
        #    seed = np.random.randint(0, np.iinfo(np.int32).max)
        # rng = np.random.default_rng(seed)

        operation = CXGate
        num_operands = 2
        for _ in range(depth):
            remaining_qubits = list(range(num_qubits))
            self.rng.shuffle(remaining_qubits)
            while remaining_qubits:
                if len(remaining_qubits) < 2:
                    break
                operands = [remaining_qubits.pop() for _ in range(num_operands)]
                register_operands = [qr[i] for i in operands]
                op = operation()

                qc.append(op, register_operands)

        return qc
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

        instance = CollectLinearFunctions().run(instance)

        # print(LinearFunction(dag_to_circuit(instance).reverse_ops().decompose()).linear.astype(int))
        # The circuit needs to be reversed for the permutation to work since it changes the rows not the columns
        composed = r_circuit.compose(
            dag_to_circuit(instance).reverse_ops().decompose().inverse(), qubits=range(5)
        )
        self.assertTrue(Operator(composed).equiv(Operator.from_label("I" * len(r_circuit.qubits))))

        self.assertTrue(
            Statevector.from_instruction(r_circuit).equiv(
                Statevector.from_instruction(dag_to_circuit(instance).reverse_ops().decompose())
            )
        )

    @patch("qiskit.transpiler.passes.synthesis.perm_row_col_synthesis.PermRowCol.perm_row_col")
    def test_run_with_mock(self, mock_perm_row_col):
        """Test run method with mocked perm_row_col"""
        parity_mat = np.array(
            [
                [0, 1, 1, 1, 1, 1],
                [1, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0, 1],
            ]
        )

        input_circ = LinearFunction(parity_mat).synthesize()

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
        # instance = CollectLinearFunctions().run(instance) # fails with this unless the resulting circuit below is decomposed
        res_circ = dag_to_circuit(instance)  # .decompose()

        self.assertTrue(
            Statevector.from_instruction(res_circ).equiv(Statevector.from_instruction(input_circ))
        )
        composed = input_circ.compose(res_circ.inverse(), qubits=range(len(res_circ.qubits)))
        self.assertTrue(Operator(composed).equiv(Operator.from_label("I" * len(input_circ.qubits))))


if __name__ == "__main__":
    unittest.main()
