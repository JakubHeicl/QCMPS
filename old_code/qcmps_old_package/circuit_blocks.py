from __future__ import annotations
from functools import lru_cache
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate as QiskitGate
from qiskit.circuit.library import XXPlusYYGate
from scipy.linalg import expm

_PAULI_MATRICES = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}

@lru_cache(maxsize=None)
def _pauli_basis(n_qubits: int) -> tuple[np.ndarray, ...]:

    basis = []
    for label in product("IXYZ", repeat=n_qubits):
        matrix = np.array([[1.0]], dtype=complex)
        for char in label:
            matrix = np.kron(matrix, _PAULI_MATRICES[char])
        basis.append(matrix)

    return tuple(basis)

class HFPreSet(QuantumCircuit):
    def __init__(self, n_entries: int):

        super().__init__(n_entries)
        self.x(0)

class Gate(QuantumCircuit):
    def __init__(self, parameters: list[float], n_params: int, n_qubits: int):
        super().__init__(n_qubits)

        if len(parameters) != n_params:
            raise ValueError(f"Expected {n_params} parameters, got {len(parameters)}.")
    
    @staticmethod
    def n_parameters() -> int:
        raise NotImplementedError("Subclasses must implement n_parameters()")

class UGate(Gate):
    def __init__(self, parameters: list[float]):
        super().__init__(parameters, 3, 2)

        a, b, c = parameters

        self.cx(0, 1)
        self.ry(a, 0)
        self.rz(b, 0)
        self.ry(c, 0)
        self.cx(0, 1)

    @staticmethod
    def n_parameters() -> int:
        return 3

class CUGate(Gate):
    def __init__(self, parameters: list[float]):
        super().__init__(parameters, 3, 2)

        theta, phi, lam = parameters

        self.cu(theta, phi, lam, 0.0, 0, 1)

    @staticmethod
    def n_parameters() -> int:
        return 3

class PCGate(Gate):
    def __init__(self, parameters: list[float]):
        super().__init__(parameters, 5, 2)

        theta, phi, pre_phase, post_phase, pair_phase = parameters

        self.rz(-pre_phase, 0)
        self.rz(pre_phase, 1)
        self.append(XXPlusYYGate(2 * theta, phi, label="PC"), [0, 1])
        self.rz(-post_phase, 0)
        self.rz(post_phase, 1)
        self.cp(pair_phase, 0, 1)

    @staticmethod
    def n_parameters() -> int:
        return 5

class GeneralUnitaryMatrixGate(QiskitGate):
    def __init__(self, n_qubits: int, parameters: list[float]):

        n_params = 4**n_qubits
        if len(parameters) != n_params:
            raise ValueError(f"Expected {n_params} parameters, got {len(parameters)}.")

        super().__init__("GU", n_qubits, list(parameters))

    def to_matrix(self) -> np.ndarray:

        coefficients = np.asarray([float(param) for param in self.params], dtype=float)
        generator = np.zeros((2**self.num_qubits, 2**self.num_qubits), dtype=complex)

        for coeff, basis_matrix in zip(coefficients, _pauli_basis(self.num_qubits), strict=True):
            generator += coeff * basis_matrix

        return expm(1.0j * generator)

class Block(QuantumCircuit):

    gate_cls: type[Gate] | None = None

    def __init__(self, n_entries: int, parameters: list[float], layers = 1):
        super().__init__(n_entries)

        if self.gate_cls is None:
            raise ValueError("gate_cls must be set before instantiating Block subclasses.")
       
        self.layers = layers
        self.gate_parameters = self.gate_cls.n_parameters()

    @classmethod
    def n_parameters(cls, n_entries: int, layers: int = 1) -> int:
        raise NotImplementedError("Subclasses must implement n_parameters()")

class AUBlock(Block):

    gate_cls = UGate

    def __init__(self, n_entries: int, parameters: list[float], layers = 1):
        super().__init__(n_entries, parameters, layers)

        index = 0

        for l in range(layers):
            for i in range(n_entries):
                for j in range(n_entries):

                    if i == j:
                        continue

                    gate_circuit = self.gate_cls(parameters[index:index+self.gate_parameters])
                    index += self.gate_parameters

                    self.append(gate_circuit.to_gate(), [i, j])

    @classmethod
    def n_parameters(cls, n_entries: int, layers: int = 1) -> int:
        return cls.gate_cls.n_parameters() * n_entries * (n_entries - 1) * layers

class LUBlock(Block):
    
    gate_cls = UGate

    def __init__(self, n_entries: int, parameters: list[float], layers = 1):
        super().__init__(n_entries, parameters, layers)

        index = 0

        for l in range(layers):
            for i in range(n_entries - 1):

                gate_circuit = self.gate_cls(parameters[index:index+self.gate_parameters])
                index += self.gate_parameters

                self.append(gate_circuit.to_gate(), [i, i + 1])

    @classmethod
    def n_parameters(cls, n_entries: int, layers: int = 1) -> int:
        return cls.gate_cls.n_parameters() * (n_entries - 1) * layers

class LPCBlock(Block):
    
    gate_cls = PCGate

    def __init__(self, n_entries: int, parameters: list[float], layers = 1):
        super().__init__(n_entries, parameters, layers)

        index = 0

        for l in range(layers):
            for i in range(n_entries - 1):

                gate_circuit = self.gate_cls(parameters[index:index+self.gate_parameters])
                index += self.gate_parameters

                self.append(gate_circuit.to_gate(), [i, i + 1])

    @classmethod
    def n_parameters(cls, n_entries: int, layers: int = 1) -> int:
        return cls.gate_cls.n_parameters() * (n_entries - 1) * layers
    
class APCBlock(Block):

    gate_cls = PCGate

    def __init__(self, n_entries: int, parameters: list[float], layers = 1):
        super().__init__(n_entries, parameters, layers)

        index = 0

        for l in range(layers):
            for i in range(n_entries):
                for j in range(n_entries):

                    if i == j:
                        continue

                    gate_circuit = self.gate_cls(parameters[index:index+self.gate_parameters])
                    index += self.gate_parameters

                    self.append(gate_circuit.to_gate(), [i, j])

    @classmethod
    def n_parameters(cls, n_entries: int, layers: int = 1) -> int:
        return cls.gate_cls.n_parameters() * n_entries * (n_entries - 1) * layers


class ACUBlock(Block):

    gate_cls = CUGate

    def __init__(self, n_entries: int, parameters: list[float], layers = 1):
        super().__init__(n_entries, parameters, layers)

        index = 0

        for l in range(layers):
            for i in range(n_entries):
                for j in range(n_entries):

                    if i == j:
                        continue

                    gate_circuit = self.gate_cls(parameters[index:index+self.gate_parameters])
                    index += self.gate_parameters

                    self.append(gate_circuit.to_gate(), [i, j])

    @classmethod
    def n_parameters(cls, n_entries: int, layers: int = 1) -> int:
        return cls.gate_cls.n_parameters() * n_entries * (n_entries - 1) * layers

class GUBlock(QuantumCircuit):
    def __init__(self, n_entries: int, parameters: list[float], layers = 1):
        super().__init__(n_entries)

        n_layer_params = self.n_parameters(n_entries, 1)
        if len(parameters) != self.n_parameters(n_entries, layers):
            raise ValueError(
                f"Expected {self.n_parameters(n_entries, layers)} parameters, got {len(parameters)}."
            )

        index = 0
        for _ in range(layers):
            self.append(
                GeneralUnitaryMatrixGate(n_entries, parameters[index : index + n_layer_params]),
                list(range(n_entries)),
            )
            index += n_layer_params

    @classmethod
    def n_parameters(cls, n_entries: int, layers: int = 1) -> int:
        return (4**n_entries) * layers
