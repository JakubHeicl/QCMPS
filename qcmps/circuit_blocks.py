from __future__ import annotations
from abc import ABC, abstractmethod

from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate

class Gate(QuantumCircuit, ABC):
    
    def __init__(self, parameters: list[float], n_qubits: int):
        n_params = self.n_parameters()

        if len(parameters) != n_params:
            raise ValueError(f"Expected {n_params} parameters, got {len(parameters)}.")
        
        super().__init__(n_qubits)
        
    @staticmethod
    @abstractmethod
    def n_parameters() -> int:
        pass

class UGate(Gate):
    def __init__(self, parameters: list[float]):
        super().__init__(parameters, 1)
        
        theta, phi, lam = parameters
        
        self.u(theta, phi, lam, 0)
        
    @staticmethod
    def n_parameters() -> int:
        return 3

class CUGate(Gate):
    def __init__(self, parameters: list[float]):
        super().__init__(parameters, 2)

        theta, phi, lam = parameters
        
        self.cu(theta, phi, lam, 0.0, 0, 1)
        
    @staticmethod
    def n_parameters() -> int:
        return 3
    
class XYZGate(Gate):
    def __init__(self, parameters: list[float]):
        super().__init__(parameters, 1)
        
        alpha, beta, gamma = parameters
        
        self.rx(alpha, 0)
        self.ry(beta, 0)
        self.rz(gamma, 0)
        
    @staticmethod
    def n_parameters() -> int:
        return 3
    
class XYZEntanglingGate(Gate):
    def __init__(self, parameters: list[float]):
        super().__init__(parameters, 2)
        
        alpha, beta, gamma = parameters
        
        self.rxx(alpha, 0, 1)
        self.ryy(beta, 0, 1)
        self.rzz(gamma, 0, 1)
        
    @staticmethod
    def n_parameters() -> int:
        return 3

class OneParameterGate(Gate):
    def __init__(self, parameters: list[float]):
        super().__init__(parameters, 1)
        
        self.ry(parameters[0], 0)
        
    @staticmethod
    def n_parameters() -> int:
        return 1
    
class OneParameterEntanglingGate(Gate):
    def __init__(self, parameters: list[float]):
        super().__init__(parameters, 2)
        
        self.rxx(parameters[0], 0, 1)
        
    @staticmethod
    def n_parameters() -> int:
        return 1

class DoubleExcitationGate(Gate):
    def __init__(self, parameters: list[float]):
        super().__init__(parameters, 4)

        theta = parameters[0]
        
        self.cx(0, 1)
        self.cx(0, 2)
        self.cx(0, 3)
        
        gate = RYGate(-2*theta).control(3, ctrl_state='110')
        self.append(gate, [1, 2, 3, 0])
        
        self.cx(0, 3)
        self.cx(0, 2)
        self.cx(0, 1)
        
    @staticmethod
    def n_parameters() -> int:
        return 1

class LayerType(ABC):
    def __init__(self, gate_type: type[Gate]):
        self.gate_type = gate_type
        
    def build(self, n_qubits: int, parameters: list[float]) -> QuantumCircuit:
        
        if len(parameters) != self.n_parameters(n_qubits):
            raise ValueError(f"Expected {self.n_parameters(n_qubits)} parameters, got {len(parameters)}.")
        
        return self._build(n_qubits, parameters)
    
    @abstractmethod
    def _build(self, n_qubits: int, parameters: list[float]) -> QuantumCircuit:
        pass
        
    def n_parameters(self, n_qubits: int) -> int:
        if n_qubits < 1:
            raise ValueError("Number of qubits must be at least 1.")
        
        return self._n_parameters(n_qubits)
    
    @abstractmethod
    def _n_parameters(self, n_qubits: int) -> int:
        pass
    
class EntanglingLayerType(LayerType):
    
    def _build(self, n_qubits: int, parameters: list[float]) -> QuantumCircuit:
        circuit = QuantumCircuit(n_qubits)
        
        index = 0
        
        for qubit in range(n_qubits - 1):
            gate_circuit = self.gate_type(parameters[index:index + self.gate_type.n_parameters()])
            index += self.gate_type.n_parameters()
            
            circuit.append(gate_circuit, [qubit, qubit + 1])
        
        return circuit
    
    def _n_parameters(self, n_qubits: int) -> int:
        return self.gate_type.n_parameters() * (n_qubits - 1)

class SingleQubitLayerType(LayerType):
    
    def _build(self, n_qubits: int, parameters: list[float]) -> QuantumCircuit:
        circuit = QuantumCircuit(n_qubits)
        
        index = 0
        
        for qubit in range(n_qubits):
            gate_circuit = self.gate_type(parameters[index:index + self.gate_type.n_parameters()])
            index += self.gate_type.n_parameters()
            
            circuit.append(gate_circuit, [qubit])
        
        return circuit
    
    def _n_parameters(self, n_qubits: int) -> int:
        return self.gate_type.n_parameters() * n_qubits