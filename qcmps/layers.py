from __future__ import annotations
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit

from .gates import Gate

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
    
class BrickWallLayerType(LayerType):
    
    def _build(self, n_qubits: int, parameters: list[float]) -> QuantumCircuit:
        circuit = QuantumCircuit(n_qubits)
        
        index = 0
        
        for qubit in range(0, n_qubits - 1, 2):
            gate_circuit = self.gate_type(parameters[index:index + self.gate_type.n_parameters()])
            index += self.gate_type.n_parameters()
            
            circuit.append(gate_circuit, [qubit, qubit + 1])
        
        for qubit in range(1, n_qubits - 1, 2):
            gate_circuit = self.gate_type(parameters[index:index + self.gate_type.n_parameters()])
            index += self.gate_type.n_parameters()
            
            circuit.append(gate_circuit, [qubit, qubit + 1])
        
        return circuit
    
    def _n_parameters(self, n_qubits: int) -> int:
        return self.gate_type.n_parameters() * (n_qubits - 1)