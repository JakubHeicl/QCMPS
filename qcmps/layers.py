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
    
class StarLayerType(LayerType):
    
    def __init__(self, gate_type: type[Gate], center_qubit: int = 0, target_qubits: list[int] | None = None):
        super().__init__(gate_type)
        self.center_qubit = center_qubit
        self.target_qubits = target_qubits
        
    def _targets(self, n_qubits: int) -> list[int]:
        if self.target_qubits is None:
            return [qubit for qubit in range(n_qubits) if qubit != self.center_qubit]
    
        if self.center_qubit in self.target_qubits:
            raise ValueError("Center qubit cannot be in the list of target qubits.")
        
        if any(qubit >= n_qubits or qubit < 0 for qubit in self.target_qubits):
            raise ValueError("Target qubits must be within the range of available qubits.")

        return self.target_qubits

    def _build(self, n_qubits: int, parameters: list[float]) -> QuantumCircuit:
        circuit = QuantumCircuit(n_qubits)
        
        index = 0
        
        for qubit in self._targets(n_qubits):
            if qubit == self.center_qubit:
                continue
            
            gate_circuit = self.gate_type(parameters[index:index + self.gate_type.n_parameters()])
            index += self.gate_type.n_parameters()
            
            circuit.append(gate_circuit, [self.center_qubit, qubit])
        
        return circuit
    
    def _n_parameters(self, n_qubits: int) -> int:
        return self.gate_type.n_parameters() * len(self._targets(n_qubits))
    
class LUBondConnectedLayerType(LayerType):
    
    def __init__(self, gate_type: type[Gate], bond_qubits: list[int] | None = None):
        super().__init__(gate_type)
        self.bond_qubits = bond_qubits
        
    def _build(self, n_qubits: int, parameters: list[float]) -> QuantumCircuit:
        circuit = QuantumCircuit(n_qubits)
        
        index = 0
        
        if self.bond_qubits is None:
            self.bond_qubits = list(range(1, n_qubits))
        
        for i in range(len(self.bond_qubits) - 1):
            qubit1 = self.bond_qubits[i]
            qubit2 = self.bond_qubits[i + 1]
            
            gate_circuit = self.gate_type(parameters[index:index + self.gate_type.n_parameters()])
            index += self.gate_type.n_parameters()
            
            circuit.append(gate_circuit, [qubit1, qubit2])
            
        return circuit
    
    def _n_parameters(self, n_qubits: int) -> int:
        if self.bond_qubits is None:
            return self.gate_type.n_parameters() * (n_qubits - 2)
        else:
            return self.gate_type.n_parameters() * (len(self.bond_qubits) - 1)
            
        
        
        
