from __future__ import annotations
from abc import ABC, abstractmethod

from qiskit import QuantumCircuit

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
    
class Block(QuantumCircuit, ABC):
    def __init__(self, n_qubits: int, parameters: list[float] | None = None, layers: int = 1):
        n_params = self.n_parameters(n_qubits, layers)
        
        if parameters is not None:
            if len(parameters) != n_params:
                raise ValueError(f"Expected {n_params} parameters, got {len(parameters)}.")
        
        super().__init__(n_qubits)
        self.layers = layers
        
    @staticmethod
    @abstractmethod
    def n_parameters(n_qubits: int, layers: int = 1) -> int:
        pass
    
class LUBlock(Block):
    def __init__(self, parameters: list[float], n_qubits: int, layers: int = 1):
        super().__init__(n_qubits, parameters, layers)
        
        index = 0
        
        for _ in range(layers):
            for qubit in range(n_qubits - 1):
                
                gate_circuit = CUGate(parameters[index:index + CUGate.n_parameters()])
                index += CUGate.n_parameters()
                
                self.append(gate_circuit, [qubit, qubit + 1])
                
    @staticmethod
    def n_parameters(n_qubits: int, layers: int = 1) -> int:
        return CUGate.n_parameters() * (n_qubits - 1) * layers
    
class FullUBlock(Block):
    def __init__(self, parameters: list[float], n_qubits: int, layers: int = 1):
        super().__init__(n_qubits, parameters, layers)
        
        index = 0
        
        for _ in range(layers):
            for qubit in range(n_qubits):
                
                gate_circuit = UGate(parameters[index:index + UGate.n_parameters()])
                index += UGate.n_parameters()
                
                self.append(gate_circuit, [qubit])
        
    @staticmethod
    def n_parameters(n_qubits: int, layers: int = 1) -> int:
        return UGate.n_parameters() * n_qubits * layers
    
class HFPreSet(Block):
    def __init__(self, n_qubits: int, block_entries: int, occupied: list[int]):
        super().__init__(n_qubits, None, 1)
        
        occupied = {i + block_entries - 1 for i in occupied}
        
        for qubit in occupied:
            self.x(qubit)
    
    @staticmethod
    def n_parameters(n_qubits: int, layers: int = 1) -> int:
        return 0
    
    
