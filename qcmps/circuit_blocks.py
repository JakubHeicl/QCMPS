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
        
class Block(QuantumCircuit, ABC):
    def __init__(self, n_qubits: int, n_params, parameters: list[float] | None = None, layers: int = 1):
        
        if parameters is not None:
            if len(parameters) != n_params:
                raise ValueError(f"Expected {n_params} parameters, got {len(parameters)}.")
        
        super().__init__(n_qubits)
        self.layers = layers
        
    @staticmethod
    @abstractmethod
    def n_parameters(n_qubits: int, layers: int = 1, *args) -> int:
        pass
    
class LUBlock(Block):
    def __init__(self, parameters: list[float], n_qubits: int, layers: int = 1):
        n_params = LUBlock.n_parameters(n_qubits, layers)
        super().__init__(n_qubits, n_params, parameters, layers)
        
        index = 0
        
        for _ in range(layers):
            for qubit in range(n_qubits - 1):
                
                gate_circuit = CUGate(parameters[index:index + CUGate.n_parameters()])
                index += CUGate.n_parameters()
                
                self.append(gate_circuit, [qubit, qubit + 1])
                
    @staticmethod
    def n_parameters(n_qubits: int, layers: int = 1, *args) -> int:
        return CUGate.n_parameters() * (n_qubits - 1) * layers
    
class FullUBlock(Block):
    def __init__(self, parameters: list[float], n_qubits: int, layers: int = 1):
        n_params = FullUBlock.n_parameters(n_qubits, layers)
        super().__init__(n_qubits, n_params, parameters, layers)
        
        index = 0
        
        for _ in range(layers):
            for qubit in range(n_qubits):
                
                gate_circuit = UGate(parameters[index:index + UGate.n_parameters()])
                index += UGate.n_parameters()
                
                self.append(gate_circuit, [qubit])
        
    @staticmethod
    def n_parameters(n_qubits: int, layers: int = 1, *args) -> int:
        return UGate.n_parameters() * n_qubits * layers
    
    
