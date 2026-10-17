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
    
class IUGate(Gate):
    def __init__(self, parameters: list[float]):
        super().__init__(parameters, 1)
        
        alpha, beta, gamma = parameters
        
        self.rx(alpha, 0)
        self.ry(beta, 0)
        self.rz(gamma, 0)
        
    @staticmethod
    def n_parameters() -> int:
        return 3
    
class ICUGate(Gate):
    def __init__(self, parameters: list[float]):
        super().__init__(parameters, 2)
        
        alpha, beta, gamma = parameters
        
        self.rxx(alpha, 0, 1)
        self.ryy(beta, 0, 1)
        self.rzz(gamma, 0, 1)
        
    @staticmethod
    def n_parameters() -> int:
        return 3

class OPUGate(Gate):
    def __init__(self, parameters: list[float]):
        super().__init__(parameters, 1)
        
        self.ry(parameters[0], 0)
        
    @staticmethod
    def n_parameters() -> int:
        return 1
    
class OPCUGate(Gate):
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
                
                gate_circuit = OPCUGate(parameters[index:index + OPCUGate.n_parameters()])
                index += OPCUGate.n_parameters()
                
                self.append(gate_circuit, [qubit, qubit + 1])
                
    @staticmethod
    def n_parameters(n_qubits: int, layers: int = 1, *args) -> int:
        return OPCUGate.n_parameters() * (n_qubits - 1) * layers
    
class FullUBlock(Block):
    def __init__(self, parameters: list[float], n_qubits: int, layers: int = 1):
        
        n_params = FullUBlock.n_parameters(n_qubits, layers)
        super().__init__(n_qubits, n_params, parameters, layers)
        
        index = 0
        
        for _ in range(layers):
            for qubit in range(n_qubits):
                
                gate_circuit = OPUGate(parameters[index:index + OPUGate.n_parameters()])
                index += OPUGate.n_parameters()
                
                self.append(gate_circuit, [qubit])
        
    @staticmethod
    def n_parameters(n_qubits: int, layers: int = 1, *args) -> int:
        return OPUGate.n_parameters() * n_qubits * layers
    
    
