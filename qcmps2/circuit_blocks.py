from __future__ import annotations
from abc import ABC, abstractmethod
from itertools import combinations

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
    
class HFPreSetBlock(Block):
    def __init__(self, n_qubits: int, block_entries: int, occupied_indices: list[int]):
        n_params = HFPreSetBlock.n_parameters(n_qubits, 1)
        super().__init__(n_qubits, n_params, None, 1)
        
        occupied_indices = {i + block_entries - 1 for i in occupied_indices}
        
        for qubit in occupied_indices:
            self.x(qubit)
    
    @staticmethod
    def n_parameters(n_qubits: int, layers: int = 1) -> int:
        return 0
    
class DoubleExcitationBlock(Block):
    def __init__(self, parameters, n_qubits: int, n_orbs: int, block_entries: int, occupied: list[int]):
        n_params = DoubleExcitationBlock.n_parameters(n_qubits, 1, n_orbs, occupied)
        super().__init__(n_qubits, n_params, parameters, 1)
        
        offset = block_entries - 1
        
        occupied = tuple(sorted(set(occupied)))
        occupied_set = set(occupied)
        
        occupied_indices = [orbital + offset for orbital in occupied]
        unoccupied_indices = [offset + orbital for orbital in range(n_orbs) if orbital not in occupied_set]
        
        index = 0
        
        for (i, j) in combinations(occupied_indices, 2):
            for (a, b) in combinations(unoccupied_indices, 2):
                self.append(DoubleExcitationGate(parameters[index:index + 1]), [i, j, a, b])
                index += 1
                
    @staticmethod
    def n_parameters(n_qubits: int, layers: int = 1, n_orbs: int = 0, occupied: list[int] = None) -> int:
        if occupied is None:
            raise ValueError("Occupied indices must be provided to calculate the number of parameters.")
        
        n_occupied = len(occupied)
        n_unoccupied = n_orbs - n_occupied
        
        return DoubleExcitationGate.n_parameters() * (n_occupied * (n_occupied - 1) // 2) * (n_unoccupied * (n_unoccupied - 1) // 2)
    
    
