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
    
class SO4Gate(Gate):
    def __init__(self, parameters: list[float]):
        super().__init__(parameters, 2)

        (
            theta_01,
            theta_02,
            theta_03,
            theta_12,
            theta_13,
            theta_23,
        ) = parameters

        # Rotace |00> <-> |01>
        self.cry(
            2 * theta_01,
            control_qubit=1,
            target_qubit=0,
            ctrl_state=0,
        )

        # Rotace |00> <-> |10>
        self.cry(
            2 * theta_02,
            control_qubit=0,
            target_qubit=1,
            ctrl_state=0,
        )

        # Po změně báze:
        # |00> <-> |11>
        # |01> <-> |10>
        self.cx(0, 1)

        self.cry(
            2 * theta_03,
            control_qubit=1,
            target_qubit=0,
            ctrl_state=0,
        )

        self.cry(
            2 * theta_12,
            control_qubit=1,
            target_qubit=0,
            ctrl_state=1,
        )

        self.cx(0, 1)

        # Rotace |01> <-> |11>
        self.cry(
            2 * theta_13,
            control_qubit=0,
            target_qubit=1,
            ctrl_state=1,
        )

        # Rotace |10> <-> |11>
        self.cry(
            2 * theta_23,
            control_qubit=1,
            target_qubit=0,
            ctrl_state=1,
        )

    @staticmethod
    def n_parameters() -> int:
        return 6