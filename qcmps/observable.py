from __future__ import annotations
from .ansatz import Ansatz

from qiskit.quantum_info import SparsePauliOp
import numpy as np

class Observable:
    def __init__(self, operator: SparsePauliOp, qubits: list[int], constant: float = 0.0, label: str = "Observable"):

        if not isinstance(operator, SparsePauliOp):
            raise TypeError("operator must be an instance of SparsePauliOp.")
        
        if not isinstance(qubits, list) or not all(isinstance(q, int) for q in qubits):
            raise TypeError("qubits must be a list of integers.")
        
        if not isinstance(constant, (int, float)):
            raise TypeError("constant must be a numeric type.")
        
        if not qubits:
            raise ValueError("At least one qubit must be specified.")
        
        if len(set(qubits)) != len(qubits):
            raise ValueError("Duplicate qubit indices are not allowed.")
        
        if operator.num_qubits != len(qubits):
            raise ValueError("The number of operator qubits must match the number of target qubits.")
        
        self.operator = operator
        self.qubits = qubits
        self.constant = constant
        self.label = label
        
    def __call__(self, ansatz: Ansatz) -> float:
        return self.exp(ansatz) 
        
    def exp(self, ansatz: Ansatz) -> float:
        
        state = ansatz.statevector()
        expectation_value = state.expectation_value(self.operator, qargs = self.qubits)
        
        return float(np.real_if_close(expectation_value)) + self.constant
    
    def var(self, ansatz: Ansatz) -> float:
        
        state = ansatz.statevector()
        
        expectation_value = state.expectation_value(self.operator, qargs = self.qubits)
        expectation_value_squared = state.expectation_value(self.operator @ self.operator, qargs = self.qubits)
        
        variance = expectation_value_squared - expectation_value**2
        
        return float(np.real_if_close(variance))
    
    @staticmethod
    def particle_number(qubits: list[int]) -> Observable:
        
        n_qubits = len(qubits)
        operator = SparsePauliOp.from_sparse_list([("Z", [i], -0.5) for i in range(n_qubits)], num_qubits=n_qubits)
        
        return Observable(operator=operator, qubits=qubits, constant = 0.5 * n_qubits, label = "Particle Number")

    @staticmethod
    def alpha_particle_number(qubits: list[int]) -> Observable:
        
        alpha_qubits, _ = Observable._split_interleaved_spin_qubits(qubits)
        observable = Observable.particle_number(alpha_qubits)
        observable.label = "Alpha Particle Number"
        
        return observable

    @staticmethod
    def beta_particle_number(qubits: list[int]) -> Observable:
        
        _, beta_qubits = Observable._split_interleaved_spin_qubits(qubits)
        observable = Observable.particle_number(beta_qubits)
        observable.label = "Beta Particle Number"
        
        return observable

    @staticmethod
    def spin_imbalance(qubits: list[int]) -> Observable:

        Observable._split_interleaved_spin_qubits(qubits)
        n_qubits = len(qubits)

        operator = SparsePauliOp.from_sparse_list([("Z", [index], -0.5 if index % 2 == 0 else 0.5) for index in range(n_qubits)], num_qubits=n_qubits)

        return Observable(operator=operator, qubits=qubits, label="Spin Imbalance (2Sz)")

    @staticmethod
    def _split_interleaved_spin_qubits(qubits: list[int]) -> tuple[list[int], list[int]]:
        
        if not isinstance(qubits, list) or not all(isinstance(qubit, int) for qubit in qubits):
            raise TypeError("qubits must be a list of integers.")

        if not qubits:
            raise ValueError("At least one alpha/beta qubit pair must be specified.")

        if len(set(qubits)) != len(qubits):
            raise ValueError("Duplicate qubit indices are not allowed.")

        if len(qubits) % 2 != 0:
            raise ValueError("Interleaved spin-orbital qubits must contain complete alpha/beta pairs.")

        return qubits[::2], qubits[1::2]
    
    @staticmethod
    def zero_state_infidelity(qubits: list[int]) -> ZeroStateInfidelity:
        
        return ZeroStateInfidelity(qubits)
    
class ZeroStateInfidelity:
    def __init__(self, qubits: list[int]):
        
        if not isinstance(qubits, list) or not all(isinstance(q, int) for q in qubits):
            raise TypeError("qubits must be a list of integers.")
        
        if not qubits:
            raise ValueError("At least one qubit must be specified.")
        
        if len(set(qubits)) != len(qubits):
            raise ValueError("Duplicate qubit indices are not allowed.")
        
        self.qubits = qubits
        self.label = "Zero State Infidelity"
        
    def __call__(self, ansatz: Ansatz) -> float:
        return self.exp(ansatz)
        
    def exp(self, ansatz: Ansatz) -> float:
        probabilities = ansatz.statevector().probabilities(qargs=self.qubits)
        
        return 1.0 - float(probabilities[0]) 
    
    def var(self, ansatz: Ansatz) -> float:
        infidelity = self(ansatz)
        return infidelity * (1.0 - infidelity)
