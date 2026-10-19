from __future__ import annotations
from .layers import LayerType
from .gates import DoubleExcitationGate

import numpy as np
from typing import Callable
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms.gradients import ReverseEstimatorGradient

from abc import ABC
from itertools import combinations

class Component(ABC, QuantumCircuit):

    def __init__(self, n_qubits: int, n_parameters: int, name: str):
        super().__init__(n_qubits)
        
        self.n_qubits = n_qubits
        self.n_parameters = n_parameters
        self.name = name
        
        if self.n_parameters > 0:
            self.parameters_symbols = ParameterVector(f"{self.name}_theta", self.n_parameters)
            self.parameters_values = np.zeros(self.n_parameters)
        else:
            self.parameters_symbols = None
            self.parameters_values = None
        
    def update_parameters(self, new_parameters: np.ndarray) -> None:
        
        if self.n_parameters == 0:
            raise ValueError("This component has no parameters to update.")
        
        if len(new_parameters) != self.n_parameters:
            raise ValueError(f"Expected {self.n_parameters} parameters, got {len(new_parameters)}.")
        
        self.parameters_values = new_parameters.copy()
        
    def current_parameters(self) -> np.ndarray:
        
        if self.n_parameters == 0:
            raise ValueError("This component has no parameters.")
        
        return self.parameters_values.copy()
    
    def uniform_parameters(self, min = 0.0, max = 2 * np.pi) -> None:
        self.parameters_values = np.random.uniform(min, max, self.n_parameters)
        
    def perturb_parameters(self, perturbation_strength: float) -> None:
        
        perturbation = np.random.uniform(-perturbation_strength, perturbation_strength, self.n_parameters)
        
        self.parameters_values += perturbation
        
    def resize_parameters(self, new_n_parameters: int) -> None:
        
        old_n_parameters = self.n_parameters
        self.n_parameters = new_n_parameters
        
        if new_n_parameters == old_n_parameters:
            return
        
        if new_n_parameters > old_n_parameters:
            if self.parameters_values is not None:
                new_parameters_values = np.zeros(new_n_parameters)
                new_parameters_values[:old_n_parameters] = self.parameters_values
                self.parameters_values = new_parameters_values
                self.parameters_symbols.resize(new_n_parameters)
            else:
                self.parameters_values = np.zeros(new_n_parameters)
                self.parameters_symbols = ParameterVector(f"{self.name}_theta", new_n_parameters)
        else:
            if self.parameters_values is not None:
                self.parameters_values = self.parameters_values[:new_n_parameters]
                self.parameters_symbols.resize(new_n_parameters)
            else:
                self.parameters_symbols = None
                self.parameters_values = None
                
    @staticmethod
    def from_fixed_circuit(circuit: QuantumCircuit) -> Component:
        
        if circuit.num_parameters != 0:
            raise ValueError("The circuit must have all parameters bound.")
        
        component = Component(circuit.num_qubits, 0, "CustomComponent")
        component.compose(circuit, inplace=True)
        
        return component

class Ansatz(QuantumCircuit):
    def __init__(self, n_qubits: int):
        super().__init__(n_qubits)
        
        self.n_qubits = n_qubits
        self.components: list[Component] = []
        self.component_qubits: list[list[int]] = []
        
        self.hamiltonian: SparsePauliOp | None = None
        self.hamiltonian_qubits: list[int] = list(range(self.n_qubits))
        self.nuclear_repulsion: float | None = None
        
        self.gradient_calculator = ReverseEstimatorGradient()
        
    def add_component(self, component: Component, qubits: list[int]):
        
        if len(qubits) != component.n_qubits:
            raise ValueError(f"Expected {component.n_qubits} qubits for the component, got {len(qubits)}.")
        
        self.components.append(component)
        self.component_qubits.append(qubits)
        
        self.compose(component, qubits=qubits, inplace=True)
        
    def rebuild(self) -> None:
        self.clear()
        
        for component, qubits in zip(self.components, self.component_qubits):
            self.compose(component, qubits=qubits, inplace=True)
        
    def add_hamiltonian(self, hamiltonian: SparsePauliOp, hamiltonian_qubits: list[int] | None = None):
        self.hamiltonian = hamiltonian
        self.hamiltonian_qubits = hamiltonian_qubits if hamiltonian_qubits is not None else list(range(self.n_qubits))

    def add_nuclear_repulsion(self, nuclear_repulsion: float):
        self.nuclear_repulsion = nuclear_repulsion
            
    def bound_circuit(self) -> QuantumCircuit:
        parameter_map = {}

        for component in self.components:
            if component.n_parameters > 0:
                parameter_map.update(zip(
                        component.parameters_symbols,
                        component.parameters_values,
                        strict=True,
                    )
                )

        return QuantumCircuit.assign_parameters(self, parameter_map, inplace=False)
    
    def energy_objective(self, component: Component) -> Callable[[np.ndarray], float]:
        
        if not any(item is component for item in self.components):
            raise ValueError("The specified component is not part of the ansatz.")
        
        def objective_function(x: np.ndarray) -> float:
            component.update_parameters(x)
            return self.energy()
        
        return objective_function
    
    def energy_gradient_function(self, component: Component) -> Callable[[np.ndarray], np.ndarray]:
        
        if not any(item is component for item in self.components):
            raise ValueError("The specified component is not part of the ansatz.")
        
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian has not been added.")
        
        def gradient_function(x: np.ndarray) -> np.ndarray:
            component.update_parameters(x)
            
            result = self.gradient_calculator.run([self], [self.hamiltonian], [x], [list(component.parameters_symbols)]).result()
            
            return np.asarray(result.gradients[0], dtype=float)
        
        return gradient_function
    
    def zero_state_objective(self, component: Component, target_qubits: list[int]) -> Callable[[np.ndarray], float]:
        
        if not any(item is component for item in self.components):
            raise ValueError("The specified component is not part of the ansatz.")
        
        def objective_function(x: np.ndarray) -> float:
            component.update_parameters(x)
            return self.zero_state_infidelity(target_qubits)
        
        return objective_function
    
    def zero_state_infidelity(self, target_qubits: list[int]) -> float:
        
        circuit = self.bound_circuit()
        state = Statevector.from_instruction(circuit)
        
        physical_zero_probability = state.probabilities(qargs=target_qubits)[0]
        return 1 - float(physical_zero_probability)

    def energy(self) -> float:
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian has not been added.")

        circuit = self.bound_circuit()
        state = Statevector.from_instruction(circuit)

        energy = state.expectation_value(self.hamiltonian, qargs=self.hamiltonian_qubits)
        return float(np.real_if_close(energy))
    
    def total_energy(self) -> float:
        if self.nuclear_repulsion is None:
            raise ValueError("Nuclear repulsion energy has not been added to the ansatz.")
        
        return self.energy() + self.nuclear_repulsion
    
    def print_energy(self, *args, **kwargs):
        energy = self.energy()
        print(f"Energy: {energy:.12f}")
    
    def print_total_energy(self, *args, **kwargs):
        total_energy = self.total_energy()
        print(f"Total Energy: {total_energy:.12f}")

class HFPreset(Component):
    def __init__(self, n_qubits: int, occupied_indices: list[int]):
        super().__init__(n_qubits, 0, "HFPreset")
        
        occupied_indices = set(occupied_indices)
        
        for qubit in occupied_indices:
            self.x(qubit)

class DoubleExcitation(Component):
    def __init__(self, n_qubits: int, occupied: list[int]):
        
        if len(set(occupied)) != len(occupied):
            raise ValueError("Occupied orbital indices must be unique.")

        if any(index < 0 or index >= n_qubits for index in occupied):
            raise ValueError("Occupied orbital index is out of range.")
        
        n_occupied = len(occupied)
        n_unoccupied = n_qubits - n_occupied
        
        n_parameters = DoubleExcitationGate.n_parameters() * (n_occupied * (n_occupied - 1) // 2) * (n_unoccupied * (n_unoccupied - 1) // 2)

        super().__init__(n_qubits, n_parameters, "DoubleExcitationInit")
        
        occupied = tuple(sorted(set(occupied)))
        occupied_set = set(occupied)
        
        occupied_indices = [orbital for orbital in occupied]
        unoccupied_indices = [orbital for orbital in range(n_qubits) if orbital not in occupied_set]
        
        index = 0
        
        for (i, j) in combinations(occupied_indices, 2):
            for (a, b) in combinations(unoccupied_indices, 2):
                
                self.append(DoubleExcitationGate(self.parameters_symbols[index:index + 1]), [i, j, a, b])
                index += 1 

class GlobalQCMPS(Component):
    def __init__(self, n_phys_qubits: int, n_bond_qubits: int, layer_types: list[LayerType]):
        self.n_phys_qubits = n_phys_qubits
        self.n_bond_qubits = n_bond_qubits
        
        self.layer_types = layer_types
        
        self.block_entries = n_bond_qubits + 1
            
        self.num_block_params = sum(layer_type.n_parameters(self.block_entries) for layer_type in layer_types)
        
        super().__init__(self.n_phys_qubits + self.n_bond_qubits, self.num_block_params * self.n_phys_qubits, "GlobalQCMPS")
        
        self.build_blocks()
        self.build_circuit()
            
    def build_blocks(self) -> None:
        
        self.blocks = []
        
        index = 0
        
        for _ in range(self.n_phys_qubits):
            
            block = QuantumCircuit(self.block_entries)
            
            for layer_type in self.layer_types:
                
                n_params = layer_type.n_parameters(self.block_entries)
                
                block.compose(
                    layer_type.build(self.block_entries, self.parameters_symbols[index:index + n_params]),
                    inplace=True,
                )
                index += n_params
            
            self.blocks.append(block)
            
    def build_circuit(self) -> None:
        self.clear()

        for orbital, block in enumerate(self.blocks):
            
            for qubit in range(self.n_bond_qubits, 0, -1):
                self.swap(orbital + qubit - 1, orbital + qubit)

            self.compose(block, qubits=list(range(orbital, orbital + self.block_entries)), inplace=True)

    def add_layers(self, layer_types: list[LayerType]) -> None:
        old_values = self.parameters_values.reshape(
            self.n_phys_qubits,
            self.num_block_params,
        )

        added_block_params = sum(
            layer.n_parameters(self.block_entries)
            for layer in layer_types
        )

        new_values = np.pad(
            old_values,
            ((0, 0), (0, added_block_params)),
        ).reshape(-1)

        self.layer_types.extend(layer_types)
        self.num_block_params += added_block_params

        self.resize_parameters(len(new_values))
        self.parameters_values = new_values

        self.build_blocks()
        self.build_circuit()