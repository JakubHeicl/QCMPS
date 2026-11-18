from __future__ import annotations
import json
from .layers import LayerType
from .gates import DoubleExcitationGate
from .utils import todo

import numpy as np
from typing import Callable
from qiskit import QuantumCircuit   
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from pathlib import Path

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
    
    def statevector(self) -> Statevector:
        bound_circuit = self.bound_circuit()
        return Statevector.from_instruction(bound_circuit)
    
    def objective(self, function: Callable[[Ansatz], float], components: Component | list[Component], parameter_groups: np.ndarray | list[np.ndarray] | None = None) -> Callable[[np.ndarray], float]:
        
        if isinstance(components, Component):
            components = [components]
        
        if any(not any(item is component for item in self.components) for component in components):
            raise ValueError("The specified component is not part of the ansatz.")
        
        if parameter_groups is None:
            parameter_groups = [np.arange(component.n_parameters) for component in components]
        elif isinstance(parameter_groups, np.ndarray):
            parameter_groups = [parameter_groups]
        
        if len(parameter_groups) != len(components):
            raise ValueError("The number of parameter groups must match the number of components.")
    
        normalized_groups = []
        baselines = []
        
        for component, group in zip(components, parameter_groups, strict = True):
            
            group = np.asarray(group, dtype=int)
            
            if group.ndim != 1:
                raise ValueError("Each parameter group must be a 1D array.")
            
            if np.any(group < 0) or np.any(group >= component.n_parameters):
                raise ValueError("Parameter group indices are out of bounds for the component.")
            
            if len(np.unique(group)) != len(group):
                raise ValueError("Parameter group indices must be unique for each component.")
            
            normalized_groups.append(group.copy())
            baselines.append(component.current_parameters())
        
        expected_parameters = sum(len(group) for group in normalized_groups)
        
        def objective_function(x: np.ndarray) -> float:
            
            x = np.asarray(x, dtype=float)
            
            if len(x) != expected_parameters:
                raise ValueError(f"Expected {expected_parameters} parameters, got {len(x)}.")
        
            offset = 0
            
            for component, group, baseline in zip(components, normalized_groups, baselines, strict = True):
                
                n_group_parameters = len(group)
                
                local_parameters = x[offset:offset + n_group_parameters]
                
                full_parameters = baseline.copy()
                full_parameters[group] = local_parameters
                
                component.update_parameters(full_parameters)
                offset += n_group_parameters   
            
            return function(self)
        
        return objective_function
    
    @todo("Need to implement calculation for multiple components at once. Maybe a general function for general objective functions.")
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
    
    def save_checkpoint(self, path: Path | str):
        path = Path(path)
        
        checkpoint_data = {
            "format_version": 1,
            "n_qubits": self.n_qubits,
            "components": [],
            "component_qubits": self.component_qubits,
        }
        
        for component in self.components:
            component_data = {
                "name": component.name,
                "type": type(component).__qualname__,
                "n_qubits": component.n_qubits,
                "n_parameters": component.n_parameters,
                "parameters_values": component.parameters_values.tolist() if component.parameters_values is not None else None,
            }
            checkpoint_data["components"].append(component_data)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=4)
            
    def load_checkpoint(self, path: Path | str):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
            
        n_qubits = checkpoint_data["n_qubits"]
        format_version = checkpoint_data["format_version"]
        component_qubits = checkpoint_data["component_qubits"]
        components_data = checkpoint_data["components"]
        
        if format_version != 1:
            raise ValueError(f"Unsupported checkpoint format version: {format_version}")

        if n_qubits != self.n_qubits:
            raise ValueError(f"Checkpoint n_qubits ({n_qubits}) does not match ansatz n_qubits ({self.n_qubits}).")
        
        if len(components_data) != len(component_qubits):
            raise ValueError("Mismatch between number of components and component qubits in checkpoint.")
        
        if len(components_data) != len(self.components):
            raise ValueError("Mismatch between number of components in checkpoint and ansatz.")
        
        if len(component_qubits) != len(self.component_qubits):
            raise ValueError("Checkpoint component placement count does not match ansatz component placement count.")
        
        validated_parameters = []
        
        for (component_data, component_qubit_list, component, current_qubit_list) in zip(components_data, component_qubits, self.components, self.component_qubits, strict = True):
            
            name = component_data["name"]
            type_name = component_data["type"]
            component_n_qubits = component_data["n_qubits"]
            n_parameters = component_data["n_parameters"]
            parameters_values = np.array(component_data["parameters_values"], dtype=float) if component_data["parameters_values"] is not None else None
            
            if name != component.name:
                raise ValueError(f"Checkpoint component name ({name}) does not match ansatz component name ({component.name}).")
            
            if type_name != type(component).__qualname__:
                raise ValueError(f"Checkpoint component type ({type_name}) does not match ansatz component type ({type(component).__qualname__}).")
            
            if component_n_qubits != component.n_qubits:
                raise ValueError(f"Checkpoint component n_qubits ({component_n_qubits}) does not match ansatz component n_qubits ({component.n_qubits}).")
            
            if n_parameters != component.n_parameters:
                raise ValueError(f"Checkpoint component n_parameters ({n_parameters}) does not match ansatz component n_parameters ({component.n_parameters}).")
            
            if component_qubit_list != current_qubit_list:
                raise ValueError(f"Checkpoint component qubit placement ({component_qubit_list}) does not match ansatz component qubit placement ({current_qubit_list}).")

            if n_parameters == 0:
                if parameters_values is not None:
                    raise ValueError(
                        "A parameterless component must not contain parameter values."
                    )
            else:
                if parameters_values is None:
                    raise ValueError(
                        f"Checkpoint component {name} is missing parameter values."
                    )

                if parameters_values.shape != (n_parameters,):
                    raise ValueError(
                        f"Checkpoint component {name} expected "
                        f"{n_parameters} parameter values, got shape "
                        f"{parameters_values.shape}."
                    )

                if not np.all(np.isfinite(parameters_values)):
                    raise ValueError(
                        f"Checkpoint component {name} contains non-finite parameters."
                    )
            
            validated_parameters.append(parameters_values)
        
        for component, parameters in zip(self.components, validated_parameters, strict = True):
        
            if component.n_parameters > 0:
                component.update_parameters(parameters)
        
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
                
                block.compose(layer_type.build(self.block_entries, self.parameters_symbols[index:index + n_params]), inplace=True)
                index += n_params
            
            self.blocks.append(block)
            
    def build_circuit(self) -> None:
        self.clear()

        for orbital, block in enumerate(self.blocks):
            
            for qubit in range(self.n_bond_qubits, 0, -1):
                self.swap(orbital + qubit - 1, orbital + qubit)

            self.compose(block, qubits=list(range(orbital, orbital + self.block_entries)), inplace=True)

    def add_layers(self, layer_types: list[LayerType], initializer: Callable[[int], np.ndarray] | None = None) -> None:
        
        self.insert_layers(len(self.layer_types), layer_types, initializer)
        
    def insert_layers(self, index: int, layer_types: list[LayerType], initializer: Callable[[int], np.ndarray] | None = None) -> None:
        
        if index < 0 or index > len(self.layer_types):
            raise ValueError(f"Index {index} is out of bounds for inserting layers into {len(self.layer_types)} existing layers.")
        
        if not layer_types:
            return
        
        old_block_params = self.num_block_params
        old_values = self.parameters_values.reshape(self.n_phys_qubits, old_block_params)
        
        inserted_block_params = sum(layer.n_parameters(self.block_entries) for layer in layer_types)
        n_inserted_parameters = self.n_phys_qubits * inserted_block_params
        
        if initializer is None:
            inserted_values = np.zeros(n_inserted_parameters)
        else:
            inserted_values = np.asarray(initializer(n_inserted_parameters), dtype=float)
            if inserted_values.shape != (n_inserted_parameters,):
                raise ValueError(f"Initializer must return {n_inserted_parameters} parameters, got shape {inserted_values.shape}.")
        
        inserted_values = inserted_values.reshape(self.n_phys_qubits, inserted_block_params)
        
        insertion_offset = sum(layer.n_parameters(self.block_entries) for layer in self.layer_types[:index])
        
        new_values = np.concatenate((old_values[:, :insertion_offset], inserted_values, old_values[:, insertion_offset:]), axis=1).reshape(-1)
        
        self.layer_types = self.layer_types[:index] + layer_types + self.layer_types[index:]
        self.num_block_params = old_block_params + inserted_block_params
        
        self.resize_parameters(len(new_values))
        self.update_parameters(new_values)
        
        self.build_blocks()
        self.build_circuit()
        
    def block_parameter_groups(self) -> list[np.ndarray]:
        
        offset = 0
        parameter_groups = []
        
        for _ in range(self.n_phys_qubits):
            
            parameter_groups.append(np.arange(offset, offset + self.num_block_params))
            offset += self.num_block_params
            
        return parameter_groups
    
    def layer_parameter_groups(self, layer_index: int) -> list[np.ndarray]:
        
        n_layers = len(self.layer_types)
        
        if layer_index < 0:
            layer_index += n_layers
            
        if layer_index < 0 or layer_index >= n_layers:
            raise ValueError(f"Layer index {layer_index} is out of bounds for {n_layers} layers.")
        
        layer_sizes = [layer.n_parameters(self.block_entries) for layer in self.layer_types]
        
        layer_start = sum(layer_sizes[:layer_index])
        layer_stop = layer_start + layer_sizes[layer_index]
        
        return [block * self.num_block_params + np.arange(layer_start, layer_stop) for block in range(self.n_phys_qubits)]
