from .circuit_blocks import Block, HFPreSetBlock, LUBlock, FullUBlock, DoubleExcitationBlock

import numpy as np
from typing import Callable
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector


class GlobalQCMPS(QuantumCircuit):
    
#--------------Building the ansatz----------------
    
    def __init__(self, n_orbs: int, n_bond_qubits: int, block_type: type[Block], layers: int = 1, hf_occ: list[int] | None = None, double_excitation: bool = False):
        
        if double_excitation and hf_occ is None:
            raise ValueError("HF occupation must be provided when double excitation is enabled.")
        
        self.n_phys_qubits = n_orbs
        self.n_bond_qubits = n_bond_qubits
        
        self.block_type = block_type
        
        self.layers = layers
        
        self.hf_occ = hf_occ
        self.double_excitation = double_excitation
        
        self.block_entries = n_bond_qubits + 1
        
        super().__init__(self.n_phys_qubits + self.n_bond_qubits)
        
        self.n_block_params = (
            FullUBlock.n_parameters(self.block_entries) * (layers + 1)
            + block_type.n_parameters(self.block_entries) * layers
        )
        
        self.n_init_params = 0 if not double_excitation else DoubleExcitationBlock.n_parameters(self.num_qubits, 1, self.n_phys_qubits, self.hf_occ)
        self.n_qcmps_params = self.n_block_params * self.n_phys_qubits

        self.n_params = self.n_qcmps_params + self.n_init_params
        
        self.parameters_symbols = ParameterVector("theta", self.n_params)
        self.parameters_values = np.zeros(self.n_params)
        self.gates = []
        
        self.build_gates()
        self.build_circuit()
        
        self.hamiltonian = None
        self.nuclear_repulsion = None
        
    def build_gates(self):
        
        index = 0
            
        if self.hf_occ is not None:
            self.gates.append(HFPreSetBlock(self.num_qubits, self.block_entries, self.hf_occ))
            
        if self.double_excitation:
            n_de_params = DoubleExcitationBlock.n_parameters(self.num_qubits, 1, self.n_phys_qubits, self.hf_occ)
            self.gates.append(DoubleExcitationBlock(self.parameters_symbols[index:index + n_de_params], self.num_qubits, self.n_phys_qubits, self.block_entries, self.hf_occ))
            index += n_de_params
        
        for _ in range(self.n_phys_qubits):
            block = QuantumCircuit(self.block_entries)

            n_u_params = FullUBlock.n_parameters(self.block_entries)
            block.compose(
                FullUBlock(self.parameters_symbols[index:index + n_u_params], self.block_entries),
                inplace=True,
            )
            index += n_u_params

            n_lu_params = self.block_type.n_parameters(self.block_entries)
            for _ in range(self.layers):
                block.compose(
                    self.block_type(
                        parameters=self.parameters_symbols[index:index + n_lu_params],
                        n_qubits=self.block_entries,
                    ),
                    inplace=True,
                )
                index += n_lu_params

                block.compose(
                    FullUBlock(self.parameters_symbols[index:index + n_u_params], self.block_entries),
                    inplace=True,
                )
                index += n_u_params

            self.gates.append(block)

    def add_hamiltonian(self, hamiltonian: SparsePauliOp):
        if hamiltonian.num_qubits != self.n_phys_qubits:
            raise ValueError(
                f"Expected a Hamiltonian on {self.n_phys_qubits} orbital qubits, "
                f"got {hamiltonian.num_qubits}."
            )
        
        self.hamiltonian = hamiltonian
    
    def add_nuclear_repulsion(self, nuclear_repulsion: float):
        self.nuclear_repulsion = nuclear_repulsion
    
    def build_circuit(self) -> None:
        self.clear()
        gate_offset = 0

        if self.hf_occ is not None:
            self.compose(self.gates[0], inplace=True)
            gate_offset = 1
            
        if self.double_excitation:
            self.compose(self.gates[1], inplace=True)
            gate_offset = 2

        for orbital, gate in enumerate(self.gates[gate_offset:]):
            for qubit in range(self.n_bond_qubits, 0, -1):
                self.swap(orbital + qubit - 1, orbital + qubit)

            self.compose(
                gate,
                qubits=list(range(orbital, orbital + self.block_entries)),
                inplace=True,
            )

#-----------------Energy and optimization functions----------------

    def energy(self) -> float:
        
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian has not been added to the ansatz.")

        circuit = self.assign_parameters(dict(zip(self.parameters_symbols, self.parameters_values, strict=True)))
        state = Statevector.from_instruction(circuit)
        energy = state.expectation_value(self.hamiltonian, qargs=list(range(self.n_phys_qubits)))
        return float(np.real_if_close(energy))
    
    def total_energy(self) -> float:
        
        if self.nuclear_repulsion is None:
            raise ValueError("Nuclear repulsion energy has not been added to the ansatz.")
        
        return self.energy() + self.nuclear_repulsion
    
    def opt_function(self) -> Callable[[np.ndarray], float]:
        
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian has not been added to the ansatz.")
        
        def opt_function(x: np.ndarray) -> float:
            self.parameters_values = x
            return self.energy()
        
        return opt_function
    
    def init_opt_function(self) -> Callable[[np.ndarray], float]:
        
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian has not been added to the ansatz.")
        
        if not self.double_excitation:
            raise ValueError("Double excitation must be enabled to use the init_opt_function.")
        
        def opt_function(x: np.ndarray) -> float:
            self.parameters_values[:self.n_init_params] = x
            return self.energy()
        
        return opt_function
    
    def qcmps_opt_function(self) -> Callable[[np.ndarray], float]:
        
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian has not been added to the ansatz.")
        
        def opt_function(x: np.ndarray) -> float:
            self.parameters_values[self.n_init_params:] = x
            return self.energy()
        
        return opt_function

#-----------------Parameter management----------------

    def current_parameters(self) -> np.ndarray:
        return self.parameters_values.copy()
    
    def current_qcmps_parameters(self) -> np.ndarray:
        return self.parameters_values[self.n_init_params:].copy()
    
    def current_init_parameters(self) -> np.ndarray:
        return self.parameters_values[:self.n_init_params].copy()
    
    def update_parameters(self, new_parameters: np.ndarray) -> None:
        if len(new_parameters) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(new_parameters)}.")
        
        self.parameters_values = new_parameters.copy()

    def update_qcmps_parameters(self, new_parameters: np.ndarray) -> None:
        if len(new_parameters) != self.n_qcmps_params:
            raise ValueError(f"Expected {self.n_qcmps_params} parameters, got {len(new_parameters)}.")
        
        self.parameters_values[self.n_init_params:] = new_parameters.copy()
        
    def update_init_parameters(self, new_parameters: np.ndarray) -> None:
        if len(new_parameters) != self.n_init_params:
            raise ValueError(f"Expected {self.n_init_params} parameters, got {len(new_parameters)}.")
        
        self.parameters_values[:self.n_init_params] = new_parameters.copy()

    def gaussian_qcmps_parameters(self, mean: float = 0.0, stddev: float = 0.1) -> None:
        
        self.parameters_values[self.n_init_params:] = np.random.normal(mean, stddev, self.n_qcmps_params)

#-----------------Printing functions----------------

    def print_energy(self):
        
        energy = self.energy()
        print(f"Energy: {energy:.12f}")
        
    def print_total_energy(self, *args, **kwargs):
        
        total_energy = self.total_energy()
        print(f"Total Energy: {total_energy:.12f}")