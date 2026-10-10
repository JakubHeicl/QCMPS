from .circuit_blocks import Block, HFPreSetBlock, LUBlock, FullUBlock, DoubleExcitationBlock

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector


class GlobalAnsatz(QuantumCircuit):
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
        
        self.n_params = self.n_block_params * self.n_phys_qubits + self.n_init_params
        
        self.params = ParameterVector("theta", self.n_params)
        self.gates = []
        
        self.build_gates()
        self.build_circuit()
        
    def build_gates(self):
        
        index = 0
            
        if self.hf_occ is not None:
            self.gates.append(HFPreSetBlock(self.num_qubits, self.block_entries, self.hf_occ))
            
        if self.double_excitation:
            n_de_params = DoubleExcitationBlock.n_parameters(self.num_qubits, 1, self.n_phys_qubits, self.hf_occ)
            self.gates.append(DoubleExcitationBlock(self.params[index:index + n_de_params], self.num_qubits, self.n_phys_qubits, self.block_entries, self.hf_occ))
            index += n_de_params
        
        for _ in range(self.n_phys_qubits):
            block = QuantumCircuit(self.block_entries)

            n_u_params = FullUBlock.n_parameters(self.block_entries)
            block.compose(
                FullUBlock(self.params[index:index + n_u_params], self.block_entries),
                inplace=True,
            )
            index += n_u_params

            n_lu_params = self.block_type.n_parameters(self.block_entries)
            for _ in range(self.layers):
                block.compose(
                    self.block_type(
                        parameters=self.params[index:index + n_lu_params],
                        n_qubits=self.block_entries,
                    ),
                    inplace=True,
                )
                index += n_lu_params

                block.compose(
                    FullUBlock(self.params[index:index + n_u_params], self.block_entries),
                    inplace=True,
                )
                index += n_u_params

            self.gates.append(block)

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

    def energy(self, parameters: list[float], hamiltonian: SparsePauliOp) -> float:

        values = np.asarray(parameters, dtype=float)
        
        if values.shape != (len(self.params),):
            raise ValueError(f"Expected {len(self.params)} parameter values, got shape {values.shape}.")
        
        if hamiltonian.num_qubits != self.n_phys_qubits:
            raise ValueError(
                f"Expected a Hamiltonian on {self.n_phys_qubits} orbital qubits, "
                f"got {hamiltonian.num_qubits}."
            )

        circuit = self.assign_parameters(dict(zip(self.params, values, strict=True)))
        state = Statevector.from_instruction(circuit)
        energy = state.expectation_value(hamiltonian, qargs=list(range(self.n_phys_qubits)))
        return float(np.real_if_close(energy))