from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info import DensityMatrix
from qiskit.circuit import ParameterVector

from typing import Callable

import numpy as np

def get_n_parameters(n_orb_qubits: int, n_bond_qubits: int, layers: int = 1) -> int:

    return 3 * (n_bond_qubits + 1) * (n_bond_qubits) * layers * n_orb_qubits

def prepare_gates(n_orbs: int, n_bond_qubits: int, layers: int = 1, hf_occ: list[int] | None = None) -> QuantumCircuit:

    gates = []

    n_params = get_n_parameters(n_orbs, n_bond_qubits, layers)

    n_entries = n_bond_qubits + 1

    params = ParameterVector("theta", n_params)

    for q in range(n_orbs):
        print(f"[qcmps] Adding AU block {q + 1}/{n_orbs} on qubits {q}..{q + n_entries - 1}")

        gate_circuit = AUBlock(n_entries, params[index:index+3*n_entries*(n_entries-1)*layers], layers)
        index += 3*n_entries*(n_entries-1)*layers

        gates.append(gate_circuit.to_gate(), list(range(q, q+n_entries)))

    return gates, params

class HFReference(QuantumCircuit):

    def __init__(self, n_qubits: int, occupied_orbitals: list[int]):
        super().__init__(n_qubits)

        print(f"[qcmps] Preparing Hartree-Fock reference on occupied qubits: {occupied_orbitals}")
        for q in occupied_orbitals:
            self.x(q)

class AUBlock(QuantumCircuit):

    def __init__(self, n_entries: int, parameters: list[float], layers = 1):
        super().__init__(n_entries)

        index = 0

        for l in range(layers):
            for i in range(n_entries):
                for j in range(n_entries):

                    if i == j:
                        continue

                    gate_circuit = u_gate(parameters[index:index+3])
                    index += 3

                    self.append(gate_circuit.to_gate(), [i, j])

def u_gate(parameters: list[float]) -> QuantumCircuit:
    
    qc = QuantumCircuit(2)

    a, b, c = parameters

    qc.cx(0, 1)
    qc.ry(a, 0)
    qc.rz(b, 0)
    qc.ry(c, 0)
    qc.cx(0, 1)

    return qc

def _apply_controlled_pauli(circ: QuantumCircuit, pauli_char: str, anc: int, target: int):
    """
    Controlled-P gate from ancilla to target physical qubit q0.
    """
    if pauli_char == "I":
        return
    elif pauli_char == "X":
        circ.cx(anc, target)
    elif pauli_char == "Y":
        circ.cy(anc, target)
    elif pauli_char == "Z":
        circ.cz(anc, target)
    else:
        raise ValueError(f"Unsupported Pauli character: {pauli_char}")

def evaluate_pauli_string_qcmps(gates: list[QuantumCircuit], label: str, verbose: bool = False) -> float:

    if len(gates) == 0:
        raise ValueError("`gates` must contain at least one block.")

    n_entries = gates[0].num_qubits
    n_sites = len(gates)

    paulis = label[::-1]

    if len(paulis) < n_sites:
        raise ValueError(
            f"Pauli string too short: got {len(paulis)} chars for {n_sites} sites."
        )

    dm = DensityMatrix.from_label("0" * (n_entries + 1))

    init = QuantumCircuit(n_entries + 1)
    init.h(0)
    dm = dm.evolve(init)

    for site, gate in enumerate(gates):
        p = paulis[site]

        step = QuantumCircuit(n_entries + 1)

        step.compose(gate, qubits=list(range(1, n_entries + 1)), inplace=True)

        _apply_controlled_pauli(step, p, anc=0, target=1)

        step.reset(1)

        if verbose:
            print(f"Site {site}: applying block U_{site}, controlled-{p}, reset(q0)")

        dm = dm.evolve(step)

    final = QuantumCircuit(n_entries + 1)
    final.h(0)
    dm = dm.evolve(final)

    probs = dm.probabilities([0])
    p0 = probs[0]
    p1 = probs[1]

    expectation = p0 - p1
    return float(np.real_if_close(expectation))

def evaluate_hamiltonian(gates: list[QuantumCircuit], op: SparsePauliOp,verbose: bool = True,) -> float:

    energy = 0.0

    for label, coeff in op.to_list():
        if verbose:
            print(f"Evaluating term {label} with coefficient {coeff}")

        exp_val = evaluate_pauli_string_qcmps(gates, label, verbose=False)
        contrib = coeff * exp_val
        energy += contrib

        if verbose:
            print(f"  <{label}> = {exp_val:.12f}")
            print(f"  contribution = {contrib}")

    return float(np.real_if_close(energy))