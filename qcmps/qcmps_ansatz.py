from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector
import numpy as np

from .circuit_blocks import *

def prepare_blocks(
    n_orbs: int,
    n_bond_qubits: int,
    block_type: type[Block],
    layers: int = 1,
    hf_occ: list[int] = None,
    mps_unitaries: list[np.ndarray] | None = None,
    n_mps_bond_qubits: int = 0,
) -> tuple[list[QuantumCircuit], ParameterVector]:

    gates = []

    n_entries = n_bond_qubits + n_mps_bond_qubits + 1
    n_params = block_type.n_parameters(n_entries, layers) * n_orbs

    params = ParameterVector("theta", n_params)

    occupied = set(hf_occ or [])

    if mps_unitaries is not None and len(mps_unitaries) != n_orbs:
        raise ValueError(f"Expected {n_orbs} MPS unitaries, got {len(mps_unitaries)}.")

    index = 0

    for q in range(n_orbs):
        print(f"[qcmps] Adding {block_type.__name__} block {q + 1}/{n_orbs} on qubits {q}..{q + n_entries - 1}")

        block = QuantumCircuit(n_entries)

        if mps_unitaries is not None:
            block.unitary(mps_unitaries[q], list(range(n_mps_bond_qubits + 1)), label="MPS")
        elif q in occupied:
            block.compose(HFPreSet(n_entries), inplace=True)

        block.compose(
            block_type(n_entries, params[index : index + block_type.n_parameters(n_entries, layers)], layers),
            inplace=True,
        )

        index += block_type.n_parameters(n_entries, layers)

        gates.append(block)

    return gates, params

def prepare_global_ansatz(gates: list[QuantumCircuit]) -> QuantumCircuit:

    if len(gates) == 0:
        raise ValueError("`gates` must contain at least one block.")

    n_entries = gates[0].num_qubits
    n_sites = len(gates)

    circuit = QuantumCircuit(n_sites + n_entries - 1)

    for site, gate in enumerate(gates):
        if gate.num_qubits != n_entries:
            raise ValueError(
                f"All blocks must have {n_entries} qubits; block {site} has {gate.num_qubits}."
            )

        shifted_gate = QuantumCircuit(n_entries)
        for q in range(n_entries - 1, 0, -1):
            shifted_gate.swap(q - 1, q)
        shifted_gate.compose(gate, inplace=True)

        circuit.compose(shifted_gate, qubits=list(range(site, site + n_entries)), inplace=True)

    return circuit

def _embed_physical_operator(op: SparsePauliOp, n_qubits: int) -> SparsePauliOp:

    if op.num_qubits > n_qubits:
        raise ValueError(
            f"Operator acts on {op.num_qubits} qubits, but circuit has only {n_qubits}."
        )

    n_trailing_bond_qubits = n_qubits - op.num_qubits
    labels = [
        ("I" * n_trailing_bond_qubits + label, coeff)
        for label, coeff in op.to_list()
    ]

    return SparsePauliOp.from_list(labels)

def evaluate_hamiltonian_global(qcmps: list[QuantumCircuit] | QuantumCircuit, op: SparsePauliOp, verbose: bool = True) -> float:

    ansatz = qcmps if isinstance(qcmps, QuantumCircuit) else prepare_global_ansatz(qcmps)

    if verbose:
        print(
            "[qcmps] Evaluating global ansatz: "
            f"{op.num_qubits} physical qubits, {ansatz.num_qubits} total qubits"
        )

    state = Statevector.from_instruction(ansatz)
    energy = state.expectation_value(_embed_physical_operator(op, ansatz.num_qubits))

    return float(np.real_if_close(energy))

def _apply_controlled_pauli(circ: QuantumCircuit, pauli_char: str, anc: int, target: int):

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
        raise ValueError(f"Pauli string too short: got {len(paulis)} chars for {n_sites} sites.")

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

def evaluate_hamiltonian(gates: list[QuantumCircuit], op: SparsePauliOp, verbose: bool = True) -> float:

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
