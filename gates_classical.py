from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.circuit import ParameterVector

import numpy as np

def get_n_parameters(n_orb_qubits: int, n_bond_qubits: int, layers: int = 1) -> int:

    return 3 * (n_bond_qubits + 1) * (n_bond_qubits) * layers * n_orb_qubits

def prepare_gates(n_orbs: int, n_bond_qubits: int, layers: int = 1) -> tuple[list[QuantumCircuit], ParameterVector]:

    gates = []

    n_params = get_n_parameters(n_orbs, n_bond_qubits, layers)

    n_entries = n_bond_qubits + 1

    params = ParameterVector("theta", n_params)

    index = 0

    for q in range(n_orbs):
        print(f"[qcmps] Adding AU block {q + 1}/{n_orbs} on qubits {q}..{q + n_entries - 1}")

        block = AUBlock(n_entries, params[index:index+3*n_entries*(n_entries-1)*layers], layers)
        index += 3*n_entries*(n_entries-1)*layers

        gates.append(block)

        #list(range(q, q+n_entries))

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

def _retained_basis_index(retained_index: int, physical_bit: int, retained_qubits: list[int]) -> int:

    full_index = physical_bit << 1

    for retained_pos, qubit in enumerate(retained_qubits):
        if (retained_index >> retained_pos) & 1:
            full_index |= 1 << qubit

    return full_index

def _kraus_from_unitary(unitary: np.ndarray, n_entries: int) -> tuple[np.ndarray, np.ndarray]:

    full_qubits = n_entries + 1
    retained_qubits = [0] + list(range(2, full_qubits))
    retained_dim = 1 << len(retained_qubits)

    input_indices = [
        _retained_basis_index(i, physical_bit=0, retained_qubits=retained_qubits)
        for i in range(retained_dim)
    ]

    return tuple(unitary[np.ix_([
                    _retained_basis_index(i, physical_bit=physical_bit, retained_qubits=retained_qubits)
                    for i in range(retained_dim)
                ], input_indices)] for physical_bit in (0, 1))

def _build_transfer_data(gates: list[QuantumCircuit]) -> tuple[list[dict[str, tuple[np.ndarray, np.ndarray]]], np.ndarray, np.ndarray]:

    n_entries = gates[0].num_qubits
    full_qubits = n_entries + 1

    channels = []

    for gate in gates:
        site_channels = {}

        for pauli_char in "IXYZ":
            step = QuantumCircuit(full_qubits)
            step.compose(gate, qubits=list(range(1, n_entries + 1)), inplace=True)
            _apply_controlled_pauli(step, pauli_char, anc=0, target=1)
            site_channels[pauli_char] = _kraus_from_unitary(Operator(step).data, n_entries)

        channels.append(site_channels)

    retained_dim = 1 << n_entries

    initial_state = np.zeros(retained_dim, dtype=complex)
    initial_state[0] = 1 / np.sqrt(2)
    initial_state[1] = 1 / np.sqrt(2)
    initial_density = np.outer(initial_state, initial_state.conj())

    ancilla_x = np.zeros((retained_dim, retained_dim), dtype=complex)
    for i in range(retained_dim):
        ancilla_x[i ^ 1, i] = 1

    return channels, initial_density, ancilla_x

def _apply_transfer_channel(channel: tuple[np.ndarray, np.ndarray], density: np.ndarray) -> np.ndarray:

    k0, k1 = channel
    return k0 @ density @ k0.conj().T + k1 @ density @ k1.conj().T

def _evaluate_pauli_string_transfer(
    transfer_data: tuple[list[dict[str, tuple[np.ndarray, np.ndarray]]], np.ndarray, np.ndarray],
    label: str,
    verbose: bool = False,
) -> float:

    channels, initial_density, ancilla_x = transfer_data
    paulis = label[::-1]

    if len(paulis) < len(channels):
        raise ValueError(
            f"Pauli string too short: got {len(paulis)} chars for {len(channels)} sites."
        )

    density = initial_density.copy()

    for site, site_channels in enumerate(channels):
        pauli_char = paulis[site]
        density = _apply_transfer_channel(site_channels[pauli_char], density)

        if verbose:
            print(f"Site {site}: applying block U_{site}, controlled-{pauli_char}, reset(q0)")

    expectation = np.trace(ancilla_x @ density)
    return float(np.real_if_close(expectation))

def _build_pauli_prefix_tree(labels_and_coeffs: list[tuple[str, complex]], n_sites: int) -> dict:

    root = {None: 0j}

    for label, coeff in labels_and_coeffs:
        paulis = label[::-1]

        if len(paulis) < n_sites:
            raise ValueError(
                f"Pauli string too short: got {len(paulis)} chars for {n_sites} sites."
            )

        node = root
        for site in range(n_sites):
            node = node.setdefault(paulis[site], {None: 0j})

        node[None] += coeff

    return root

def _get_pauli_prefix_tree(op: SparsePauliOp, labels_and_coeffs: list[tuple[str, complex]], n_sites: int) -> dict:

    signature = (op.num_qubits, len(labels_and_coeffs), n_sites)
    cached = getattr(op, "_qcmps_pauli_prefix_tree", None)

    if cached is not None and cached[0] == signature:
        return cached[1]

    tree = _build_pauli_prefix_tree(labels_and_coeffs, n_sites)
    op._qcmps_pauli_prefix_tree = (signature, tree)
    return tree

def _evaluate_pauli_prefix_tree(
    node: dict,
    site: int,
    density: np.ndarray,
    channels: list[dict[str, tuple[np.ndarray, np.ndarray]]],
    ancilla_x: np.ndarray,
) -> complex:

    coeff = node[None]
    energy = coeff * np.trace(ancilla_x @ density) if coeff else 0j

    if site == len(channels):
        return energy

    for pauli_char, child in node.items():
        if pauli_char is None:
            continue

        next_density = _apply_transfer_channel(channels[site][pauli_char], density)
        energy += _evaluate_pauli_prefix_tree(child, site + 1, next_density, channels, ancilla_x)

    return energy

def evaluate_pauli_string_qcmps(gates: list[QuantumCircuit], label: str, verbose: bool = False) -> float:

    if len(gates) == 0:
        raise ValueError("`gates` must contain at least one block.")

    transfer_data = _build_transfer_data(gates)
    return _evaluate_pauli_string_transfer(transfer_data, label, verbose)

def evaluate_hamiltonian(gates: list[QuantumCircuit], op: SparsePauliOp,verbose: bool = True,) -> float:

    if len(gates) == 0:
        raise ValueError("`gates` must contain at least one block.")

    transfer_data = _build_transfer_data(gates)
    labels_and_coeffs = op.to_list()

    if not verbose:
        channels, initial_density, ancilla_x = transfer_data
        tree = _get_pauli_prefix_tree(op, labels_and_coeffs, len(gates))
        energy = _evaluate_pauli_prefix_tree(tree, 0, initial_density, channels, ancilla_x)
        return float(np.real_if_close(energy))

    energy = 0.0

    for label, coeff in labels_and_coeffs:
        if verbose:
            print(f"Evaluating term {label} with coefficient {coeff}")

        exp_val = _evaluate_pauli_string_transfer(transfer_data, label, verbose=False)
        contrib = coeff * exp_val
        energy += contrib

        if verbose:
            print(f"  <{label}> = {exp_val:.12f}")
            print(f"  contribution = {contrib}")

    return float(np.real_if_close(energy))
