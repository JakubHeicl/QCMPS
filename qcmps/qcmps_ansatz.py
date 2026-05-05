from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import DensityMatrix, SparsePauliOp
import numpy as np

from .circuit_blocks import *

def prepare_blocks(n_orbs: int, 
                   n_bond_qubits: int,
                   block_type: type[Block], 
                   layers: int = 1, hf_occ: list[int] = None) -> tuple[list[QuantumCircuit], ParameterVector]:

    gates = []

    n_entries = n_bond_qubits + 1
    n_params = block_type.n_parameters(n_entries, layers) * n_orbs

    params = ParameterVector("theta", n_params)

    occupied = set(hf_occ or [])

    index = 0

    for q in range(n_orbs):
        print(f"[qcmps] Adding {block_type.__name__} block {q + 1}/{n_orbs} on qubits {q}..{q + n_entries - 1}")

        block = QuantumCircuit(n_entries)

        if q in occupied:
            block.compose(HFPreSet(n_entries), inplace=True)

        block.compose(
            block_type(n_entries, params[index : index + block_type.n_parameters(n_entries, layers)], layers),
            inplace=True,
        )

        index += block_type.n_parameters(n_entries, layers)

        gates.append(block)

    return gates, params

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