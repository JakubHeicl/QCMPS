from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

from typing import Callable

import numpy as np

def get_n_parameters(n_orb_qubits: int, n_bond_qubits: int, layers: int = 1) -> int:

    return 3 * (n_bond_qubits + 1) * (n_bond_qubits) * layers * n_orb_qubits

def build_ansatz(n_orb_qubits: int, n_bond_qubits: int, layers: int = 1, hf_occ: list[int] | None = None) -> QuantumCircuit:

    n_params = get_n_parameters(n_orb_qubits, n_bond_qubits, layers)

    n_entries = n_bond_qubits + 1

    print(
        "[qcmps] Building ansatz: "
        f"{n_orb_qubits} orbital qubits, "
        f"{n_bond_qubits} bond qubits, "
        f"{layers} layer(s), "
        f"{n_params} parameters"
    )

    params = ParameterVector("theta", n_params)

    ansatz = QuantumCircuit(n_orb_qubits + n_bond_qubits)

    index = 0

    if hf_occ is not None:
        print(f"[qcmps] Preparing Hartree-Fock reference on occupied qubits: {hf_occ}")
        for q in hf_occ:
            ansatz.x(q)

    for q in range(n_orb_qubits):
        print(f"[qcmps] Adding AU block {q + 1}/{n_orb_qubits} on qubits {q}..{q + n_entries - 1}")

        gate_circuit = AU_block(n_entries, params[index:index+3*n_entries*(n_entries-1)*layers], layers)
        index += 3*n_entries*(n_entries-1)*layers

        ansatz.append(gate_circuit.to_gate(), list(range(q, q+n_entries)))

    print(f"[qcmps] Ansatz construction finished with {ansatz.num_qubits} qubits")
    return ansatz, params

def exp_from_ansatz(ansatz: QuantumCircuit, op: SparsePauliOp) -> float:

    print(f"[qcmps] Simulating statevector for {ansatz.num_qubits} qubits")
    state = Statevector.from_instruction(ansatz)

    print("[qcmps] Computing expectation value")
    return float(np.real_if_close(state.expectation_value(op)))

def identity_op(num_qubits: int) -> SparsePauliOp:
    return SparsePauliOp.from_list([("I" * num_qubits, 1.0)])

def extend_hamiltonian(op: SparsePauliOp, n_aux_qubits: int) -> SparsePauliOp:

    print(f"[qcmps] Extending Hamiltonian from {op.num_qubits} to {op.num_qubits + n_aux_qubits} qubits")
    extended = op.tensor(identity_op(n_aux_qubits)).simplify()
    print(f"[qcmps] Extended Hamiltonian has {extended.num_qubits} qubits and {len(extended)} Pauli terms")
    return extended

def build_energy_objective_statevector(
    ansatz: QuantumCircuit,
    params: ParameterVector,
    operator: SparsePauliOp,
    callback: Callable[[int, float, np.ndarray], None] | None = None,
) -> Callable[[np.ndarray], float]:
    counter = {"n": 0}

    def objective(x: np.ndarray) -> float:
        counter["n"] += 1
        print(f"[qcmps] Evaluation {counter['n']}: assigning parameters")
        bound = ansatz.assign_parameters(dict(zip(params, x)), inplace=False)
        print(f"[qcmps] Evaluation {counter['n']}: evaluating energy")
        energy = exp_from_ansatz(bound, operator)
        if callback is not None:
            callback(counter["n"], energy, x)
        return energy

    return objective

def build_energy_objective_estimator(
    ansatz: QuantumCircuit,
    params: ParameterVector,
    operator: SparsePauliOp,
    estimator: Estimator,
    callback: Callable[[int, float, np.ndarray], None] | None = None,
) -> Callable[[np.ndarray], float]:
    counter = {"n": 0}

    def objective(x: np.ndarray) -> float:
        counter["n"] += 1
        print(f"[qcmps] Evaluation {counter['n']}: evaluating energy with MPS estimator")

        result = estimator.run(
            ansatz,
            operator,
            parameter_values=x,
        ).result()

        energy = float(np.real_if_close(result.values[0]))

        if callback is not None:
            callback(counter["n"], energy, x)

        return energy

    return objective

def occ_hf(num_alpha: int, num_beta: int, n_spatial_orbitals: int) -> list[int]:

    alpha_occ = [1] * num_alpha + [0] * (n_spatial_orbitals - num_alpha)
    beta_occ = [1] * num_beta + [0] * (n_spatial_orbitals - num_beta)

    occupied = []
    for i, occ in enumerate(alpha_occ):
        if int(round(occ)) == 1:
            occupied.append(i)
    for i, occ in enumerate(beta_occ):
        if int(round(occ)) == 1:
            occupied.append(n_spatial_orbitals + i)
    return occupied

def run(fcidump_path: Path, layers: int = 1, n_bond_qubits: int = N_BOND_QUBITS, optimizer: str = "COBYLA", maxiter: int = 1000):
    print("[qcmps] Starting QCMPS run")
    print(
        "[qcmps] Settings: "
        f"fcidump={fcidump_path}, "
        f"layers={layers}, "
        f"bond_qubits={n_bond_qubits}, "
        f"optimizer={optimizer}, "
        f"maxiter={maxiter}"
    )

    hamiltonian, nuclear_repulsion, n_spin_orbitals, n_spatial_orbitals, num_alpha, num_beta, num_particles_total = build_from_fcidump(fcidump_path)

    n_orb_qubits = n_spin_orbitals

    print("[qcmps] Building Hartree-Fock occupation list")
    hf_occ = occ_hf(num_alpha, num_beta, n_spatial_orbitals)

    ansatz, params = build_ansatz(n_orb_qubits, n_bond_qubits, layers, hf_occ)

    hamiltonian = extend_hamiltonian(hamiltonian, n_bond_qubits)

    print("[qcmps] Creating random initial parameter vector")
    x0 = np.random.normal(0, 1e-2, size=len(params))
    print(f"[qcmps] Initial vector has length {len(x0)}")

    def cb(it: int, energy: float, x: np.ndarray) -> None:
        print(f"eval={it:4d}  E={energy + nuclear_repulsion: .12f}")

    print("[qcmps] Creating MPS estimator")
    estimator = build_mps_estimator(
        max_bond_dimension=128,
        truncation_threshold=1e-10,
    )

    #objective = build_energy_objective_estimator(ansatz, params, hamiltonian, estimator, callback=cb)

    objective = build_energy_objective_statevector(ansatz, params, hamiltonian, callback=cb)

    print("[qcmps] Starting classical optimization")
    result = minimize(
        objective,
        x0,
        method=optimizer,
        options={"maxiter": maxiter},
    )

    print("[qcmps] Optimization finished")
    return result