from typing import Callable

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.formats import fcidump_to_problem
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.mappers import JordanWignerMapper

from pathlib import Path
import numpy as np
from scipy.optimize import minimize, differential_evolution

from .qcmps_ansatz import evaluate_hamiltonian

def hf_occupancy(num_alpha: int, num_beta: int, n_spatial_orbitals: int) -> list[int]:

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

def build_from_fcidump(file_path: Path) -> tuple[SparsePauliOp, float, int, int, int, int, int]:

    print(f"[qcmps] Loading FCIDUMP from {file_path}")
    fcidump = FCIDump.from_file(file_path)

    print("[qcmps] Converting FCIDUMP to electronic structure problem")
    problem = fcidump_to_problem(fcidump)

    print("[qcmps] Building second-quantized Hamiltonian")
    hamiltonian = problem.second_q_ops()[0]

    print("[qcmps] Mapping fermionic Hamiltonian to qubits with Jordan-Wigner")
    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(hamiltonian).simplify()

    nuclear_repulsion = problem.nuclear_repulsion_energy

    n_spin_orbitals = problem.num_spin_orbitals
    n_spatial_orbitals = n_spin_orbitals // 2

    num_alpha = int(problem.num_alpha)
    num_beta = int(problem.num_beta)
    num_particles_total = num_alpha + num_beta

    print(
        "[qcmps] Loaded system: "
        f"{n_spatial_orbitals} spatial orbitals, "
        f"{n_spin_orbitals} spin orbitals, "
        f"{num_particles_total} electrons "
        f"({num_alpha} alpha, {num_beta} beta)"
    )
    print(f"[qcmps] Qubit Hamiltonian uses {hamiltonian.num_qubits} qubits and {len(hamiltonian)} Pauli terms")

    return hamiltonian, nuclear_repulsion, n_spin_orbitals, n_spatial_orbitals, num_alpha, num_beta, num_particles_total

def build_energy_objective_mps(gates: list[QuantumCircuit], params: ParameterVector, operator: SparsePauliOp, callback: Callable[[int, float, np.ndarray], None] | None = None, verbose: bool = False) -> Callable[[np.ndarray], float]:

    counter = {"n": 0}

    param_list = list(params)
    n_expected = len(param_list)

    def objective(x: np.ndarray) -> float:
        counter["n"] += 1
        x = np.asarray(x, dtype=float)

        if x.shape != (n_expected,):
            raise ValueError(
                f"Expected parameter vector of shape ({n_expected},), got {x.shape}."
            )

        bind_map = dict(zip(param_list, x, strict=True))

        bound_gates = [
            gate.assign_parameters(
                {p: bind_map[p] for p in gate.parameters},
                inplace=False,
            )
            for gate in gates
        ]

        energy = evaluate_hamiltonian(bound_gates, operator, verbose=False)
        energy = float(np.real_if_close(energy))

        if callback is not None:
            callback(counter["n"], energy, x.copy())

        if verbose:
            print(f"[qcmps] Evaluation {counter['n']}: E = {energy:.12f}")

        return energy

    return objective

def global_then_local(objective, n_params: int, optimizer: str = "COBYLA", maxiter: int = 1000):
    bounds = [(-np.pi, np.pi)] * n_params

    print("[qcmps] Starting differential evolution")

    res_global = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=10,
        popsize=10,
        tol=1e-8,
        polish=False,
        updating="immediate",
        workers=1,
        seed=42,
    )

    print("[qcmps] Global result:")
    print(res_global.fun)

    print("[qcmps] Polishing with " + optimizer)

    res_local = minimize(
        objective,
        res_global.x,
        method=optimizer,
        options={
            "maxiter": maxiter,
            "gtol": 1e-8,
            "disp": True,
        },
    )

    return res_global, res_local