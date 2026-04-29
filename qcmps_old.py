from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

from qiskit_nature.second_q.formats import fcidump_to_problem
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.mappers import JordanWignerMapper

import numpy as np
from pathlib import Path
from typing import Callable
from scipy.optimize import minimize, differential_evolution

import argparse

from gates_classical import add_particle_number_penalty

N_BOND_QUBITS = 1

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

def build_energy_objective_with_penalty(
    gates,
    params,
    physical_hamiltonian: SparsePauliOp,
    penalty_hamiltonian: SparsePauliOp,
    nuclear_repulsion: float = 0.0,
    callback=None,
    verbose: bool = True,
    use_quantum_gates: bool = False,
):
    if use_quantum_gates:
        from gates_quantum import evaluate_hamiltonian
    else:
        from gates_classical import evaluate_hamiltonian

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

        e_phys = evaluate_hamiltonian(bound_gates, physical_hamiltonian, verbose=False)
        e_eff = evaluate_hamiltonian(bound_gates, penalty_hamiltonian, verbose=False)

        e_phys = float(np.real_if_close(e_phys))
        e_eff = float(np.real_if_close(e_eff))

        if callback is not None:
            callback(counter["n"], e_phys, e_eff, x.copy())

        if verbose:
            print(
                f"eval={counter['n']:4d} "
                f"Ephys={e_phys + nuclear_repulsion: .12f} "
                f"Eeff={e_eff + nuclear_repulsion: .12f}"
            )

        return e_eff

    return objective

def build_energy_objective_mps(gates: list[QuantumCircuit], params: ParameterVector, operator: SparsePauliOp, callback: Callable[[int, float, np.ndarray], None] | None = None, verbose: bool = True, use_quantum_gates: bool = True) -> Callable[[np.ndarray], float]:

    if use_quantum_gates:
        from gates_quantum import evaluate_hamiltonian
    else:
        from gates_classical import evaluate_hamiltonian

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
        if verbose:
            print(f"[qcmps] Evaluation {counter['n']}: assigning parameters")

        bind_map = dict(zip(param_list, x, strict=True))

        bound_gates = [
            gate.assign_parameters(
                {p: bind_map[p] for p in gate.parameters},
                inplace=False,
            )
            for gate in gates
        ]

        if verbose:
            print(f"[qcmps] Evaluation {counter['n']}: evaluating energy")

        energy = evaluate_hamiltonian(bound_gates, operator, verbose=False)
        energy = float(np.real_if_close(energy))

        if callback is not None:
            callback(counter["n"], energy, x.copy())

        if verbose:
            print(f"[qcmps] Evaluation {counter['n']}: E = {energy:.12f}")

        return energy

    return objective

def global_then_local(objective, n_params: int):
    bounds = [(-np.pi, np.pi)] * n_params

    print("[qcmps] Starting differential evolution")

    res_global = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=300,
        popsize=20,
        tol=1e-8,
        polish=False,
        updating="immediate",
        workers=1,
        seed=42,
    )

    print("[qcmps] Global result:")
    print(res_global.fun)

    print("[qcmps] Polishing with BFGS")

    res_local = minimize(
        objective,
        res_global.x,
        method="BFGS",
        options={
            "maxiter": 1000,
            "gtol": 1e-8,
            "disp": True,
        },
    )

    return res_global, res_local

def run(fcidump_path: Path, layers: int = 1, n_bond_qubits: int = N_BOND_QUBITS, optimizer: str = "COBYLA", maxiter: int = 1000, use_quantum_gates: bool = True):

    if use_quantum_gates:
        print("[qcmps] Using quantum gate-based evaluation")
        from gates_quantum import prepare_gates
    else:
        print("[qcmps] Using classical transfer matrix evaluation")
        from gates_classical import prepare_gates

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

    hamiltonian_eff = add_particle_number_penalty(
        hamiltonian,
        n_electrons=num_particles_total,
        penalty_strength=1.0,
    )

    hf_occ = occ_hf(num_alpha, num_beta, n_spatial_orbitals)

    n_orbs = n_spin_orbitals

    print("[qcmps] Creating random initial parameter vector")

    gates, params = prepare_gates(n_orbs, n_bond_qubits, layers, hf_occ)

    x0 = np.random.normal(0, 2, size=len(params))
    print(f"[qcmps] Initial vector has length {len(x0)}")

    def cb(it: int, energy: float, x: np.ndarray) -> None:
        print(f"eval={it:4d}  E={energy + nuclear_repulsion: .12f}")

    #objective = build_energy_objective_mps(gates, params, hamiltonian, cb, use_quantum_gates=use_quantum_gates)

    objective = build_energy_objective_with_penalty(
        gates,
        params,
        physical_hamiltonian=hamiltonian,
        penalty_hamiltonian=hamiltonian_eff,
        nuclear_repulsion=nuclear_repulsion,
        use_quantum_gates=use_quantum_gates,
    )

    print("[qcmps] Starting classical optimization")

    res_global, res_local = global_then_local(objective, len(params))

    print("[qcmps] Optimization finished")
    return res_global, res_local

def parse_args():
    parser = argparse.ArgumentParser(description="Run the QCMPS algorithm on a given FCIDump file.")
    parser.add_argument("fcidump_path", type=Path, help="Path to the FCIDump file.")
    parser.add_argument("--layers", type=int, default=1, help="Number of layers in the ansatz.")
    parser.add_argument("--bond_qubits", type=int, default=N_BOND_QUBITS, help="Number of bond qubits to use.")
    parser.add_argument("--optimizer", type=str, default="COBYLA", help="Optimizer to use for the classical optimization.")
    parser.add_argument("--maxiter", type=int, default=5000, help="Maximum number of iterations for the optimizer.")
    parser.add_argument("--use_quantum_gates", action="store_true", help="Use quantum gate-based evaluation.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    res_global, res_local = run(args.fcidump_path, layers=args.layers, n_bond_qubits=args.bond_qubits, optimizer=args.optimizer, maxiter=args.maxiter, use_quantum_gates=args.use_quantum_gates)

    print("Optimization result:")
    print(res_global)
    print(res_local)
