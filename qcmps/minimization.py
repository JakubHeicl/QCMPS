from typing import Callable

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

import numpy as np
from scipy.optimize import minimize, differential_evolution

from .qcmps_ansatz import evaluate_hamiltonian

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

def global_then_local(objective, n_params: int, optimizer: str = "COBYLA", maxiter: int = 10000):
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
            "disp": True,
        },
    )

    return res_global, res_local

def local_only(objective, n_params: int, optimizer: str = "COBYLA", maxiter: int = 10000, initial_guess: np.ndarray | None = None):
    
    if initial_guess is None:
        initial_guess = np.random.uniform(-np.pi, np.pi, size=n_params)

    print("[qcmps] Starting local optimization with " + optimizer)

    res_local = minimize(
        objective,
        initial_guess,
        method=optimizer,
        options={
            "maxiter": maxiter,
            "disp": True,
        },
    )

    return res_local