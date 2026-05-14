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
        
    #initial_guess = np.random.normal(loc=0.0, scale=0.1, size=n_params)

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

def one_site_sweep_optimize(objective, n_params: int, n_orbs: int, n_sweeps: int = 10, local_maxiter: int = 30, local_popsize: int = 5):
    
    local_maxiter = 30
    local_popsize = 5
    
    block_n_params = n_params // n_orbs
    x = np.zeros(n_params)
    
    best_e = 0

    for sweep in range(n_sweeps):
        print(f"\n[sweep] sweep {sweep + 1}/{n_sweeps}")

        if sweep % 2 == 0:
            site_order = range(n_orbs)
        else:
            site_order = reversed(range(n_orbs))

        for site in site_order:
            sl = slice(site * block_n_params, (site + 1) * block_n_params)
            
            dim = sl.stop - sl.start

            bounds = [(-np.pi, np.pi)] * dim

            def local_objective(y):
                x_trial = x.copy()
                x_trial[sl] = y
                return objective(x_trial)
            
            print(f"[sweep] Optimizing site {site + 1}/{n_orbs} with differential evolution")

            res = differential_evolution(
                local_objective,
                bounds=bounds,
                maxiter=local_maxiter,
                popsize=local_popsize,
                polish=False,
                seed=None,
                workers=1,
                updating="immediate",
            )

            x[sl] = res.x

            if res.fun < best_e:
                best_e = res.fun

            print(
                f"[sweep] site={site:3d} "
                f"E_site={res.fun:.12f} "
                f"E_best={best_e:.12f}"
            )

    final_e = objective(x)
    return x, final_e

def prescan(objective, n_params: int, n_scans: int = 10000):
    
    n_scans = 1000
    
    best_e = 0

    for scan in range(n_scans):
        x = np.random.uniform(-np.pi, np.pi, size=n_params)
        e = objective(x)

        if e < best_e:
            best_e = e

def random_preoptimize(objective, n_params: int, n_trials: int = 100):
    n_trials = 1000
    
    best_e = 0
    best_x = None

    for trial in range(n_trials):
        x = np.random.uniform(-np.pi, np.pi, size=n_params)
        
        res = minimize(
            objective,
            x,
            method="COBYLA",
            options={
                "maxiter": 100,
                "disp": True,
            },
        )

        if res.fun < best_e:
            best_e = res.fun
            best_x = res.x.copy()

    return best_x, best_e

def two_site_sweep_optimize(objective, n_params: int, n_orbs: int, n_sweeps: int = 10, local_maxiter: int = 30, local_popsize: int = 5, bounds_width: float = np.pi, initial_scale: float = 0.0, seed: int | None = None):
    
    initial_scale = 0.0
    
    block_n_params = n_params // n_orbs
    
    if block_n_params * n_orbs != n_params:
        raise ValueError(
            f"n_params={n_params} is not divisible by n_orbs={n_orbs}."
        )

    rng = np.random.default_rng(seed)

    if initial_scale == 0.0:
        x = np.zeros(n_params)
    else:
        x = rng.uniform(-initial_scale, initial_scale, size=n_params)

    current_e = objective(x)
    best_x = x.copy()
    best_e = current_e

    print(f"[two-site sweep] initial E = {current_e:.12f}")

    for sweep in range(n_sweeps):
        print(f"\n[two-site sweep] sweep {sweep + 1}/{n_sweeps}")

        if sweep % 2 == 0:
            pair_order = range(n_orbs - 1)
        else:
            pair_order = reversed(range(n_orbs - 1))

        for site in pair_order:
            sl = slice(
                site * block_n_params,
                (site + 2) * block_n_params,
            )

            dim = sl.stop - sl.start
            bounds = [(-bounds_width, bounds_width)] * dim

            x_before = x.copy()
            e_before = current_e

            def local_objective(y):
                x_trial = x.copy()
                x_trial[sl] = y
                return objective(x_trial)

            print(
                f"[two-site sweep] Optimizing sites "
                f"({site + 1},{site + 2})/{n_orbs}, "
                f"dim={dim}, E_before={e_before:.12f}"
            )
            
            res = differential_evolution(
                local_objective,
                bounds=bounds,
                maxiter=local_maxiter,
                popsize=local_popsize,
                polish=False,
                seed=int(rng.integers(0, 2**32 - 1)),
                workers=1,
                updating="immediate",
                x0=x[sl],
            )
            
            if res.fun < e_before:
                x[sl] = res.x
                current_e = res.fun
                accepted = True
            else:
                x = x_before
                current_e = e_before
                accepted = False

            if current_e < best_e:
                best_e = current_e
                best_x = x.copy()

            print(
                f"[two-site sweep] sites=({site:3d},{site+1:3d}) "
                f"E_before={e_before:.12f} "
                f"E_pair={res.fun:.12f} "
                f"E_current={current_e:.12f} "
                f"E_best={best_e:.12f} "
                f"accepted={accepted}"
            )

    final_e = objective(best_x)
    return best_x, final_e