from pathlib import Path
import numpy as np
import argparse

from .minimization import global_then_local, build_energy_objective_mps, local_only, one_site_sweep_optimize, prescan, random_preoptimize, two_site_sweep_optimize
from .loader import add_particle_number_penalty, build_from_fcidump, hf_occupancy, load_initial_guess, truncate_pauli_terms
from .mps_initial_guess import available_mps_initial_guesses, load_mps_initial_guess
from .qcmps_ansatz import prepare_blocks
from .circuit_blocks import ACUBlock, APCBlock, Block, AUBlock, GUBlock, LPCBlock, LUBlock

def run(
    fcidump_path: Path,
    n_bond_qubits: int | None,
    block_type: type[Block],
    layers: int = 1,
    optimizer: str = "COBYLA",
    maxiter: int = 1000,
    type_opt: str = "local",
    initial_guess: Path | None = None,
    mps_initial_guess: str | None = None,
    hamiltonian_cutoff: float = 0.0,
    use_global_ansatz: bool = False,
    particle_penalty: float = 0.0,
) -> tuple[dict[str, float] | None, dict[str, float]]:

    if particle_penalty < 0.0:
        raise ValueError(f"particle_penalty must be non-negative, got {particle_penalty}.")

    if n_bond_qubits is None:
        n_bond_qubits = 0 if mps_initial_guess is not None else 2

    print("[qcmps] Starting QCMPS run")
    print(
        "[qcmps] Settings: "
        f"fcidump={fcidump_path}, "
        f"layers={layers}, "
        f"bond_qubits={n_bond_qubits}, "
        f"optimizer={optimizer}, "
        f"maxiter={maxiter}, "
        f"hamiltonian_cutoff={hamiltonian_cutoff}, "
        f"global_ansatz={use_global_ansatz}, "
        f"particle_penalty={particle_penalty}"
    )

    hamiltonian, nuclear_repulsion, n_spin_orbitals, n_spatial_orbitals, num_alpha, num_beta, num_particles_total = build_from_fcidump(fcidump_path)
    hamiltonian = truncate_pauli_terms(hamiltonian, hamiltonian_cutoff)
    if particle_penalty > 0.0:
        print(
            "[qcmps] Adding particle-number penalty "
            f"{particle_penalty} * (N - {num_particles_total})^2"
        )
        hamiltonian = add_particle_number_penalty(
            hamiltonian,
            num_particles_total,
            particle_penalty,
        )

    mps_unitaries = None
    n_mps_bond_qubits = 0
    n_orbs = n_spin_orbitals

    if mps_initial_guess is None:
        hf_occ = hf_occupancy(num_alpha, num_beta, n_spatial_orbitals)
    else:
        print(f"[qcmps] Loading MPS initial guess '{mps_initial_guess}'")
        mps_unitaries, n_mps_bond_qubits = load_mps_initial_guess(mps_initial_guess)
        hf_occ = None

        if len(mps_unitaries) != n_spin_orbitals:
            global_ansatz_qubits = len(mps_unitaries) + n_mps_bond_qubits + n_bond_qubits

            if not use_global_ansatz or global_ansatz_qubits != hamiltonian.num_qubits:
                raise ValueError(
                    f"MPS initial guess has {len(mps_unitaries)} sites, "
                    f"but Hamiltonian has {n_spin_orbitals} spin orbitals "
                    f"({hamiltonian.num_qubits} qubits). With the current bond settings, "
                    f"the global ansatz would have {global_ansatz_qubits} qubits."
                )

            print(
                "[qcmps] MPS site count differs from spin-orbital count, "
                f"but global ansatz has {global_ansatz_qubits} qubits matching the Hamiltonian"
            )

        n_orbs = len(mps_unitaries)

        print(
            "[qcmps] MPS adds "
            f"{n_mps_bond_qubits} bond ancilla qubit(s); "
            f"{n_bond_qubits} extra variational bond qubit(s)"
        )

    print("[qcmps] Creating QCMPS blocks and parameter vector")

    blocks, params = prepare_blocks(
        n_orbs,
        n_bond_qubits,
        block_type,
        layers,
        hf_occ,
        mps_unitaries,
        n_mps_bond_qubits,
    )

    min_energy = 0

    def cb(it: int, energy: float, x: np.ndarray) -> None:

        total_energy = energy + nuclear_repulsion
        print(f"eval={it:4d}  E={total_energy: .12f}")

        nonlocal min_energy
        min_energy = min(min_energy, total_energy)

        print(f"Current minimum energy: {min_energy: .12f}")

    objective = build_energy_objective_mps(
        blocks,
        params,
        hamiltonian,
        cb,
        use_global_ansatz=use_global_ansatz,
    )

    if type_opt == "evaluate":
        print("[qcmps] Evaluating initial parameters")
    else:
        print("[qcmps] Starting optimization")
    
    match type_opt:
        case "evaluate":
            if initial_guess is not None:
                x0 = load_initial_guess(initial_guess)
            else:
                x0 = np.zeros(len(params))
            energy = objective(x0)
            print(f"[qcmps] Initial total energy: {energy + nuclear_repulsion: .12f}")
            print("[qcmps] Evaluation finished")
            return None, {"x": x0, "fun": energy}
        case "global":
            res_global, res_local = global_then_local(objective, len(params), optimizer=optimizer, maxiter=maxiter)
            print("[qcmps] Optimization finished")
            return res_global, res_local
        case "local":
            if initial_guess is not None:
                x0 = load_initial_guess(initial_guess)
            elif mps_unitaries is not None:
                x0 = np.zeros(len(params))
            else:
                x0 = None
            res_local = local_only(objective, len(params), optimizer=optimizer, maxiter=maxiter, initial_guess=x0)
            print("[qcmps] Optimization finished")
            return None, res_local
        case "one_site_sweep":
            x, final_e = one_site_sweep_optimize(objective, len(params), n_orbs, n_sweeps=10, local_maxiter=maxiter, local_popsize=5)
            print("[qcmps] Optimization finished")
            return None, {"x": x, "fun": final_e}
        case "two_site_sweep":
            x, final_e = two_site_sweep_optimize(objective, len(params), n_orbs, n_sweeps=10, local_maxiter=maxiter, local_popsize=5)
            print("[qcmps] Optimization finished")
            return None, {"x": x, "fun": final_e}
        case "prescan":
            prescan(objective, len(params), n_scans=1000)
            print("[qcmps] Optimization finished")
            return None, {"x": None, "fun": None}
        case "random_preoptimize":
            x, final_e = random_preoptimize(objective, len(params), n_trials=100)
            print("[qcmps] Optimization finished")
            return None, {"x": x, "fun": final_e}
        case _:
            raise ValueError(f"Invalid optimization type: {type_opt}")
        
def parse_args():
    parser = argparse.ArgumentParser(description="Run the QCMPS algorithm on a given FCIDump file.")
    
    parser.add_argument("fcidump_path", type=Path, help="Path to the FCIDump file.")
    parser.add_argument("--layers", type=int, default=1, help="Number of layers in the ansatz.")
    parser.add_argument("--bond_qubits", type=int, default=None, help="Number of extra variational bond qubits to use. Defaults to 2 without MPS and 0 with MPS.")
    parser.add_argument("--optimizer", type=str, default="COBYLA", help="Optimizer to use for the classical optimization.")
    parser.add_argument("--block_type", type=str, default="AU", choices=["AU", "LU", "LPC", "APC", "ACU", "GU"], help="Type of block to use in the ansatz.")
    parser.add_argument("--maxiter", type=int, default=10000, help="Maximum number of iterations for the optimizer.")
    parser.add_argument("--type_opt", type=str, default="local", choices=["evaluate", "global", "local", "one_site_sweep", "two_site_sweep", "prescan", "random_preoptimize"], help="Type of optimization to perform.")
    parser.add_argument("--initial_guess", type=Path, default=None, help="Path to a file containing an initial guess for the parameters (used only if --global_opt is not set).")
    parser.add_argument("--mps_initial_guess", type=str, default=None, choices=available_mps_initial_guesses(), help="Built-in MPS initial guess to use instead of HF.")
    parser.add_argument("--hamiltonian_cutoff", type=float, default=0.0, help="Drop Pauli terms with abs(coeff) below this value.")
    parser.add_argument("--global_ansatz", action="store_true", help="Evaluate the explicit global QCMPS ansatz instead of the Hadamard-test transfer circuit.")
    parser.add_argument("--particle_penalty", type=float, default=0.0, help="Add this strength times (N - N_e)^2 to keep the ansatz in the target particle-number sector.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.block_type == "AU":
        block_type = AUBlock
    elif args.block_type == "LPC":
        block_type = LPCBlock
    elif args.block_type == "APC":
        block_type = APCBlock
    elif args.block_type == "ACU":
        block_type = ACUBlock
    elif args.block_type == "GU":
        block_type = GUBlock
    else:
        block_type = LUBlock

    run(
        args.fcidump_path,
        args.bond_qubits,
        block_type,
        args.layers,
        args.optimizer,
        args.maxiter,
        args.type_opt,
        args.initial_guess,
        args.mps_initial_guess,
        args.hamiltonian_cutoff,
        args.global_ansatz,
        args.particle_penalty,
    )
