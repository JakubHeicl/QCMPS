from pathlib import Path
import numpy as np
import argparse

from .minimization import global_then_local, build_energy_objective_mps, local_only
from .loader import build_from_fcidump, hf_occupancy, load_initial_guess
from .qcmps_ansatz import prepare_blocks
from .circuit_blocks import ACUBlock, APCBlock, Block, AUBlock, LPCBlock, LUBlock

def run(fcidump_path: Path, n_bond_qubits: int, block_type: type[Block], layers: int = 1, optimizer: str = "COBYLA", maxiter: int = 1000, global_opt: bool = False, initial_guess: Path | None = None) -> tuple[dict[str, float] | None, dict[str, float]]:

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

    hf_occ = hf_occupancy(num_alpha, num_beta, n_spatial_orbitals)

    n_orbs = n_spin_orbitals

    print("[qcmps] Creating random initial parameter vector")

    blocks, params = prepare_blocks(n_orbs, n_bond_qubits, block_type, layers, hf_occ)

    min_energy = 0

    def cb(it: int, energy: float, x: np.ndarray) -> None:

        total_energy = energy + nuclear_repulsion
        print(f"eval={it:4d}  E={total_energy: .12f}")

        nonlocal min_energy
        min_energy = min(min_energy, total_energy)

        print(f"Current minimum energy: {min_energy: .12f}")

    objective = build_energy_objective_mps(blocks, params, hamiltonian, cb)

    print("[qcmps] Starting classical optimization")
    
    if global_opt:
        res_global, res_local = global_then_local(objective, len(params), optimizer=optimizer, maxiter=maxiter)
        print("[qcmps] Optimization finished")
        return res_global, res_local
    else:
        if initial_guess is not None:
            x0 = load_initial_guess(initial_guess)
        else:
            x0 = None
        res_local = local_only(objective, len(params), optimizer=optimizer, maxiter=maxiter, initial_guess=x0)
        print("[qcmps] Optimization finished")
        return None, res_local

def parse_args():
    parser = argparse.ArgumentParser(description="Run the QCMPS algorithm on a given FCIDump file.")
    parser.add_argument("fcidump_path", type=Path, help="Path to the FCIDump file.")
    parser.add_argument("--layers", type=int, default=1, help="Number of layers in the ansatz.")
    parser.add_argument("--bond_qubits", type=int, default=2, help="Number of bond qubits to use.")
    parser.add_argument("--optimizer", type=str, default="COBYLA", help="Optimizer to use for the classical optimization.")
    parser.add_argument("--block_type", type=str, default="AU", choices=["AU", "LU", "LPC", "APC", "ACU"], help="Type of block to use in the ansatz.")
    parser.add_argument("--maxiter", type=int, default=10000, help="Maximum number of iterations for the optimizer.")
    parser.add_argument("--global_opt", default=False, action="store_true", help="Whether to perform the global optimization step.")
    parser.add_argument("--initial_guess", type=Path, default=None, help="Path to a file containing an initial guess for the parameters (used only if --global_opt is not set).")
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
    else:
        block_type = LUBlock

    run(args.fcidump_path, args.bond_qubits, block_type, args.layers, args.optimizer, args.maxiter, args.global_opt, args.initial_guess)