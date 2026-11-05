from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._common import H4_FCIDUMP_PATH
from qcmps import (
    Ansatz,
    BrickWallLayerType,
    FCIDumpLoader,
    GlobalOptimizer,
    GlobalQCMPS,
    HFPreset,
    Logger,
    SO4Gate,
    SweepOptimizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overnight adaptive-layer H4 QCMPS benchmark."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--bond-qubits", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--initial-global-maxiter", type=int, default=150)
    parser.add_argument("--polish-global-maxiter", type=int, default=250)
    parser.add_argument("--max-sweeps", type=int, default=5)
    parser.add_argument("--local-maxiter", type=int, default=5)
    parser.add_argument("--sweep-tolerance", type=float, default=1e-6)
    parser.add_argument("--gtol", type=float, default=1e-6)
    parser.add_argument("--perturbation", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "h4_overnight_results",
    )
    return parser.parse_args()


def build_ansatz(n_bond_qubits: int, n_layers: int):
    loader = FCIDumpLoader(H4_FCIDUMP_PATH)
    loader.interleave_spin_orbitals()

    n_phys_qubits = loader.n_spin_orbitals
    n_total_qubits = n_phys_qubits + n_bond_qubits

    hamiltonian = SparsePauliOp.from_list(
        [
            ("I" * n_bond_qubits + label, coefficient)
            for label, coefficient in loader.hamiltonian.to_list()
        ]
    )

    ansatz = Ansatz(n_total_qubits)
    hf_preset = HFPreset(n_phys_qubits, loader.hf_occupancy())
    layers = [BrickWallLayerType(SO4Gate) for _ in range(n_layers)]
    qcmps = GlobalQCMPS(n_phys_qubits, n_bond_qubits, layers)

    ansatz.add_component(
        hf_preset,
        list(range(n_bond_qubits, n_total_qubits)),
    )
    ansatz.add_component(qcmps, list(range(n_total_qubits)))
    ansatz.add_hamiltonian(hamiltonian, list(range(n_total_qubits)))
    ansatz.add_nuclear_repulsion(loader.nuclear_repulsion)

    return ansatz, qcmps


def perturb_new_layer(
    qcmps: GlobalQCMPS,
    old_block_parameters: int,
    added_block_parameters: int,
    strength: float,
    seed: int,
) -> None:
    parameters = qcmps.current_parameters()
    rng = np.random.default_rng(seed)

    for block in range(qcmps.n_phys_qubits):
        start = block * qcmps.num_block_params + old_block_parameters
        stop = start + added_block_parameters
        parameters[start:stop] += rng.uniform(
            -strength,
            strength,
            added_block_parameters,
        )

    qcmps.update_parameters(parameters)


def result_dict(result) -> dict:
    return {
        "initial_electronic_energy": result.initial_energy,
        "final_electronic_energy": result.final_energy,
        "n_function_evaluations": result.n_function_evaluations,
        "n_iterations": result.n_iterations,
        "converged": bool(result.converged),
        "message": str(result.message),
        "energy_history": [float(value) for value in result.energy_history],
    }


def save_stage(
    output_dir: Path,
    seed: int,
    n_bond_qubits: int,
    layer: int,
    ansatz: Ansatz,
    qcmps: GlobalQCMPS,
    stage_results: dict,
) -> None:
    stem = f"h4_q{n_bond_qubits}_seed{seed}_layer{layer}"
    np.savez_compressed(
        output_dir / f"{stem}.npz",
        parameters=qcmps.current_parameters(),
    )

    record = {
        "system": "H4",
        "seed": seed,
        "n_bond_qubits": n_bond_qubits,
        "bond_dimension": 2**n_bond_qubits,
        "n_layers": layer,
        "n_parameters": qcmps.n_parameters,
        "total_energy": ansatz.total_energy(),
        **stage_results,
    }
    with (output_dir / f"{stem}.json").open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)


def latest_checkpoint(
    output_dir: Path,
    seed: int,
    n_bond_qubits: int,
    max_layers: int,
) -> tuple[int, Path | None]:
    for layer in range(max_layers, 0, -1):
        path = output_dir / f"h4_q{n_bond_qubits}_seed{seed}_layer{layer}.npz"
        if path.exists():
            return layer, path
    return 0, None


def run_seed(seed: int, args: argparse.Namespace) -> None:
    completed_layer, checkpoint = latest_checkpoint(
        args.output_dir,
        seed,
        args.bond_qubits,
        args.max_layers,
    )

    if completed_layer == args.max_layers:
        print(f"[skip] seed {seed} is already complete")
        return

    if checkpoint is None:
        ansatz, qcmps = build_ansatz(args.bond_qubits, 1)
        hf_energy = ansatz.total_energy()
        rng = np.random.default_rng(seed)
        qcmps.update_parameters(
            rng.uniform(0.0, 2.0 * np.pi, qcmps.n_parameters)
        )

        logger = Logger(
            str(
                args.output_dir
                / f"h4_q{args.bond_qubits}_seed{seed}_layer1.log"
            )
        )
        logger.log(f"Seed: {seed}")
        logger.log(f"HF energy: {hf_energy:.12f}")
        logger.log(f"Initial energy: {ansatz.total_energy():.12f}")

        optimizer = GlobalOptimizer(logger, args.log_every)
        global_result = optimizer.optimize(
            ansatz,
            qcmps,
            options={
                "maxiter": args.initial_global_maxiter,
                "gtol": args.gtol,
            },
        )

        logger.log(f"Final energy: {ansatz.total_energy():.12f}")
        save_stage(
            args.output_dir,
            seed,
            args.bond_qubits,
            1,
            ansatz,
            qcmps,
            {"global": result_dict(global_result)},
        )
        completed_layer = 1
    else:
        ansatz, qcmps = build_ansatz(args.bond_qubits, completed_layer)
        with np.load(checkpoint) as data:
            qcmps.update_parameters(data["parameters"])
        print(
            f"[resume] seed {seed} from layer {completed_layer}, "
            f"energy {ansatz.total_energy():.12f}"
        )

    for layer in range(completed_layer + 1, args.max_layers + 1):
        logger = Logger(
            str(
                args.output_dir
                / f"h4_q{args.bond_qubits}_seed{seed}_layer{layer}.log"
            )
        )
        logger.log(f"Seed: {seed}, adding layer {layer}")
        logger.log(f"Energy before adding layer: {ansatz.total_energy():.12f}")

        new_layer = BrickWallLayerType(SO4Gate)
        old_block_parameters = qcmps.num_block_params
        added_block_parameters = new_layer.n_parameters(qcmps.block_entries)

        qcmps.add_layers([new_layer])
        ansatz.rebuild()
        perturb_new_layer(
            qcmps,
            old_block_parameters,
            added_block_parameters,
            args.perturbation,
            seed=seed * 10_000 + layer,
        )
        logger.log(f"Energy after perturbation: {ansatz.total_energy():.12f}")

        sweep = SweepOptimizer(
            logger=logger,
            log_every_n_steps=args.log_every,
            max_sweeps=args.max_sweeps,
            local_maxiter=args.local_maxiter,
            tolerance=args.sweep_tolerance,
        )
        sweep_result = sweep.optimize(
            ansatz,
            qcmps,
            options={"gtol": args.gtol},
        )
        logger.log(f"Energy after sweep: {ansatz.total_energy():.12f}")

        polish = GlobalOptimizer(logger, args.log_every)
        polish_result = polish.optimize(
            ansatz,
            qcmps,
            options={
                "maxiter": args.polish_global_maxiter,
                "gtol": args.gtol,
            },
        )
        logger.log(f"Final energy: {ansatz.total_energy():.12f}")

        save_stage(
            args.output_dir,
            seed,
            args.bond_qubits,
            layer,
            ansatz,
            qcmps,
            {
                "sweep": result_dict(sweep_result),
                "global_polish": result_dict(polish_result),
            },
        )


def write_summary(output_dir: Path) -> None:
    records = []
    for path in sorted(output_dir.glob("h4_*.json")):
        with path.open(encoding="utf-8") as file:
            records.append(json.load(file))

    if not records:
        return

    columns = [
        "system",
        "seed",
        "n_bond_qubits",
        "bond_dimension",
        "n_layers",
        "n_parameters",
        "total_energy",
    ]
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if args.bond_qubits < 1:
        raise ValueError("bond-qubits must be at least 1")
    if args.max_layers < 1:
        raise ValueError("max-layers must be at least 1")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        write_summary(args.output_dir)
        print(f"Summary written to {args.output_dir / 'summary.csv'}")
        return

    for seed in args.seeds:
        try:
            run_seed(seed, args)
        except Exception:
            error_path = args.output_dir / (
                f"h4_q{args.bond_qubits}_seed{seed}_error.txt"
            )
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[failed] seed {seed}; details in {error_path}")

    if not args.skip_summary:
        write_summary(args.output_dir)
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
