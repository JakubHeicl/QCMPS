"""Command-line runner for the differentiable PyTorch QCMPS ansatz.

The default configuration runs the H4_FCIDUMP3 calculation::

    python run_torch_qcmps.py

Use ``python run_torch_qcmps.py --help`` to see all options.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from qcmps.mps_initial_guess import (
    available_mps_initial_guesses,
    load_mps_initial_guess,
)
from qcmps.torch_ansatz import (
    SparsePauliHamiltonian,
    TorchQCMPS,
    optimize_qcmps,
)


DEFAULT_FCIDUMP = Path("fcidumps/H4_FCIDUMP3")
DEFAULT_OUTPUT = Path("H4/torch_qcmps_H4_FCIDUMP3.pt")
KNOWN_HF_OCCUPANCIES = {
    # This FCIDUMP orders the occupied spatial orbitals as 0 and 2.  After the
    # spin-block -> interleaved permutation their spin orbitals are 0, 1, 4, 5.
    "H4_FCIDUMP3": [0, 1, 4, 5],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a global QCMPS ansatz with PyTorch gradient descent."
    )
    parser.add_argument(
        "fcidump_path",
        nargs="?",
        type=Path,
        default=DEFAULT_FCIDUMP,
        help=f"FCIDUMP input file (default: {DEFAULT_FCIDUMP}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Checkpoint output path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--bond-qubits",
        type=int,
        default=0,
        help=(
            "Number of extra variational bond qubits in addition to the MPS "
            "bond qubit (default: 0)."
        ),
    )
    parser.add_argument(
        "--mps-initial-guess",
        choices=("none", *available_mps_initial_guesses()),
        default="h4_stretched_bd2",
        help=(
            "MPS state used before the trainable blocks; 'none' uses HF "
            "(default: h4_stretched_bd2)."
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of general-unitary layers per site (default: 1).",
    )
    parser.add_argument(
        "--shared",
        action="store_true",
        help="Reuse one trainable block at every site.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of Adam steps (default: 1000).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-2,
        help="Adam learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--initial-scale",
        type=float,
        default=0.0,
        help=(
            "Standard deviation of initial parameters. Zero reproduces the "
            "input MPS in the first step (default: 0)."
        ),
    )
    parser.add_argument(
        "--particle-penalty",
        type=float,
        default=1.0,
        help="Strength of (N - N_e)^2 added during training (default: 1).",
    )
    parser.add_argument(
        "--no-nuclear-repulsion",
        action="store_true",
        help="Omit nuclear repulsion from per-step reported energies.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PyTorch random seed (default: 42).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Real parameter precision (default: float64).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device (default: auto).",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print progress every this many steps; zero disables it (default: 1).",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def run(args: argparse.Namespace) -> dict[str, object]:
    # qiskit-nature is only needed to parse and map the FCIDUMP.  Keeping this
    # import local allows --help to work even before that optional dependency is
    # installed.
    try:
        from qcmps.loader import (
            add_particle_number_penalty,
            build_from_fcidump,
            hf_occupancy,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "qiskit_nature":
            raise RuntimeError(
                "qiskit-nature is required to load FCIDUMP files. Install it with "
                "`python -m pip install qiskit-nature`."
            ) from exc
        raise

    if not args.fcidump_path.is_file():
        raise FileNotFoundError(f"FCIDUMP file not found: {args.fcidump_path}")
    if args.bond_qubits < 0:
        raise ValueError("--bond-qubits must be non-negative.")
    if args.layers < 1:
        raise ValueError("--layers must be positive.")
    if args.steps < 1:
        raise ValueError("--steps must be positive.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args.initial_scale < 0.0:
        raise ValueError("--initial-scale must be non-negative.")
    if args.particle_penalty < 0.0:
        raise ValueError("--particle-penalty must be non-negative.")
    if args.print_every < 0:
        raise ValueError("--print-every must be non-negative.")

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print("[torch-qcmps] Loading", args.fcidump_path)
    (
        physical_hamiltonian,
        nuclear_repulsion,
        n_spin_orbitals,
        n_spatial_orbitals,
        num_alpha,
        num_beta,
        num_electrons,
    ) = build_from_fcidump(args.fcidump_path)

    occupied: list[int] | None = None
    mps_unitaries = None
    n_mps_bond_qubits = 0
    if args.mps_initial_guess == "none":
        occupied = KNOWN_HF_OCCUPANCIES.get(args.fcidump_path.name)
        if occupied is None:
            occupied = hf_occupancy(num_alpha, num_beta, n_spatial_orbitals)
        else:
            occupied = occupied.copy()
        initial_state_description = "HF"
    else:
        mps_unitaries, n_mps_bond_qubits = load_mps_initial_guess(
            args.mps_initial_guess
        )
        if len(mps_unitaries) != n_spin_orbitals:
            raise ValueError(
                f"MPS initial guess '{args.mps_initial_guess}' has "
                f"{len(mps_unitaries)} sites, but the Hamiltonian has "
                f"{n_spin_orbitals} spin orbitals."
            )
        initial_state_description = args.mps_initial_guess

    training_hamiltonian = physical_hamiltonian
    if args.particle_penalty > 0.0:
        training_hamiltonian = add_particle_number_penalty(
            physical_hamiltonian,
            num_electrons,
            args.particle_penalty,
        )

    model = TorchQCMPS(
        n_sites=n_spin_orbitals,
        n_bond_qubits=args.bond_qubits,
        layers=args.layers,
        shared=args.shared,
        hf_occ=occupied,
        mps_unitaries=mps_unitaries,
        n_mps_bond_qubits=n_mps_bond_qubits,
        initial_scale=args.initial_scale,
        dtype=dtype,
        device=device,
    )
    training_operator = SparsePauliHamiltonian.from_sparse_pauli_op(
        training_hamiltonian,
        dtype=dtype,
        device=device,
    )
    physical_operator = SparsePauliHamiltonian.from_sparse_pauli_op(
        physical_hamiltonian,
        dtype=dtype,
        device=device,
    )

    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        "[torch-qcmps] Settings: "
        f"spin_orbitals={n_spin_orbitals}, electrons={num_electrons}, "
        f"initial_state={initial_state_description}, "
        f"mps_bond_qubits={n_mps_bond_qubits}, "
        f"extra_bond_qubits={args.bond_qubits}, "
        f"total_block_bond_qubits={n_mps_bond_qubits + args.bond_qubits}, "
        f"layers={args.layers}, "
        f"shared={args.shared}, parameters={n_parameters}, device={device}, "
        f"dtype={args.dtype}, steps={args.steps}, lr={args.learning_rate}, "
        f"particle_penalty={args.particle_penalty}, "
        f"include_nuclear_repulsion={not args.no_nuclear_repulsion}"
    )
    if occupied is not None:
        print(f"[torch-qcmps] HF occupied sites: {occupied}")

    def report_state(
        step: int,
        penalized_electronic_energy: float,
        state: torch.Tensor,
    ) -> None:
        if args.print_every and (
            (step + 1) % args.print_every == 0 or step == args.steps - 1
        ):
            with torch.no_grad():
                physical_electronic_energy = float(physical_operator(state).cpu())

            if args.no_nuclear_repulsion:
                print(
                    f"step={step + 1:5d}  "
                    f"electronic_energy={physical_electronic_energy: .12f}  "
                    f"penalized_electronic_energy="
                    f"{penalized_electronic_energy: .12f}"
                )
            else:
                physical_total = physical_electronic_energy + nuclear_repulsion
                penalized_total = penalized_electronic_energy + nuclear_repulsion
                print(
                    f"step={step + 1:5d}  "
                    f"total_energy={physical_total: .12f}  "
                    f"penalized_total_energy={penalized_total: .12f}"
                )

    history = optimize_qcmps(
        model,
        training_operator,
        steps=args.steps,
        learning_rate=args.learning_rate,
        state_callback=report_state,
    )

    with torch.no_grad():
        final_state = model()
        electronic_energy = float(physical_operator(final_state).cpu())
        penalized_electronic_energy = float(training_operator(final_state).cpu())
        state_norm = float(torch.vdot(final_state, final_state).real.cpu())

    total_energy = electronic_energy + nuclear_repulsion
    penalized_total_energy = penalized_electronic_energy + nuclear_repulsion

    print(f"[torch-qcmps] Electronic energy:       {electronic_energy: .12f}")
    print(f"[torch-qcmps] Total energy:            {total_energy: .12f}")
    print(f"[torch-qcmps] Penalized total energy:  {penalized_total_energy: .12f}")
    print(f"[torch-qcmps] State norm:              {state_norm: .12f}")

    checkpoint: dict[str, object] = {
        "model_state_dict": model.state_dict(),
        "history": history,
        "electronic_energy": electronic_energy,
        "total_energy": total_energy,
        "penalized_total_energy": penalized_total_energy,
        "nuclear_repulsion": nuclear_repulsion,
        "state_norm": state_norm,
        "config": {
            "fcidump_path": str(args.fcidump_path),
            "n_sites": n_spin_orbitals,
            "n_bond_qubits": args.bond_qubits,
            "mps_initial_guess": args.mps_initial_guess,
            "n_mps_bond_qubits": n_mps_bond_qubits,
            "layers": args.layers,
            "shared": args.shared,
            "hf_occ": occupied,
            "initial_scale": args.initial_scale,
            "particle_penalty": args.particle_penalty,
            "no_nuclear_repulsion": args.no_nuclear_repulsion,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "dtype": args.dtype,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print("[torch-qcmps] Saved checkpoint to", args.output)
    return checkpoint


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"[torch-qcmps] Error: {exc}") from exc


if __name__ == "__main__":
    main()
