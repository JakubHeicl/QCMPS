from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_BOND_DIMENSIONS = (2, 4, 8, 16)


def _parse_arguments(system_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Compute fixed-bond-dimension DMRG reference energies for "
            f"{system_name} from its FCIDUMP file."
        )
    )
    parser.add_argument(
        "--bond-dimensions",
        type=int,
        nargs="+",
        default=list(DEFAULT_BOND_DIMENSIONS),
        metavar="M",
        help="DMRG bond dimensions to evaluate (default: 2 4 8 16).",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=30,
        help="Maximum number of DMRG sweeps for each bond dimension.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-10,
        help="DMRG energy convergence tolerance.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of Block2 threads.",
    )
    parser.add_argument(
        "--memory-gb",
        type=int,
        default=2,
        help="Memory limit passed to Block2 in GB.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=3,
        help="Number of independent DMRG initializations for each M.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed used for reproducible DMRG initializations.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="Block2 output level (default: 1).",
    )
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if not arguments.bond_dimensions:
        raise ValueError("At least one bond dimension must be specified.")

    if any(bond_dimension < 1 for bond_dimension in arguments.bond_dimensions):
        raise ValueError("Bond dimensions must be positive integers.")

    if len(set(arguments.bond_dimensions)) != len(arguments.bond_dimensions):
        raise ValueError("Bond dimensions must not contain duplicates.")

    if arguments.max_sweeps < 1:
        raise ValueError("max-sweeps must be positive.")

    if arguments.tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    if arguments.threads < 1:
        raise ValueError("threads must be positive.")

    if arguments.memory_gb < 1:
        raise ValueError("memory-gb must be positive.")

    if arguments.restarts < 1:
        raise ValueError("restarts must be positive.")

    if arguments.seed < 0:
        raise ValueError("seed must be non-negative.")


def _load_dependencies():
    try:
        from pyscf import tools
    except ImportError as error:
        raise RuntimeError(
            "PySCF is not available. Install it inside the active virtual "
            "environment with:\n"
            "  python -m pip install pyscf"
        ) from error

    try:
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    except ImportError as error:
        raise RuntimeError(
            "Block2's Python interface is not available. Install it inside "
            "the active virtual environment with:\n"
            "  python -m pip install block2"
        ) from error

    return tools, DMRGDriver, SymmetryTypes


def _hf_spatial_occupations(
    number_of_orbitals: int,
    number_of_electrons: int,
    spin_twice: int,
) -> list[float]:
    if (number_of_electrons + spin_twice) % 2:
        raise ValueError("NELEC and MS2 do not define integer alpha/beta counts.")

    number_of_alpha = (number_of_electrons + spin_twice) // 2
    number_of_beta = number_of_electrons - number_of_alpha

    if not 0 <= number_of_beta <= number_of_alpha <= number_of_orbitals:
        raise ValueError("Invalid electron or spin count for the active space.")

    return (
        [2.0] * number_of_beta
        + [1.0] * (number_of_alpha - number_of_beta)
        + [0.0] * (number_of_orbitals - number_of_alpha)
    )


def _noise_schedule(number_of_sweeps: int) -> list[float]:
    schedule = []
    for sweep in range(number_of_sweeps):
        if sweep < 4:
            schedule.append(1e-3)
        elif sweep < 8:
            schedule.append(1e-4)
        elif sweep < 12:
            schedule.append(1e-5)
        else:
            schedule.append(0.0)

    # Convergence is tested only during zero-noise sweeps. Always reserve at
    # least the last two sweeps for an unperturbed variational refinement.
    for sweep in range(max(0, number_of_sweeps - 2), number_of_sweeps):
        schedule[sweep] = 0.0

    return schedule


def _serialize_sweep_history(driver) -> list[dict]:
    bond_dimensions, discarded_weights, energies = driver.get_dmrg_results()
    history = []

    for sweep, (bond_dimension, discarded_weight, energy) in enumerate(
        zip(bond_dimensions, discarded_weights, energies)
    ):
        history.append(
            {
                "sweep": sweep,
                "bond_dimension": int(bond_dimension),
                "discarded_weight": float(discarded_weight),
                "energy_hartree": float(np.asarray(energy).reshape(-1)[0]),
            }
        )

    return history


def _write_json(path: Path, data: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nominal_qcmps_bond_qubits(bond_dimension: int) -> int | None:
    if bond_dimension & (bond_dimension - 1):
        return None
    return int(math.log2(bond_dimension))


def run_dmrg_preparation(
    system_name: str,
    system_directory: Path,
    fcidump_filename: str,
    reference_filename: str,
    output_filename: str,
) -> None:
    arguments = _parse_arguments(system_name)
    _validate_arguments(arguments)

    tools, DMRGDriver, SymmetryTypes = _load_dependencies()

    system_directory = system_directory.resolve()
    fcidump_path = system_directory / fcidump_filename
    reference_path = system_directory / reference_filename
    output_path = system_directory / output_filename

    if not fcidump_path.is_file():
        raise FileNotFoundError(f"FCIDUMP file not found: {fcidump_path}")

    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    fcidump = tools.fcidump.read(str(fcidump_path), verbose=False)

    number_of_orbitals = int(fcidump["NORB"])
    number_of_electrons = int(fcidump["NELEC"])
    spin_twice = int(fcidump.get("MS2", reference.get("spin_2S", 0)))
    core_energy = float(fcidump["ECORE"])

    if number_of_orbitals != int(reference["fcidump_norb"]):
        raise ValueError("FCIDUMP and reference disagree about NORB.")

    if number_of_electrons != int(reference["fcidump_nelec"]):
        raise ValueError("FCIDUMP and reference disagree about NELEC.")

    hf_occupations = _hf_spatial_occupations(
        number_of_orbitals,
        number_of_electrons,
        spin_twice,
    )
    noise_schedule = _noise_schedule(arguments.max_sweeps)
    davidson_threshold = max(arguments.tolerance * 0.1, 1e-12)

    exact_energy = float(reference["reference_total_energy_hartree"])
    hartree_fock_energy = float(reference["scf_total_energy_hartree"])
    correlation_energy = hartree_fock_energy - exact_energy

    output = {
        "format_version": 2,
        "system": system_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fcidump": fcidump_path.name,
        "fcidump_sha256": _file_sha256(fcidump_path),
        "norb": number_of_orbitals,
        "nelec": number_of_electrons,
        "spin_2S": spin_twice,
        "ecore_hartree": core_energy,
        "hartree_fock_energy_hartree": hartree_fock_energy,
        "exact_reference_energy_hartree": exact_energy,
        "backend": {
            "interface": "pyblock2.driver.core.DMRGDriver",
            "solver": "block2",
        },
        "settings": {
            "bond_dimensions": arguments.bond_dimensions,
            "max_sweeps": arguments.max_sweeps,
            "tolerance": arguments.tolerance,
            "davidson_threshold": davidson_threshold,
            "threads": arguments.threads,
            "memory_gb": arguments.memory_gb,
            "restarts": arguments.restarts,
            "base_seed": arguments.seed,
            "noise_schedule": noise_schedule,
            "initial_occupations": hf_occupations,
            "orbital_ordering": "FCIDUMP spatial-orbital order",
            "site_representation": "spatial orbitals with local dimension 4",
            "spin_adaptation": "non-spin-adapted SZ representation",
            "sweep_algorithm": "two-site",
        },
        "comparison_note": (
            "A spatial-orbital DMRG bond dimension is directly comparable to "
            "the QCMPS spin-orbital ordering only at cuts between complete "
            "alpha/beta orbital pairs; it does not impose the additional cut "
            "inside each pair."
        ),
        "results": [],
    }
    _write_json(output_path, output)

    for bond_dimension in arguments.bond_dimensions:
        print(
            f"\n[{system_name}] Starting DMRG with bond dimension "
            f"M={bond_dimension}..."
        )

        attempts = []
        for restart in range(arguments.restarts):
            seed = arguments.seed + restart
            print(
                f"[{system_name}] M={bond_dimension}, initialization "
                f"{restart + 1}/{arguments.restarts}, seed={seed}"
            )

            with tempfile.TemporaryDirectory(
                prefix=(
                    f"{system_name.lower()}_dmrg_m{bond_dimension}_"
                    f"r{restart}_"
                )
            ) as temporary_directory:
                driver = DMRGDriver(
                    scratch=temporary_directory,
                    symm_type=SymmetryTypes.SZ,
                    stack_mem=arguments.memory_gb << 30,
                    n_threads=arguments.threads,
                )
                driver.read_fcidump(
                    filename=str(fcidump_path),
                    pg="d2h",
                    iprint=0,
                )
                driver.initialize_system(
                    n_sites=driver.n_sites,
                    n_elec=driver.n_elec,
                    spin=driver.spin,
                    orb_sym=driver.orb_sym,
                )
                driver.bw.b.Random.rand_seed(seed)

                mpo = driver.get_qc_mpo(
                    h1e=driver.h1e,
                    g2e=driver.g2e,
                    ecore=driver.ecore,
                    iprint=0,
                )
                ket = driver.get_random_mps(
                    tag="KET",
                    bond_dim=bond_dimension,
                    nroots=1,
                    occs=hf_occupations,
                    full_fci=True,
                )
                energy = driver.dmrg(
                    mpo,
                    ket,
                    n_sweeps=arguments.max_sweeps,
                    tol=arguments.tolerance,
                    bond_dims=[bond_dimension] * arguments.max_sweeps,
                    noises=noise_schedule,
                    thrds=[davidson_threshold] * arguments.max_sweeps,
                    cutoff=0.0,
                    iprint=arguments.verbose,
                )
                sweep_history = _serialize_sweep_history(driver)

                del ket
                del mpo
                del driver

            energy = float(np.asarray(energy).reshape(-1)[0])
            attempts.append(
                {
                    "restart": restart,
                    "seed": seed,
                    "energy_hartree": energy,
                    "sweeps": sweep_history,
                }
            )

        best_attempt = min(attempts, key=lambda attempt: attempt["energy_hartree"])
        energy = best_attempt["energy_hartree"]
        error = energy - exact_energy
        recovered_correlation = hartree_fock_energy - energy
        validity_tolerance = max(1e-8, 10.0 * arguments.tolerance)
        below_hf = energy <= hartree_fock_energy + validity_tolerance
        above_exact = energy >= exact_energy - validity_tolerance

        result = {
            "bond_dimension": bond_dimension,
            "nominal_qcmps_bond_qubits": _nominal_qcmps_bond_qubits(
                bond_dimension
            ),
            "energy_hartree": energy,
            "best_restart": best_attempt["restart"],
            "valid_variational_result": below_hf and above_exact,
            "below_or_equal_hf": below_hf,
            "above_or_equal_exact": above_exact,
            "error_from_exact_hartree": error,
            "error_from_exact_millihartree": 1000.0 * error,
            "correlation_energy_recovered_percent": (
                100.0 * recovered_correlation / correlation_energy
                if correlation_energy != 0.0
                else None
            ),
            "attempts": attempts,
        }
        output["results"].append(result)
        _write_json(output_path, output)

        print(f"[{system_name}] M={bond_dimension} energy: {energy:.16f} Eh")
        print(
            f"[{system_name}] M={bond_dimension} error:  "
            f"{1000.0 * error:.6f} mEh"
        )
        if not below_hf:
            print(
                f"[{system_name}] WARNING: M={bond_dimension} remained above "
                "the HF energy and is not a valid optimized DMRG reference."
            )
        if not above_exact:
            print(
                f"[{system_name}] WARNING: M={bond_dimension} fell below the "
                "exact reference beyond numerical tolerance."
            )

    print(f"\nWrote {output_path}")

