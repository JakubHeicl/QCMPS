import json
from pathlib import Path

import numpy as np

from experiments._common import H4_FCIDUMP_PATH
from qcmps import (
    Ansatz,
    FCIDumpLoader,
    GlobalQCMPS,
    HFPreset,
    Logger,
    Observable,
    SO4Gate,
    StarLayerType,
    TwoSiteOptimizer,
)


RUN_A = 1
RUN_B = 5


def load_qcmps_parameters(path: Path, n_phys_qubits: int) -> np.ndarray:
    with path.open(encoding="utf-8") as file:
        checkpoint = json.load(file)

    qcmps_data = next(
        component
        for component in checkpoint["components"]
        if component["type"] == "GlobalQCMPS"
    )

    parameters = np.asarray(qcmps_data["parameters_values"], dtype=float)
    return parameters.reshape(n_phys_qubits, SO4Gate.n_parameters())


def main():
    output_dir = Path(__file__).parent / "output"

    loader = FCIDumpLoader(H4_FCIDUMP_PATH)
    loader.interleave_spin_orbitals()

    n_phys_qubits = loader.n_spin_orbitals
    n_bond_qubits = 2
    n_qubits = n_phys_qubits + n_bond_qubits
    physical_qubits = list(range(n_phys_qubits))

    parameters_a = load_qcmps_parameters(output_dir / f"h4_{RUN_A}_check.json", n_phys_qubits)
    parameters_b = load_qcmps_parameters(output_dir / f"h4_{RUN_B}_check.json", n_phys_qubits)

    star_parameters = np.concatenate((parameters_a, parameters_b), axis=1).reshape(-1)

    ansatz = Ansatz(n_qubits)
    hf_preset = HFPreset(n_phys_qubits, loader.hf_occupancy())
    
    qcmps = GlobalQCMPS(n_phys_qubits, n_bond_qubits, [StarLayerType(SO4Gate)])
    qcmps.update_parameters(star_parameters)

    ansatz.add_component(hf_preset, list(range(n_bond_qubits, n_qubits)))
    ansatz.add_component(qcmps, list(range(n_qubits)))

    hamiltonian = Observable(loader.hamiltonian, physical_qubits, constant=loader.nuclear_repulsion, label="Total energy")
    particle_number = Observable.particle_number(physical_qubits)

    logger = Logger(output_dir / f"h4_star_{RUN_A}_{RUN_B}.log", value_file=None, continue_logging=False)

    logger.log(f"Star initialization from runs {RUN_A} and {RUN_B}.")
    logger.log(f"Initial total energy: {hamiltonian(ansatz):.12f}")
    logger.log(f"Initial energy variance: {hamiltonian.var(ansatz):.12f}")
    logger.log(f"Initial particle number: {particle_number(ansatz):.12f}")
    logger.log(f"Initial particle number variance: {particle_number.var(ansatz):.12f}")

    optimizer = TwoSiteOptimizer(
        logger,
        log_every_n_steps=None,
        max_sweeps=50,
        local_maxiter=10,
        tolerance=1e-8,
    )
    
    optimizer.optimize(ansatz, hamiltonian, qcmps)

    logger.log(f"Final total energy: {hamiltonian(ansatz):.12f}")
    logger.log(f"Final energy variance: {hamiltonian.var(ansatz):.12f}")
    logger.log(f"Final particle number: {particle_number(ansatz):.12f}")
    logger.log(f"Final particle number variance: {particle_number.var(ansatz):.12f}")

    ansatz.save_checkpoint(output_dir / f"h4_star_{RUN_A}_{RUN_B}_check.json")


if __name__ == "__main__":
    main()
