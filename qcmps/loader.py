from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.formats import fcidump_to_problem
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit.quantum_info import SparsePauliOp

from pathlib import Path

import numpy as np

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

def load_initial_guess(file: Path) -> np.ndarray:
    print(f"[qcmps] Loading initial guess from {file}")
    return np.loadtxt(file).reshape(-1)