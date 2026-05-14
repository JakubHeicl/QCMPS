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
    
    
    #hamiltonian = permute_pauli_qubits(hamiltonian, [0, 4, 1, 5, 2, 6, 3, 7])
    hamiltonian = permute_pauli_qubits(hamiltonian, [0, 2, 4, 6, 1, 3, 5, 7])

    return hamiltonian, nuclear_repulsion, n_spin_orbitals, n_spatial_orbitals, num_alpha, num_beta, num_particles_total

def permute_pauli_qubits(hamiltonian: SparsePauliOp, permutation: list[int] | None = None) -> SparsePauliOp:
    
    n_qubits = hamiltonian.num_qubits

    if permutation is None:
        permutation = list(reversed(range(n_qubits)))

    if len(permutation) != n_qubits:
        raise ValueError(f"Permutation length must be {n_qubits}, got {len(permutation)}.")

    if sorted(permutation) != list(range(n_qubits)):
        raise ValueError("Permutation must contain each qubit index exactly once.")

    permuted_terms = []
    for label, coeff in hamiltonian.to_list():
        permuted = ["I"] * n_qubits
        for old_qubit, new_qubit in enumerate(permutation):
            permuted[n_qubits - 1 - new_qubit] = label[n_qubits - 1 - old_qubit]
        permuted_terms.append(("".join(permuted), coeff))

    return SparsePauliOp.from_list(permuted_terms).simplify()

def hf_occupancy(num_alpha: int, num_beta: int, n_spatial_orbitals: int, permutation: list[int] | None = None) -> list[int]:
    
    #permutation = [0, 4, 1, 5, 2, 6, 3, 7]
    permutation = [0, 2, 4, 6, 1, 3, 5, 7]

    alpha_occ = [1] * num_alpha + [0] * (n_spatial_orbitals - num_alpha)
    beta_occ = [1] * num_beta + [0] * (n_spatial_orbitals - num_beta)

    occupied = []
    for i, occ in enumerate(alpha_occ):
        if int(round(occ)) == 1:
            occupied.append(i)
    for i, occ in enumerate(beta_occ):
        if int(round(occ)) == 1:
            occupied.append(n_spatial_orbitals + i)
    if permutation is not None:
        occupied = [permutation[i] for i in occupied]
    return occupied

def load_initial_guess(file: Path) -> np.ndarray:
    print(f"[qcmps] Loading initial guess from {file}")
    return np.loadtxt(file).reshape(-1)
