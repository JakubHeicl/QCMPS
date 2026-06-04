from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.formats import fcidump_to_problem
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.operators.symmetric_two_body import fold, unfold
from qiskit.quantum_info import SparsePauliOp

from pathlib import Path

import numpy as np

def spin_block_to_interleaved_permutation(n_spatial_orbitals: int) -> list[int]:
    return [
        index
        for spatial in range(n_spatial_orbitals)
        for index in (spatial, n_spatial_orbitals + spatial)
    ]

def _looks_like_interleaved_spin_orbital_fcidump(fcidump: FCIDump, tol: float = 1e-10) -> bool:

    n_orbitals = fcidump.num_orbitals
    if n_orbitals % 2 != 0:
        return False

    if fcidump.hij_b is not None or fcidump.hijkl_bb is not None or fcidump.hijkl_ba is not None:
        return False

    h1 = fcidump.hij
    if h1 is None:
        return False

    alpha_h1 = h1[0::2, 0::2]
    beta_h1 = h1[1::2, 1::2]
    cross_h1 = max(
        np.max(np.abs(h1[0::2, 1::2])),
        np.max(np.abs(h1[1::2, 0::2])),
    )

    if np.max(np.abs(alpha_h1 - beta_h1)) > tol or cross_h1 > tol:
        return False

    h2 = unfold(fcidump.hijkl).array
    alpha_h2 = h2[0::2, 0::2, 0::2, 0::2]
    beta_h2 = h2[1::2, 1::2, 1::2, 1::2]
    mixed_h2 = h2[0::2, 0::2, 1::2, 1::2]

    return (
        np.max(np.abs(alpha_h2 - beta_h2)) <= tol
        and np.max(np.abs(alpha_h2 - mixed_h2)) <= tol
    )

def _collapse_interleaved_spin_orbital_fcidump(fcidump: FCIDump) -> FCIDump:

    h1 = fcidump.hij[0::2, 0::2]
    h2 = unfold(fcidump.hijkl).array[0::2, 0::2, 0::2, 0::2]

    orbsym = fcidump.orbsym[0::2] if fcidump.orbsym is not None else None

    return FCIDump(
        num_electrons=fcidump.num_electrons,
        hij=h1,
        hijkl=fold(h2),
        constant_energy=fcidump.constant_energy,
        multiplicity=fcidump.multiplicity,
        orbsym=orbsym,
        isym=fcidump.isym,
    )

def build_from_fcidump(file_path: Path) -> tuple[SparsePauliOp, float, int, int, int, int, int]:

    print(f"[qcmps] Loading FCIDUMP from {file_path}")
    fcidump = FCIDump.from_file(file_path)

    if _looks_like_interleaved_spin_orbital_fcidump(fcidump):
        print(
            "[qcmps] Detected interleaved spin-orbital FCIDUMP; "
            "collapsing to spatial-orbital integrals"
        )
        fcidump = _collapse_interleaved_spin_orbital_fcidump(fcidump)

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
    
    
    hamiltonian = permute_pauli_qubits(
        hamiltonian,
        spin_block_to_interleaved_permutation(n_spatial_orbitals),
    )

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

def truncate_pauli_terms(hamiltonian: SparsePauliOp, cutoff: float = 0.0) -> SparsePauliOp:

    if cutoff <= 0.0:
        return hamiltonian

    kept_terms = [(label, coeff) for label, coeff in hamiltonian.to_list() if abs(coeff) >= cutoff]

    if len(kept_terms) == 0:
        raise ValueError(f"Hamiltonian cutoff {cutoff} removed all Pauli terms.")

    truncated = SparsePauliOp.from_list(kept_terms).simplify()
    print(
        "[qcmps] Hamiltonian cutoff: "
        f"kept {len(truncated)}/{len(hamiltonian)} Pauli terms "
        f"(abs(coeff) >= {cutoff})"
    )

    return truncated

def _z_label(n_qubits: int, qubits: tuple[int, ...]) -> str:

    label = ["I"] * n_qubits
    for qubit in qubits:
        label[n_qubits - 1 - qubit] = "Z"
    return "".join(label)

def particle_number_penalty_operator(
    n_qubits: int,
    n_electrons: int,
    penalty_strength: float = 1.0,
) -> SparsePauliOp:

    if not 0 <= n_electrons <= n_qubits:
        raise ValueError(f"Expected 0 <= n_electrons <= {n_qubits}, got {n_electrons}.")
    if penalty_strength < 0.0:
        raise ValueError(f"Penalty strength must be non-negative, got {penalty_strength}.")

    identity = "I" * n_qubits
    terms = [(identity, float(n_electrons**2))]

    single_coeff = 1.0 - 2.0 * n_electrons
    for q in range(n_qubits):
        terms.append((identity, 0.5 * single_coeff))
        terms.append((_z_label(n_qubits, (q,)), -0.5 * single_coeff))

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            terms.append((identity, 0.5))
            terms.append((_z_label(n_qubits, (i,)), -0.5))
            terms.append((_z_label(n_qubits, (j,)), -0.5))
            terms.append((_z_label(n_qubits, (i, j)), 0.5))

    return (SparsePauliOp.from_list(terms).simplify() * penalty_strength).simplify()

def add_particle_number_penalty(
    hamiltonian: SparsePauliOp,
    n_electrons: int,
    penalty_strength: float = 1.0,
) -> SparsePauliOp:

    penalty = particle_number_penalty_operator(
        hamiltonian.num_qubits,
        n_electrons,
        penalty_strength,
    )

    return (hamiltonian + penalty).simplify()

def hf_occupancy(num_alpha: int, num_beta: int, n_spatial_orbitals: int, permutation: list[int] | None = None) -> list[int]:
    
    if permutation is None:
        permutation = spin_block_to_interleaved_permutation(n_spatial_orbitals)

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
