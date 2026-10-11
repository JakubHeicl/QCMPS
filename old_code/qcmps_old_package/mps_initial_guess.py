from __future__ import annotations

import numpy as np


def _orthonormalize_columns(columns: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    q_columns = []

    for col in range(columns.shape[1]):
        vector = columns[:, col].astype(complex, copy=True)

        for q_col in q_columns:
            vector -= q_col * np.vdot(q_col, vector)
        for q_col in q_columns:
            vector -= q_col * np.vdot(q_col, vector)

        norm = np.linalg.norm(vector)
        if norm <= tol:
            raise ValueError("MPS tensor contains linearly dependent columns.")

        q_columns.append(vector / norm)

    return np.column_stack(q_columns)


def _complete_orthonormal_columns(columns: np.ndarray, total_dim: int, tol: float = 1e-10) -> np.ndarray:
    q_columns = [columns[:, col].astype(complex, copy=True) for col in range(columns.shape[1])]
    basis = np.eye(total_dim, dtype=complex)

    for basis_col in range(total_dim):
        vector = basis[:, basis_col].copy()

        for q_col in q_columns:
            vector -= q_col * np.vdot(q_col, vector)
        for q_col in q_columns:
            vector -= q_col * np.vdot(q_col, vector)

        norm = np.linalg.norm(vector)
        if norm <= tol:
            continue

        q_columns.append(vector / norm)

        if len(q_columns) == total_dim:
            return np.column_stack(q_columns)

    raise ValueError("Could not complete MPS isometry to a unitary matrix.")


def _mps_tensor_to_unitary(
    physical_slices: tuple[np.ndarray, np.ndarray],
    n_bond_qubits: int,
) -> np.ndarray:
    max_bond_dim = 2**n_bond_qubits
    total_dim = 2 * max_bond_dim

    tensors = tuple(np.asarray(tensor, dtype=complex) for tensor in physical_slices)
    if tensors[0].shape != tensors[1].shape:
        raise ValueError(
            f"Expected matching physical MPS slice shapes, got {tensors[0].shape} and {tensors[1].shape}."
        )

    left_dim, right_dim = tensors[0].shape
    if left_dim > max_bond_dim or right_dim > max_bond_dim:
        raise ValueError(
            f"MPS tensor shape {tensors[0].shape} does not fit in {n_bond_qubits} bond qubits."
        )

    active_columns = np.zeros((total_dim, right_dim), dtype=complex)
    for physical, tensor in enumerate(tensors):
        rows = (np.arange(left_dim) << 1) | physical
        active_columns[rows, :] = tensor

    active_columns = _orthonormalize_columns(active_columns)
    completed_columns = _complete_orthonormal_columns(active_columns, total_dim)

    unitary = np.zeros((total_dim, total_dim), dtype=complex)
    active_input_columns = [right << 1 for right in range(right_dim)]
    inactive_input_columns = [
        column for column in range(total_dim) if column not in active_input_columns
    ]

    for source, target in enumerate(active_input_columns):
        unitary[:, target] = completed_columns[:, source]
    for source, target in enumerate(inactive_input_columns, start=right_dim):
        unitary[:, target] = completed_columns[:, source]

    return unitary


def h4_stretched_bd2_unitaries() -> list[np.ndarray]:
    return [
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-0.8578385621186476, 0.0, 0.5139192556639721, 0.0],
                [-0.5139192556639723, 0.0, -0.8578385621186478, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=complex,
        ),
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=complex,
        ),
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=complex,
        ),
        np.array(
            [
                [-0.9999988136571801, 0.001540351983175043, 0.0, 0.0],
                [0.0, 0.0, 0.9999907899578676, 0.004291852681508773],
                [-0.001540351983330135, -0.9999988136571804, 0.0, 0.0],
                [0.0, 0.0, -0.004291852681527857, 0.9999907899578676],
            ],
            dtype=complex,
        ),
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.8578398980357089, 0.0, -0.513917025732836, 0.0],
                [0.513917025732836, 0.0, 0.857839898035709, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=complex,
        ),
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=complex,
        ),
        np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=complex,
        ),
        np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=complex,
        ),
    ]


def h6_stretched_bd4_tensors() -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (
            np.array(
                [
                    [1.0, 0.0],
                ],
                dtype=complex,
            ),
            np.array(
                [
                    [0.0, 1.0],
                ],
                dtype=complex,
            ),
        ),
        (
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.40583e-06, -1.0, 0.0],
                ],
                dtype=complex,
            ),
            np.array(
                [
                    [0.0, -1.0, 1.40583e-06, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=complex,
            ),
        ),
        (
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.09541e-06, -6.41874e-07, 0.0, 0.0],
                    [0.861737, -0.507355, 0.0, 0.0],
                    [0.0, 0.0, -0.945996, 0.0],
                ],
                dtype=complex,
            ),
            np.array(
                [
                    [0.507355, 0.861737, 0.0, 0.0],
                    [0.0, 0.0, -0.324178, 0.0],
                    [0.0, 0.0, 4.55773e-07, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=complex,
            ),
        ),
        (
            np.array(
                [
                    [-0.526945, -0.849899, 0.0, 0.0],
                    [-0.849899, 0.526945, 0.0, 0.0],
                    [0.0, 0.0, 0.834324, 0.0],
                    [0.0, 0.0, 0.0, -1.0],
                ],
                dtype=complex,
            ),
            np.array(
                [
                    [0.0, 0.0, 0.00316414, 0.0],
                    [0.0, 0.0, 0.551265, 0.0],
                    [0.0, 0.0, 0.0, -4.51759e-08],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype=complex,
            ),
        ),
        (
            np.array(
                [
                    [0.634597, 0.0, 0.0, 0.0],
                    [0.772843, 0.0, 0.0, 0.0],
                    [0.0, -5.27936e-09, -1.0, 0.0],
                    [0.0, 0.0, 0.0, -0.081383],
                ],
                dtype=complex,
            ),
            np.array(
                [
                    [0.0, -0.119331, 2.06114e-08, 0.0],
                    [0.0, 0.992855, -2.84009e-09, 0.0],
                    [0.0, 0.0, 0.0, 0.996683],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype=complex,
            ),
        ),
        (
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [-2.1617e-08, 0.0, 0.0, 0.0],
                    [0.997573, 0.0, 0.0, 0.0],
                    [0.0, 5.84059e-06, 0.985488, 0.0],
                ],
                dtype=complex,
            ),
            np.array(
                [
                    [0.0696232, 0.0, 0.0, 0.0],
                    [0.0, 1.07261e-06, 0.169747, 0.0],
                    [0.0, -1.0, 5.9379e-06, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=complex,
            ),
        ),
        (
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [2.0787e-06, 0.0, 0.0, 0.0],
                    [0.632999, 0.0, 0.0, 0.0],
                    [0.0, 2.91987e-07, 0.907587, 0.0],
                ],
                dtype=complex,
            ),
            np.array(
                [
                    [0.774153, 0.0, 0.0, 0.0],
                    [0.0, -4.27986e-06, 0.419863, 0.0],
                    [0.0, -1.0, -1.53195e-06, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=complex,
            ),
        ),
        (
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 6.24836e-09, 0.0],
                    [0.0, 1.3834e-07, 0.827351, 0.0],
                    [0.0, 0.0, 0.0, -0.89219],
                ],
                dtype=complex,
            ),
            np.array(
                [
                    [0.0, -2.14896e-07, 0.561685, 0.0],
                    [0.0, 0.0, 0.0, -0.45166],
                    [0.0, 0.0, 0.0, 4.20154e-06],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype=complex,
            ),
        ),
        (
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [-0.784754, 7.06528e-08, 0.0, 0.0],
                    [-1.55869e-07, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -0.997931, -0.0642871],
                ],
                dtype=complex,
            ),
            np.array(
                [
                    [0.619808, -1.62024e-07, 0.0, 0.0],
                    [0.0, 0.0, 2.49934e-07, 1.7147e-08],
                    [0.0, 0.0, -0.0642871, 0.997931],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype=complex,
            ),
        ),
    ]


def h6_stretched_bd4_unitaries() -> list[np.ndarray]:
    return [
        _mps_tensor_to_unitary(tensor, n_bond_qubits=2)
        for tensor in h6_stretched_bd4_tensors()
    ]


MPS_INITIAL_GUESSES = {
    "h4_stretched_bd2": (h4_stretched_bd2_unitaries, 1),
    "h6_stretched_bd4": (h6_stretched_bd4_unitaries, 2),
}


def available_mps_initial_guesses() -> tuple[str, ...]:
    return tuple(MPS_INITIAL_GUESSES)


def load_mps_initial_guess(name: str) -> tuple[list[np.ndarray], int]:
    try:
        unitary_factory, n_mps_bond_qubits = MPS_INITIAL_GUESSES[name]
    except KeyError as exc:
        choices = ", ".join(available_mps_initial_guesses())
        raise ValueError(f"Unknown MPS initial guess '{name}'. Available choices: {choices}") from exc

    return unitary_factory(), n_mps_bond_qubits
