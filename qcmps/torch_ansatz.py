"""Differentiable PyTorch implementation of the global QCMPS ansatz.

The trainable blocks use the same parametrization as ``GUBlock``:

    U(theta) = exp(i * sum_j theta_j P_j)

where ``P_j`` are Pauli strings.  The all-identity string is omitted because it
only changes the global phase.  All operations in the forward pass remain in
PyTorch, so the energy can be optimized with ordinary PyTorch optimizers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


_PAULI_NUMPY = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    if real_dtype == torch.float32:
        return torch.complex64
    if real_dtype == torch.float64:
        return torch.complex128
    raise ValueError("Only torch.float32 and torch.float64 are supported.")


def pauli_basis(
    n_qubits: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    exclude_identity: bool = True,
) -> Tensor:
    """Return the Pauli-product basis in the order used by ``GUBlock``."""

    if n_qubits < 1:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}.")

    matrices: list[np.ndarray] = []
    for label in product("IXYZ", repeat=n_qubits):
        if exclude_identity and all(char == "I" for char in label):
            continue

        matrix = np.array([[1.0]], dtype=np.complex128)
        for char in label:
            matrix = np.kron(matrix, _PAULI_NUMPY[char])
        matrices.append(matrix)

    return torch.as_tensor(
        np.stack(matrices),
        dtype=_complex_dtype(dtype),
        device=device,
    )


def _apply_unitary(state: Tensor, unitary: Tensor, qubits: Sequence[int]) -> Tensor:
    """Apply a matrix using Qiskit's little-endian ordering for ``qubits``."""

    n_qubits = state.ndim
    qubits = tuple(qubits)
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"Qubit indices must be unique, got {qubits}.")
    if any(qubit < 0 or qubit >= n_qubits for qubit in qubits):
        raise ValueError(f"Qubits {qubits} are invalid for a {n_qubits}-qubit state.")

    dimension = 2 ** len(qubits)
    if unitary.shape != (dimension, dimension):
        raise ValueError(
            f"Expected a ({dimension}, {dimension}) matrix for {len(qubits)} qubits, "
            f"got {tuple(unitary.shape)}."
        )

    # A Qiskit matrix treats qubits[0] as its least-significant tensor factor.
    leading_axes = tuple(reversed(qubits))
    remaining_axes = tuple(axis for axis in range(n_qubits) if axis not in qubits)
    permutation = leading_axes + remaining_axes
    inverse_permutation = tuple(np.argsort(permutation).tolist())

    permuted = state.permute(permutation).reshape(dimension, -1)
    result = (unitary @ permuted).reshape((2,) * n_qubits)
    return result.permute(inverse_permutation)


def _swap_qubits(state: Tensor, first: int, second: int) -> Tensor:
    if first == second:
        return state
    return state.transpose(first, second)


class GeneralUnitaryBlocks(nn.Module):
    """One general unitary per QCMPS site and layer, optionally shared."""

    def __init__(
        self,
        n_sites: int,
        n_block_qubits: int,
        *,
        layers: int = 1,
        shared: bool = False,
        initial_scale: float = 1e-3,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if n_sites < 1:
            raise ValueError(f"n_sites must be positive, got {n_sites}.")
        if n_block_qubits < 1:
            raise ValueError(
                f"n_block_qubits must be positive, got {n_block_qubits}."
            )
        if layers < 1:
            raise ValueError(f"layers must be positive, got {layers}.")

        self.n_sites = n_sites
        self.n_block_qubits = n_block_qubits
        self.layers = layers
        self.shared = shared

        basis = pauli_basis(
            n_block_qubits,
            dtype=dtype,
            device=device,
            exclude_identity=True,
        )
        self.register_buffer("basis", basis, persistent=False)

        parameter_sites = 1 if shared else n_sites
        theta = initial_scale * torch.randn(
            parameter_sites,
            layers,
            basis.shape[0],
            dtype=dtype,
            device=device,
        )
        self.theta = nn.Parameter(theta)

    @property
    def parameters_per_layer(self) -> int:
        return 4**self.n_block_qubits - 1

    def forward(self) -> Tensor:
        """Return matrices with shape ``(n_sites, layers, d, d)``."""

        generators = torch.einsum(
            "slp,pij->slij", self.theta.to(self.basis.dtype), self.basis
        )
        unitaries = torch.matrix_exp(1j * generators)
        if self.shared:
            unitaries = unitaries.expand(self.n_sites, -1, -1, -1)
        return unitaries


class TorchQCMPS(nn.Module):
    """Global, differentiable statevector implementation of QCMPS."""

    def __init__(
        self,
        n_sites: int,
        n_bond_qubits: int,
        *,
        layers: int = 1,
        shared: bool = False,
        hf_occ: Sequence[int] | None = None,
        mps_unitaries: Sequence[np.ndarray | Tensor] | None = None,
        n_mps_bond_qubits: int = 0,
        initial_scale: float = 1e-3,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if n_bond_qubits < 0 or n_mps_bond_qubits < 0:
            raise ValueError("The numbers of bond qubits must be non-negative.")
        if hf_occ is not None and mps_unitaries is not None:
            raise ValueError("Use either hf_occ or mps_unitaries, not both.")

        self.n_sites = n_sites
        self.n_bond_qubits = n_bond_qubits
        self.n_mps_bond_qubits = n_mps_bond_qubits
        self.n_block_qubits = n_bond_qubits + n_mps_bond_qubits + 1
        self.n_total_qubits = n_sites + self.n_block_qubits - 1
        self.hf_occ = frozenset(hf_occ or ())
        self.real_dtype = dtype

        invalid_occupied = [site for site in self.hf_occ if site < 0 or site >= n_sites]
        if invalid_occupied:
            raise ValueError(f"Occupied sites outside [0, {n_sites}): {invalid_occupied}.")

        self.blocks = GeneralUnitaryBlocks(
            n_sites,
            self.n_block_qubits,
            layers=layers,
            shared=shared,
            initial_scale=initial_scale,
            dtype=dtype,
            device=device,
        )

        self._mps_buffer_names: list[str] = []
        if mps_unitaries is not None:
            if len(mps_unitaries) != n_sites:
                raise ValueError(
                    f"Expected {n_sites} MPS unitaries, got {len(mps_unitaries)}."
                )

            mps_dimension = 2 ** (n_mps_bond_qubits + 1)
            for site, matrix in enumerate(mps_unitaries):
                tensor = torch.as_tensor(
                    matrix,
                    dtype=_complex_dtype(dtype),
                    device=device,
                )
                if tensor.shape != (mps_dimension, mps_dimension):
                    raise ValueError(
                        f"MPS unitary {site} must have shape "
                        f"({mps_dimension}, {mps_dimension}), got {tuple(tensor.shape)}."
                    )
                name = f"mps_unitary_{site}"
                self.register_buffer(name, tensor)
                self._mps_buffer_names.append(name)

    @property
    def complex_dtype(self) -> torch.dtype:
        return _complex_dtype(self.blocks.theta.dtype)

    def _initial_state(self) -> Tensor:
        shape = (2,) * self.n_total_qubits
        state = torch.zeros(
            shape,
            dtype=self.complex_dtype,
            device=self.blocks.theta.device,
        )
        state[(0,) * self.n_total_qubits] = 1.0
        return state

    def forward(self) -> Tensor:
        """Build and return the normalized global state as a flat vector."""

        state = self._initial_state()
        unitaries = self.blocks()

        x_matrix = torch.as_tensor(
            _PAULI_NUMPY["X"],
            dtype=self.complex_dtype,
            device=state.device,
        )

        for site in range(self.n_sites):
            # This is the swap ladder in prepare_global_ansatz().
            for local_qubit in range(self.n_block_qubits - 1, 0, -1):
                state = _swap_qubits(
                    state,
                    site + local_qubit - 1,
                    site + local_qubit,
                )

            if self._mps_buffer_names:
                mps_unitary = getattr(self, self._mps_buffer_names[site])
                mps_qubits = range(site, site + self.n_mps_bond_qubits + 1)
                state = _apply_unitary(state, mps_unitary, mps_qubits)
            elif site in self.hf_occ:
                state = _apply_unitary(state, x_matrix, (site,))

            block_qubits = range(site, site + self.n_block_qubits)
            for layer in range(self.blocks.layers):
                state = _apply_unitary(
                    state,
                    unitaries[site, layer],
                    block_qubits,
                )

        return state.reshape(-1)


class SparsePauliHamiltonian(nn.Module):
    """A differentiable sum of Pauli strings in Qiskit label convention."""

    def __init__(
        self,
        labels: Sequence[str],
        coefficients: Sequence[complex] | np.ndarray | Tensor,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if len(labels) == 0:
            raise ValueError("The Hamiltonian must contain at least one Pauli term.")
        if any(len(label) != len(labels[0]) for label in labels):
            raise ValueError("All Pauli labels must have the same length.")
        if any(set(label) - set("IXYZ") for label in labels):
            raise ValueError("Pauli labels may only contain I, X, Y and Z.")

        coeffs = torch.as_tensor(
            coefficients,
            dtype=_complex_dtype(dtype),
            device=device,
        )
        if coeffs.shape != (len(labels),):
            raise ValueError(
                f"Expected {len(labels)} coefficients, got shape {tuple(coeffs.shape)}."
            )

        self.labels = tuple(labels)
        self.n_qubits = len(labels[0])
        self.register_buffer("coefficients", coeffs)
        self.register_buffer(
            "paulis",
            torch.as_tensor(
                np.stack([_PAULI_NUMPY[char] for char in "IXYZ"]),
                dtype=_complex_dtype(dtype),
                device=device,
            ),
            persistent=False,
        )
        self._pauli_indices = {char: index for index, char in enumerate("IXYZ")}

    @classmethod
    def from_sparse_pauli_op(
        cls,
        operator: Any,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> "SparsePauliHamiltonian":
        """Construct from a Qiskit ``SparsePauliOp`` without importing Qiskit."""

        terms = operator.to_list()
        return cls(
            [label for label, _ in terms],
            [coefficient for _, coefficient in terms],
            dtype=dtype,
            device=device,
        )

    def _apply_pauli_string(self, state: Tensor, label: str) -> Tensor:
        n_state_qubits = state.ndim
        if len(label) > n_state_qubits:
            raise ValueError(
                f"A {len(label)}-qubit operator cannot act on a "
                f"{n_state_qubits}-qubit state."
            )

        result = state
        # Rightmost Qiskit label character acts on logical qubit zero.
        for qubit, char in enumerate(reversed(label)):
            if char != "I":
                matrix = self.paulis[self._pauli_indices[char]]
                result = _apply_unitary(result, matrix, (qubit,))
        return result

    def forward(self, state: Tensor) -> Tensor:
        """Return the real Hamiltonian expectation value of a flat state."""

        expected_dimension = 2**self.n_qubits
        if state.ndim != 1:
            raise ValueError(f"Expected a flat statevector, got shape {tuple(state.shape)}.")
        if state.numel() < expected_dimension or state.numel() & (state.numel() - 1):
            raise ValueError(
                f"State dimension {state.numel()} is incompatible with the Hamiltonian."
            )

        n_state_qubits = state.numel().bit_length() - 1
        tensor_state = state.reshape((2,) * n_state_qubits)
        energy = torch.zeros((), dtype=state.dtype, device=state.device)

        for label, coefficient in zip(
            self.labels, self.coefficients, strict=True
        ):
            pauli_state = self._apply_pauli_string(tensor_state, label)
            expectation = torch.vdot(state, pauli_state.reshape(-1))
            energy = energy + coefficient * expectation

        return energy.real


def optimize_qcmps(
    model: TorchQCMPS,
    hamiltonian: SparsePauliHamiltonian,
    *,
    steps: int = 1000,
    learning_rate: float = 1e-2,
    callback: Callable[[int, float], None] | None = None,
    state_callback: Callable[[int, float, Tensor], None] | None = None,
) -> list[float]:
    """Optimize with Adam and return the energy history.

    ``state_callback`` receives a detached statevector matching the reported
    energy.  It can be used to evaluate observables that are not part of the
    training loss without performing another QCMPS forward pass.
    """

    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}.")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}.")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []

    for step in range(steps):
        optimizer.zero_grad()
        state = model()
        energy = hamiltonian(state)

        energy_value = float(energy.detach().cpu())
        history.append(energy_value)
        if callback is not None:
            callback(step, energy_value)
        if state_callback is not None:
            state_callback(step, energy_value, state.detach())

        energy.backward()
        optimizer.step()

    return history
