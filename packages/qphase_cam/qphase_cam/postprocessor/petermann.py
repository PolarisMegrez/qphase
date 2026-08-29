"""Petermann factors and linearized fluctuation spectra for CAM solutions."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field
from qphase.backend.xputil import convert_to_numpy
from scipy.linalg import eig

from qphase_cam.model import CAMBogoliubovModel

from .base import CAMPostprocessor, CAMPostprocessorConfig


class PetermannSpectrumConfig(CAMPostprocessorConfig):
    """Configuration for chunked left/right eigensystem evaluation."""

    chunk_size: int = Field(default=20_000, ge=1)


def build_bogoliubov_matrix(
    hamiltonian: np.ndarray, interaction: np.ndarray
) -> np.ndarray:
    """Build ``[[H+V, V], [-V*, -H*-V*]]`` without transposing starred blocks."""
    hamiltonian, interaction = _validate_blocks(hamiltonian, interaction)
    n_modes = hamiltonian.shape[-1]
    output = np.empty(
        hamiltonian.shape[:-2] + (2 * n_modes, 2 * n_modes),
        dtype=np.complex128,
    )
    output[..., :n_modes, :n_modes] = hamiltonian + interaction
    output[..., :n_modes, n_modes:] = interaction
    output[..., n_modes:, :n_modes] = -np.conjugate(interaction)
    output[..., n_modes:, n_modes:] = -np.conjugate(hamiltonian) - np.conjugate(
        interaction
    )
    return output


def build_monodromy_matrix(
    hamiltonian: np.ndarray,
    interaction: np.ndarray,
    rayleigh_frequency: np.ndarray,
) -> np.ndarray:
    """Build the Rayleigh-frequency rotating-frame monodromy matrix."""
    hamiltonian, interaction = _validate_blocks(hamiltonian, interaction)
    frequency = np.asarray(rayleigh_frequency)
    if frequency.shape != hamiltonian.shape[:-2]:
        raise ValueError("rayleigh_frequency must match the matrix stack shape")
    n_modes = hamiltonian.shape[-1]
    shift = frequency[..., None, None] * np.eye(n_modes, dtype=np.complex128)
    output = build_bogoliubov_matrix(hamiltonian, interaction)
    output[..., :n_modes, :n_modes] -= shift
    output[..., n_modes:, n_modes:] += shift
    return output


def eigensystem_petermann(
    matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lexicographically sorted eigenvalues and aligned Petermann factors."""
    matrices = np.asarray(matrices)
    if matrices.ndim < 2 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("matrices must have shape (..., n, n)")
    matrix_size = matrices.shape[-1]
    flat = matrices.reshape(-1, matrix_size, matrix_size)
    values = np.empty((flat.shape[0], matrix_size), dtype=np.complex128)
    factors = np.empty((flat.shape[0], matrix_size), dtype=float)

    for index, matrix in enumerate(flat):
        eigenvalues, left, right = eig(
            matrix, left=True, right=True, check_finite=False
        )
        left_norm = np.sum(np.abs(left) ** 2, axis=0)
        right_norm = np.sum(np.abs(right) ** 2, axis=0)
        overlap = np.sum(np.conjugate(left) * right, axis=0)
        denominator = np.abs(overlap) ** 2
        petermann = np.full(denominator.shape, np.inf, dtype=float)
        np.divide(
            left_norm * right_norm,
            denominator,
            out=petermann,
            where=denominator > 0.0,
        )
        order = np.lexsort((np.imag(eigenvalues), np.real(eigenvalues)))
        values[index] = eigenvalues[order]
        factors[index] = petermann[order]

    shape = matrices.shape[:-2] + (matrix_size,)
    return values.reshape(shape), factors.reshape(shape)


class PetermannSpectrum(CAMPostprocessor[PetermannSpectrumConfig]):
    """Compute H, Bogoliubov, and monodromy spectra for converged CAM states."""

    name: ClassVar[str] = "petermann_spectrum"
    description: ClassVar[str] = (
        "Compute Hamiltonian, Bogoliubov, and monodromy Petermann spectra"
    )
    config_schema: ClassVar[type[PetermannSpectrumConfig]] = PetermannSpectrumConfig

    def process(self, result: Any, model: Any, backend: Any) -> dict[str, Any]:
        del backend
        if not isinstance(model, CAMBogoliubovModel):
            raise TypeError(
                f"model {model.name!r} does not provide cam_bogoliubov_interaction"
            )

        mask = np.asarray(convert_to_numpy(result.valid_mask), dtype=bool) & np.asarray(
            convert_to_numpy(result.success), dtype=bool
        )
        indices = [tuple(int(value) for value in index) for index in np.argwhere(mask)]
        result_shape = mask.shape
        n_modes = int(model.n_modes)
        h_values = _complex_output(result_shape + (n_modes,))
        h_factors = _real_output(result_shape + (n_modes,))
        doubled_shape = result_shape + (2 * n_modes,)
        bogoliubov_values = _complex_output(doubled_shape)
        bogoliubov_factors = _real_output(doubled_shape)
        monodromy_values = _complex_output(doubled_shape)
        monodromy_factors = _real_output(doubled_shape)
        states = np.asarray(convert_to_numpy(result.states))

        for start in range(0, len(indices), self.config.chunk_size):
            chunk_indices = indices[start : start + self.config.chunk_size]
            hamiltonians = []
            interactions = []
            frequencies = []
            for index in chunk_indices:
                state = states[index]
                params = result.params_at(index)
                hamiltonian = np.asarray(
                    convert_to_numpy(model.cam_hamiltonian(state, params)),
                    dtype=np.complex128,
                )
                interaction = np.asarray(
                    convert_to_numpy(model.cam_bogoliubov_interaction(state, params)),
                    dtype=np.complex128,
                )
                hamiltonians.append(hamiltonian)
                interactions.append(interaction)
                trace = np.trace(state)
                frequencies.append(
                    np.nan
                    if abs(trace) < 1e-300
                    else float(np.real(np.trace(hamiltonian @ state) / trace))
                )

            h_stack = np.asarray(hamiltonians)
            v_stack = np.asarray(interactions)
            frequency_stack = np.asarray(frequencies)
            h_chunk, h_factor_chunk = eigensystem_petermann(h_stack)
            b_chunk, b_factor_chunk = eigensystem_petermann(
                build_bogoliubov_matrix(h_stack, v_stack)
            )
            m_chunk, m_factor_chunk = eigensystem_petermann(
                build_monodromy_matrix(h_stack, v_stack, frequency_stack)
            )
            for offset, index in enumerate(chunk_indices):
                h_values[index] = h_chunk[offset]
                h_factors[index] = h_factor_chunk[offset]
                bogoliubov_values[index] = b_chunk[offset]
                bogoliubov_factors[index] = b_factor_chunk[offset]
                monodromy_values[index] = m_chunk[offset]
                monodromy_factors[index] = m_factor_chunk[offset]

        self.result_metadata = {
            "interaction_capability": "cam_bogoliubov_interaction",
            "eigenvalue_order": "real_then_imag",
            "monodromy_frame": "rayleigh_frequency",
            "star_operation": "elementwise_conjugation",
        }
        return {
            "hamiltonian_eigenvalues": h_values,
            "mode_frequencies": np.real(h_values),
            "hamiltonian_petermann_factors": h_factors,
            "bogoliubov_eigenvalues": bogoliubov_values,
            "bogoliubov_petermann_factors": bogoliubov_factors,
            "monodromy_eigenvalues": monodromy_values,
            "monodromy_petermann_factors": monodromy_factors,
        }


def _validate_blocks(
    hamiltonian: np.ndarray, interaction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    hamiltonian = np.asarray(hamiltonian)
    interaction = np.asarray(interaction)
    if hamiltonian.shape != interaction.shape:
        raise ValueError("hamiltonian and interaction must have equal shape")
    if hamiltonian.ndim < 2 or hamiltonian.shape[-1] != hamiltonian.shape[-2]:
        raise ValueError("matrix blocks must have shape (..., n, n)")
    return hamiltonian, interaction


def _complex_output(shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, np.nan + 1j * np.nan, dtype=np.complex128)


def _real_output(shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, np.nan, dtype=float)
