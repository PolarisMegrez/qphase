"""Rayleigh and Hamiltonian-spectrum CAM postprocessors."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from .base import CAMPostprocessor, CAMPostprocessorConfig


class FrequencyConfig(CAMPostprocessorConfig):
    pass


def _successful_indices(result: Any) -> list[tuple[int, ...]]:
    mask = np.asarray(result.valid_mask) & np.asarray(result.success)
    return [tuple(index) for index in np.argwhere(mask)]


class RayleighFrequency(CAMPostprocessor[FrequencyConfig]):
    name: ClassVar[str] = "rayleigh_frequency"
    description: ClassVar[str] = "Compute Re Tr(HR) / Tr(R)"
    config_schema: ClassVar[type[FrequencyConfig]] = FrequencyConfig

    def process(self, result: Any, model: Any, backend: Any) -> dict[str, Any]:
        del backend
        output = np.full(np.asarray(result.valid_mask).shape, np.nan)
        for index in _successful_indices(result):
            state = np.asarray(result.states[index])
            hamiltonian = np.asarray(
                model.cam_hamiltonian(state, result.params_at(index))
            )
            denominator = np.trace(state)
            if abs(denominator) >= 1e-300:
                output[index] = float(
                    np.real(np.trace(hamiltonian @ state) / denominator)
                )
        return {"rayleigh_frequency": output}


class HamiltonianSpectrum(CAMPostprocessor[FrequencyConfig]):
    name: ClassVar[str] = "hamiltonian_spectrum"
    description: ClassVar[str] = "Compute all eigenvalues of H(R)"
    config_schema: ClassVar[type[FrequencyConfig]] = FrequencyConfig

    def process(self, result: Any, model: Any, backend: Any) -> dict[str, Any]:
        del backend
        shape = np.asarray(result.valid_mask).shape + (int(model.n_modes),)
        eigenvalues = np.full(shape, np.nan + 1j * np.nan)
        for index in _successful_indices(result):
            state = np.asarray(result.states[index])
            hamiltonian = np.asarray(
                model.cam_hamiltonian(state, result.params_at(index))
            )
            eigenvalues[index] = np.linalg.eigvals(hamiltonian)
        return {
            "hamiltonian_eigenvalues": eigenvalues,
            "mode_frequencies": np.real(eigenvalues),
        }
