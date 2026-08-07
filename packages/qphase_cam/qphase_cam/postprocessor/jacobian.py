"""Jacobian spectrum and stability postprocessor."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field
from qphase.backend.xputil import convert_to_numpy

from qphase_cam.core.jacobian import JacobianResolver

from .base import CAMPostprocessor, CAMPostprocessorConfig
from .frequency import _successful_indices


class JacobianSpectrumConfig(CAMPostprocessorConfig):
    allow_finite_difference: bool = False
    finite_difference_epsilon: float = Field(1e-7, gt=0.0)
    stability_tolerance: float = 1e-10


class JacobianSpectrum(CAMPostprocessor[JacobianSpectrumConfig]):
    name: ClassVar[str] = "jacobian_spectrum"
    description: ClassVar[str] = "Compute CAM Jacobian eigenvalues and stability"
    config_schema: ClassVar[type[JacobianSpectrumConfig]] = JacobianSpectrumConfig

    def process(self, result: Any, model: Any, backend: Any) -> dict[str, Any]:
        n_coordinates = int(model.n_modes) ** 2
        shape = np.asarray(result.valid_mask).shape + (n_coordinates,)
        eigenvalues = np.full(shape, np.nan + 1j * np.nan)
        sources = np.full(np.asarray(result.valid_mask).shape, "", dtype=object)
        resolver = JacobianResolver(
            self.config.allow_finite_difference,
            self.config.finite_difference_epsilon,
        )
        for index in _successful_indices(result):
            state = backend.asarray(result.states[index])
            jacobian = resolver.resolve(model, state, result.params_at(index), backend)
            eigenvalues[index] = np.linalg.eigvals(convert_to_numpy(jacobian))
            sources[index] = resolver.last_source
        used_sources = sorted({str(value) for value in sources.flat if value})
        self.result_metadata = {"jacobian_sources": used_sources}
        if "finite_difference" in used_sources:
            self.result_metadata["finite_difference_epsilon"] = (
                self.config.finite_difference_epsilon
            )
        stable = np.all(
            np.real(eigenvalues) < self.config.stability_tolerance, axis=-1
        ) & np.asarray(result.success)
        return {
            "jacobian_eigenvalues": eigenvalues,
            "jacobian_source": sources,
            "is_stable": stable,
        }
