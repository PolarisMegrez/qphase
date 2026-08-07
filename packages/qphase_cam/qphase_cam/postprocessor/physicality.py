"""Physicality checks for CAM steady states."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field

from .base import CAMPostprocessor, CAMPostprocessorConfig
from .frequency import _successful_indices


class PhysicalityConfig(CAMPostprocessorConfig):
    hermitian_tolerance: float = Field(1e-10, gt=0.0)
    psd_tolerance: float = 1e-8
    residual_tolerance: float = Field(1e-7, gt=0.0)


class Physicality(CAMPostprocessor[PhysicalityConfig]):
    name: ClassVar[str] = "physicality"
    description: ClassVar[str] = "Check Hermitian and positive-semidefinite states"
    config_schema: ClassVar[type[PhysicalityConfig]] = PhysicalityConfig

    def process(self, result: Any, model: Any, backend: Any) -> dict[str, Any]:
        del model, backend
        shape = np.asarray(result.valid_mask).shape
        is_hermitian = np.zeros(shape, dtype=bool)
        is_psd = np.zeros(shape, dtype=bool)
        minimum_eigenvalue = np.full(shape, np.nan)
        residual_ok = np.zeros(shape, dtype=bool)
        for index in _successful_indices(result):
            state = np.asarray(result.states[index])
            is_hermitian[index] = np.allclose(
                state,
                state.conj().T,
                atol=self.config.hermitian_tolerance,
                rtol=0.0,
            )
            minimum_eigenvalue[index] = float(np.min(np.linalg.eigvalsh(state)))
            is_psd[index] = minimum_eigenvalue[index] >= -self.config.psd_tolerance
            residual_ok[index] = (
                float(np.asarray(result.residuals)[index])
                <= self.config.residual_tolerance
            )
        return {
            "is_hermitian": is_hermitian,
            "is_positive_semidefinite": is_psd,
            "minimum_state_eigenvalue": minimum_eigenvalue,
            "residual_within_tolerance": residual_ok,
            "is_physical": is_hermitian & is_psd & residual_ok,
        }
