"""Physicality checks for CAM steady states."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field

from .base import CAMAnnotationContext, CAMPostprocessor, CAMPostprocessorConfig
from .frequency import _annotation, _successful
from .state_view import field_column, state_rows


class PhysicalityConfig(CAMPostprocessorConfig):
    hermitian_tolerance: float = Field(1e-10, gt=0.0)
    psd_tolerance: float = 1e-8
    residual_tolerance: float = Field(1e-7, gt=0.0)


class Physicality(CAMPostprocessor[PhysicalityConfig]):
    name: ClassVar[str] = "physicality"
    description: ClassVar[str] = "Check Hermitian and positive-semidefinite states"
    config_schema: ClassVar[type[PhysicalityConfig]] = PhysicalityConfig

    annotation_kind: ClassVar[str] = "physicality"
    accepted_product_roles: ClassVar[tuple[str, ...]] = ("state", "state_scan")
    requires_model: ClassVar[bool] = False
    requires_backend: ClassVar[bool] = False

    def process(self, product: Any, context: CAMAnnotationContext) -> Any:
        del context
        rows = state_rows(product)
        is_hermitian = field_column(rows, dtype=bool)
        is_psd = field_column(rows, dtype=bool)
        minimum_eigenvalue = field_column(rows, dtype=float)
        residual_ok = field_column(rows, dtype=bool)
        for position in _successful(rows):
            row = rows[position]
            is_hermitian[position] = np.allclose(
                row.state,
                row.state.conj().T,
                atol=self.config.hermitian_tolerance,
                rtol=0.0,
            )
            minimum_eigenvalue[position] = float(np.min(np.linalg.eigvalsh(row.state)))
            is_psd[position] = (
                minimum_eigenvalue[position] >= -self.config.psd_tolerance
            )
            residual_ok[position] = row.residual <= self.config.residual_tolerance
        is_physical = is_hermitian & is_psd & residual_ok
        return _annotation(
            self,
            rows,
            {
                "is_hermitian": is_hermitian,
                "is_positive_semidefinite": is_psd,
                "minimum_state_eigenvalue": minimum_eigenvalue,
                "residual_within_tolerance": residual_ok,
                "is_physical": is_physical,
            },
            healthy=[bool(value) for value in is_physical],
        )
