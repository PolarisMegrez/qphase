"""Jacobian spectrum and stability postprocessor."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field
from qphase.backend.xputil import convert_to_numpy

from qphase_cam.core.jacobian import JacobianResolver
from qphase_cam.data import build_annotation_product
from qphase_cam.outcomes import CAMOutcomeStatus

from .base import CAMAnnotationContext, CAMPostprocessor, CAMPostprocessorConfig
from .frequency import _successful
from .state_view import annotation_status, field_column, state_rows


class JacobianSpectrumConfig(CAMPostprocessorConfig):
    allow_finite_difference: bool = False
    finite_difference_epsilon: float = Field(1e-7, gt=0.0)
    stability_tolerance: float = 1e-10


class JacobianSpectrum(CAMPostprocessor[JacobianSpectrumConfig]):
    name: ClassVar[str] = "jacobian_spectrum"
    description: ClassVar[str] = "Compute CAM Jacobian eigenvalues and stability"
    config_schema: ClassVar[type[JacobianSpectrumConfig]] = JacobianSpectrumConfig

    annotation_kind: ClassVar[str] = "jacobian_spectrum"
    accepted_product_roles: ClassVar[tuple[str, ...]] = ("state", "state_scan")
    requires_model: ClassVar[bool] = True
    requires_backend: ClassVar[bool] = True

    def process(self, product: Any, context: CAMAnnotationContext) -> Any:
        rows = state_rows(product)
        n_coordinates = int(context.model.n_modes) ** 2
        eigenvalues = field_column(rows, tail=(n_coordinates,), dtype=complex)
        sources = field_column(rows, dtype="U32")
        resolver = JacobianResolver(
            self.config.allow_finite_difference,
            self.config.finite_difference_epsilon,
        )
        for position in _successful(rows):
            row = rows[position]
            state = context.backend.asarray(row.state)
            jacobian = resolver.resolve(
                context.model, state, row.parameters, context.backend
            )
            eigenvalues[position] = np.linalg.eigvals(convert_to_numpy(jacobian))
            sources[position] = resolver.last_source
        used_sources = sorted({str(value) for value in sources if value})
        self.result_metadata = {"jacobian_sources": used_sources}
        if "finite_difference" in used_sources:
            self.result_metadata["finite_difference_epsilon"] = (
                self.config.finite_difference_epsilon
            )
        stable = np.all(np.real(eigenvalues) < self.config.stability_tolerance, axis=-1)
        statuses = annotation_status(
            rows,
            healthy=[bool(value) for value in stable],
            unhealthy=CAMOutcomeStatus.UNSTABLE,
        )
        return build_annotation_product(
            annotation_kind=self.annotation_kind,
            target_kind="state",
            target_identity=[row.identity for row in rows],
            status=statuses,
            fields={
                "jacobian_eigenvalues": eigenvalues,
                "jacobian_source": sources,
                "is_stable": stable,
            },
            field_axes={"jacobian_eigenvalues": ("jacobian_coordinate",)},
            axes={"jacobian_coordinate": n_coordinates},
            method_metadata=[dict(self.result_metadata) for _ in rows],
        )
