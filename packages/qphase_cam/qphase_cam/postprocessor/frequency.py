"""Rayleigh and Hamiltonian-spectrum CAM postprocessors."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from qphase_cam.data import build_annotation_product

from .base import CAMAnnotationContext, CAMPostprocessor, CAMPostprocessorConfig
from .state_view import annotation_status, field_column, state_rows


class FrequencyConfig(CAMPostprocessorConfig):
    pass


def _successful(rows: list[Any]) -> list[int]:
    return [position for position, row in enumerate(rows) if row.success]


def _annotation(
    processor: Any,
    rows: list[Any],
    fields: dict[str, Any],
    *,
    field_axes: dict[str, Any] | None = None,
    axes: dict[str, int] | None = None,
    healthy: list[bool] | None = None,
) -> Any:
    return build_annotation_product(
        annotation_kind=processor.annotation_kind,
        target_kind="state",
        target_identity=[row.identity for row in rows],
        status=annotation_status(rows, healthy=healthy),
        fields=fields,
        field_axes=field_axes,
        axes=axes,
        method_metadata=[dict(processor.result_metadata) for _ in rows],
    )


class RayleighFrequency(CAMPostprocessor[FrequencyConfig]):
    name: ClassVar[str] = "rayleigh_frequency"
    description: ClassVar[str] = "Compute Re Tr(HR) / Tr(R)"
    config_schema: ClassVar[type[FrequencyConfig]] = FrequencyConfig

    annotation_kind: ClassVar[str] = "rayleigh_frequency"
    accepted_product_roles: ClassVar[tuple[str, ...]] = ("state", "state_scan")
    requires_model: ClassVar[bool] = True
    requires_backend: ClassVar[bool] = False

    def process(self, product: Any, context: CAMAnnotationContext) -> Any:
        rows = state_rows(product)
        output = field_column(rows, dtype=float)
        for position in _successful(rows):
            row = rows[position]
            hamiltonian = np.asarray(
                context.model.cam_hamiltonian(row.state, row.parameters)
            )
            denominator = np.trace(row.state)
            if abs(denominator) >= 1e-300:
                output[position] = float(
                    np.real(np.trace(hamiltonian @ row.state) / denominator)
                )
        return _annotation(self, rows, {"rayleigh_frequency": output})


class HamiltonianSpectrum(CAMPostprocessor[FrequencyConfig]):
    name: ClassVar[str] = "hamiltonian_spectrum"
    description: ClassVar[str] = "Compute all eigenvalues of H(R)"
    config_schema: ClassVar[type[FrequencyConfig]] = FrequencyConfig

    annotation_kind: ClassVar[str] = "hamiltonian_spectrum"
    accepted_product_roles: ClassVar[tuple[str, ...]] = ("state", "state_scan")
    requires_model: ClassVar[bool] = True
    requires_backend: ClassVar[bool] = False

    def process(self, product: Any, context: CAMAnnotationContext) -> Any:
        rows = state_rows(product)
        n_modes = int(context.model.n_modes)
        eigenvalues = field_column(rows, tail=(n_modes,), dtype=complex)
        for position in _successful(rows):
            row = rows[position]
            hamiltonian = np.asarray(
                context.model.cam_hamiltonian(row.state, row.parameters)
            )
            eigenvalues[position] = np.linalg.eigvals(hamiltonian)
        return _annotation(
            self,
            rows,
            {
                "hamiltonian_eigenvalues": eigenvalues,
                "mode_frequencies": np.real(eigenvalues),
            },
            field_axes={
                "hamiltonian_eigenvalues": ("mode",),
                "mode_frequencies": ("mode",),
            },
            axes={"mode": n_modes},
        )
