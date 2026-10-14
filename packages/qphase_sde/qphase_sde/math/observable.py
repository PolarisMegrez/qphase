"""qphase_sde: Hermitian Quadratic Observable Mathematics (2.0)
---------------------------------------------------------
Shared definition, validation and backend evaluation of Hermitian quadratic
observables ``y_o(t) = Re[alpha(t)^dagger Q_o alpha(t)] - c_o``. Both the
``quadratic_moments`` analyser and the ``observable_transform`` analyser use
this single implementation; there is no second evaluator.

Public API
----------
``QuadraticObservableSpec`` : Config schema of one observable (JSON-safe
    complex matrix representation).
``CompiledQuadraticObservables`` : Validated matrices/centers plus the
    requested/used provenance of the Hermitian projection.
``compile_quadratic_observables`` : Validate specs and compile them for
    backend evaluation.
``stage_quadratic_observables`` : Stage used matrices/centers on a backend.
``evaluate_quadratic_observables`` : Backend einsum evaluator with a real
    output residual check.
``complex_matrix_json`` : Strict-JSON complex matrix representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy

__all__ = [
    "CompiledQuadraticObservables",
    "QuadraticObservableSpec",
    "compile_quadratic_observables",
    "complex_matrix_json",
    "evaluate_quadratic_observables",
    "stage_quadratic_observables",
]


class QuadraticObservableSpec(BaseModel):
    """One real observable ``alpha^dagger Q alpha - center``."""

    model_config = ConfigDict(extra="forbid")

    matrix: list[list[complex]] = Field(
        description="Hermitian matrix Q in the configured mode order",
    )
    center: float | None = Field(
        None,
        description="Optional scalar center subtracted from alpha^dagger Q alpha",
    )
    reference_matrix: list[list[complex]] | None = Field(
        None,
        description="Optional Hermitian R_ref defining center=Tr(Q R_ref)",
    )

    @model_validator(mode="after")
    def validate_center(self) -> QuadraticObservableSpec:
        if self.center is not None and self.reference_matrix is not None:
            raise ValueError("center and reference_matrix are mutually exclusive")
        return self

    @field_serializer("matrix", "reference_matrix")
    def serialize_complex_matrix(
        self, value: list[list[complex]] | None
    ) -> list[list[str]] | None:
        if value is None:
            return None
        return [[str(entry) for entry in row] for row in value]


@dataclass(frozen=True)
class CompiledQuadraticObservables:
    """Validated Hermitian quadratic observables ready for evaluation.

    ``matrices_used`` and ``reference_matrices`` are the Hermitian-projected
    arrays actually evaluated (a missing reference is the zero matrix);
    ``matrices_requested`` preserves the configured coefficients.
    ``matrix_residuals``/``reference_residuals`` record the relative
    non-Hermitian residual the projection removed (``None`` when no reference
    was configured). ``centers`` are the subtracted constants, either given
    directly or computed as ``Re Tr(Q_o R_ref)``.
    """

    names: tuple[str, ...]
    matrices_requested: np.ndarray
    matrices_used: np.ndarray
    centers: np.ndarray
    reference_matrices: np.ndarray
    matrix_residuals: tuple[float, ...]
    reference_residuals: tuple[float | None, ...]
    tolerance: float


def _validate_hermitian(
    value: Any, n_modes: int, tolerance: float, label: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the as-configured matrix, its Hermitian projection and residual."""
    requested = np.asarray(value, dtype=complex)
    if requested.shape != (n_modes, n_modes):
        raise ValueError(f"{label} must have shape ({n_modes}, {n_modes})")
    if not np.all(np.isfinite(requested)):
        raise ValueError(f"{label} must contain finite values")
    scale = max(float(np.linalg.norm(requested)), 1.0)
    residual = float(np.linalg.norm(requested - requested.conj().T)) / scale
    if residual > tolerance:
        raise ValueError(f"{label} is not Hermitian: relative residual {residual:.3e}")
    return requested, (requested + requested.conj().T) / 2.0, residual


def compile_quadratic_observables(
    specs: Any,
    n_modes: int,
    tolerance: float,
) -> CompiledQuadraticObservables:
    """Validate named quadratic observable specs for an ``n_modes`` selection.

    Matrices within ``tolerance`` of Hermitian are projected to
    ``(Q + Q^dagger)/2``; the requested coefficients and the removed relative
    residual are retained for provenance. Larger residuals fail.
    """
    names: list[str] = []
    requested: list[np.ndarray] = []
    used: list[np.ndarray] = []
    centers: list[float] = []
    references: list[np.ndarray] = []
    matrix_residuals: list[float] = []
    reference_residuals: list[float | None] = []
    for name, spec in specs.items():
        matrix_requested, matrix, matrix_residual = _validate_hermitian(
            spec.matrix, n_modes, tolerance, f"observable {name!r} matrix"
        )
        if spec.reference_matrix is None:
            reference = np.zeros_like(matrix)
            reference_residual = None
            center = 0.0 if spec.center is None else float(spec.center)
        else:
            _, reference, reference_residual = _validate_hermitian(
                spec.reference_matrix,
                n_modes,
                tolerance,
                f"observable {name!r} reference_matrix",
            )
            traced = np.trace(matrix @ reference)
            if abs(float(np.imag(traced))) > tolerance * max(
                1.0, abs(float(np.real(traced)))
            ):
                raise ValueError(f"observable {name!r} reference center is not real")
            center = float(np.real(traced))
        names.append(name)
        requested.append(matrix_requested)
        used.append(matrix)
        centers.append(center)
        references.append(reference)
        matrix_residuals.append(matrix_residual)
        reference_residuals.append(reference_residual)
    return CompiledQuadraticObservables(
        names=tuple(names),
        matrices_requested=np.asarray(requested),
        matrices_used=np.asarray(used),
        centers=np.asarray(centers),
        reference_matrices=np.asarray(references),
        matrix_residuals=tuple(matrix_residuals),
        reference_residuals=tuple(reference_residuals),
        tolerance=float(tolerance),
    )


def stage_quadratic_observables(
    compiled: CompiledQuadraticObservables, values: Any, backend: BackendBase
) -> tuple[Any, Any]:
    """Stage used matrices and centers on ``backend`` matching ``values``."""
    matrices = backend.asarray(compiled.matrices_used, dtype=values.dtype)
    centers = backend.asarray(compiled.centers, dtype=values.real.dtype)
    return matrices, centers


def evaluate_quadratic_observables(
    values: Any,
    matrices: Any,
    centers: Any,
    *,
    backend: BackendBase,
    tolerance: float,
) -> Any:
    """Evaluate ``Re(alpha^dagger Q_o alpha) - c_o`` over trailing mode axes.

    ``values`` is a backend complex array ``(..., n_modes)``; ``matrices``
    and ``centers`` are the staged used arrays. The result is real with a
    trailing observable axis. The imaginary part of a Hermitian quadratic
    form is pure rounding: an imaginary level above ``tolerance`` relative to
    the real level (floored at 256 eps of the real dtype so reduced-precision
    rounding never trips the guard) signals a logic error and raises. NaN
    inputs censor silently — NaN comparisons never trip the guard.
    """
    z = backend.einsum("...i,oij,...j->...o", values.conj(), matrices, values)
    real = backend.real(z)
    imag_level = float(convert_to_numpy(backend.mean(backend.abs(backend.imag(z)))))
    real_level = float(convert_to_numpy(backend.mean(backend.abs(real))))
    eps_floor = 256.0 * float(np.finfo(np.dtype(real.dtype)).eps)
    if imag_level > max(tolerance, eps_floor) * max(1.0, real_level):
        raise ValueError(
            "quadratic observable has a non-negligible imaginary part: "
            f"mean |Im| {imag_level:.3e} vs mean |Re| {real_level:.3e}"
        )
    return real - centers


def complex_matrix_json(matrix: Any) -> list[list[str]]:
    """Strict-JSON complex matrix: the ``str(complex)`` per-entry form."""
    return [
        [str(complex(entry)) for entry in row]
        for row in np.asarray(matrix, dtype=complex)
    ]
