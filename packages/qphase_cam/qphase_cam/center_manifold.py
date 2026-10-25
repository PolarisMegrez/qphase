"""qphase_cam: centre-manifold covariance correction
---------------------------------------------------------
The linear-noise assessment corrects one state with the full (critical plus
stable) covariance. The centre-manifold correction keeps only the **stable**
subspace: the critical left eigenvector defines the direction the normal form
already describes, the noise is projected onto its complement, the stable Lyapunov
covariance is solved there, lifted back to the full space, and the Itô drift it
induces is projected onto the stable subspace and along the critical direction.

The numbers are the ones
``scripts/tools/probe_kerr_full_3mode_stochastic_center_manifold.py`` computes
(``_stable_covariance_correction`` and ``_readout_hessian_correction``), without
that script's model, mode index or readout:

* ``stable_basis = null_space(left)``, ``stable_jacobian = Bᵀ J B``,
  ``stable_noise = Bᵀ D B``, ``stable_covariance`` solves the stable Lyapunov
  equation, ``full_stable_covariance = B Σ_s Bᵀ``;
* ``drift_correction = ½ Σ_{ij} Σ_s[ij] · B(e_i, e_j)`` with the shared
  :func:`qphase_cam.nonlinear.bilinear_form` (the same symmetric forward Jacobian
  difference the nonlinear order recursion uses);
* ``stable_mean_shift = B · (−J_s⁻¹ Bᵀ c)``,
  ``critical_drift = l·c``, ``critical_shift = −critical_drift / (l J r)``;
* the Lyapunov relative residual is reported, never clamped;
* the readout curvature is evaluated at several step factors and reported as a
  :class:`~qphase_cam.observable.DerivativeRecord` (median, step list and the
  relative spread between factors), so a step-sensitive number cannot pass as a
  converged one.

The critical mode, the Lyapunov solve and the canonical noise covariance come from
:mod:`qphase_cam.validity` and :mod:`qphase_cam.core.noise`; nothing here
re-implements them. Failures are statuses: a non-finite state, an unresolved mode
or a degenerate stable basis, a noise covariance that is not positive
semidefinite, and a stable covariance that does not solve its Lyapunov equation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.linalg import null_space

from .core.noise import canonical_sample_matrix_covariance
from .nonlinear import bilinear_form
from .observable import CAMObservable, DerivativeRecord
from .outcomes import CAMOutcomeStatus
from .validity import critical_mode, lyapunov_covariance, minimum_eigenvalue

__all__ = [
    "CenterManifoldConfig",
    "CenterManifoldCorrection",
    "center_manifold_correction",
]


class CenterManifoldConfig(BaseModel):
    """Steps, tolerances and the readout-curvature controls."""

    model_config = ConfigDict(extra="forbid")

    hessian_step_factors: tuple[float, ...] = Field(
        default=(1e-3, 1e-4, 1e-5),
        min_length=1,
        description="Step factors (times max(|x|, 1)) of the readout curvature",
    )
    lyapunov_relative_residual_tolerance: float = Field(default=1e-8, ge=0.0)
    overlap_tolerance: float = Field(default=1e-10, gt=0.0)
    noise_tolerance: float = Field(default=1e-9, ge=0.0)


class CenterManifoldCorrection(BaseModel):
    """The stable-subspace correction of one state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CAMOutcomeStatus
    reason: str | None = None
    reason_code: str | None = None
    #: The critical eigenvectors the correction was built on (empty on failure), so a
    #: downstream correction reuses them instead of solving the eigenproblem again.
    left: tuple[float, ...] = ()
    right: tuple[float, ...] = ()
    critical_eigenvalue_real: float = float("nan")

    stable_basis: tuple[tuple[float, ...], ...] = ()
    stable_jacobian: tuple[tuple[float, ...], ...] = ()
    stable_noise: tuple[tuple[float, ...], ...] = ()
    stable_covariance: tuple[tuple[float, ...], ...] = ()
    full_stable_covariance: tuple[tuple[float, ...], ...] = ()
    drift_correction: tuple[float, ...] = ()
    stable_mean_shift: tuple[float, ...] = ()
    critical_drift_correction: float = float("nan")
    critical_linear_shift: float = float("nan")
    lyapunov_relative_residual: float = float("nan")

    readout_hessian_correction: float | None = None
    readout_derivative: DerivativeRecord | None = None

    method: dict[str, Any] = Field(default_factory=dict)


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in np.asarray(matrix))


def _failed(
    status: CAMOutcomeStatus, code: str, reason: str
) -> CenterManifoldCorrection:
    return CenterManifoldCorrection(
        status=status,
        reason=reason,
        reason_code=code,
        method={"service": "qphase_cam.center_manifold.center_manifold_correction"},
    )


def center_manifold_correction(
    adapter: Any,
    model: Any,
    observable: CAMObservable,
    *,
    state: Any,
    params: Mapping[str, Any],
    diffusion: Any | None = None,
    config: CenterManifoldConfig | None = None,
) -> CenterManifoldCorrection:
    """Correct the stationary covariance with the centre-manifold split.

    ``diffusion`` is the canonical noise covariance; without it the model's
    ``cam_diffusion`` is transformed through the shared
    :func:`~qphase_cam.core.noise.canonical_sample_matrix_covariance`.
    """
    settings = config or CenterManifoldConfig()
    point = np.asarray(state, dtype=float).reshape(-1)
    parameter_map = dict(params or {})
    method: dict[str, Any] = {
        "service": "qphase_cam.center_manifold.center_manifold_correction",
        "stable_basis": "null_space of the critical left eigenvector",
        "stable_covariance": (
            "continuous Lyapunov solve on the stable basis (direct Sylvester solve; "
            "least-squares Kronecker form when the spectrum is singular)"
        ),
        "drift_correction": "1/2 sum_ij Sigma_s[ij] B(e_i, e_j)",
        "readout_curvature": "median of 1/2 trace(H_s Sigma_s) over the step factors",
    }
    if point.size == 0 or not np.all(np.isfinite(point)):
        return _failed(
            CAMOutcomeStatus.NOT_CONVERGED,
            "state_not_finite",
            "the state is not finite",
        )
    jacobian = np.asarray(adapter.jacobian(point, parameter_map), dtype=float)
    if not np.all(np.isfinite(jacobian)):
        return _failed(
            CAMOutcomeStatus.NOT_CONVERGED,
            "jacobian_not_finite",
            "the dynamics Jacobian is not finite",
        )
    mode = critical_mode(
        jacobian, overlap_tolerance=settings.overlap_tolerance, rule="nearest_zero"
    )
    if mode is None:
        return _failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            "mode_unresolved",
            "the critical mode is unresolved",
        )
    left = np.asarray(mode.left, dtype=float)
    right = np.asarray(mode.right, dtype=float)
    if diffusion is None:
        if model is None or not callable(getattr(model, "cam_diffusion", None)):
            raise ValueError(
                "a centre-manifold correction needs the noise covariance or a model "
                "implementing cam_diffusion(state_matrix, params)"
            )
        matrix = np.asarray(adapter.state_matrix(point, parameter_map), dtype=complex)
        noise = canonical_sample_matrix_covariance(
            matrix, model.cam_diffusion(matrix, parameter_map)
        )
    else:
        noise = np.asarray(diffusion, dtype=float)
    if noise.shape != jacobian.shape:
        raise ValueError(
            f"the diffusion shape {noise.shape} does not match the Jacobian shape "
            f"{jacobian.shape}"
        )
    if not np.all(np.isfinite(noise)):
        return _failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "noise_not_finite",
            "the noise covariance is not finite",
        )
    smallest, _ = minimum_eigenvalue(noise)
    if smallest < -settings.noise_tolerance:
        return _failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "noise_not_psd",
            f"the noise covariance is not positive semidefinite ({smallest:.3e})",
        )
    basis = np.asarray(null_space(left[None, :]), dtype=float)
    if basis.shape[1] == 0 or basis.shape != (point.size, point.size - 1):
        return _failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            "stable_basis_degenerate",
            "the stable tangent basis is degenerate",
        )
    stable_jacobian = basis.T @ jacobian @ basis
    stable_noise = basis.T @ noise @ basis
    stable_covariance, residual = lyapunov_covariance(stable_jacobian, stable_noise)
    stable_covariance = (stable_covariance + stable_covariance.T) / 2.0
    full_stable = basis @ stable_covariance @ basis.T
    correction = np.zeros_like(point)
    for first in range(basis.shape[1]):
        for second in range(basis.shape[1]):
            correction += (
                0.5
                * stable_covariance[first, second]
                * bilinear_form(
                    adapter, point, parameter_map, basis[:, first], basis[:, second]
                )
            )
    if not np.all(np.isfinite(correction)):
        return _failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "drift_correction_not_finite",
            "the drift correction is not finite",
        )
    if residual > settings.lyapunov_relative_residual_tolerance:
        return _failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "lyapunov_residual",
            "the stable covariance does not solve the Lyapunov equation "
            f"(relative residual {residual:.3e})",
        )
    stable_shift = -np.linalg.solve(stable_jacobian, basis.T @ correction)
    denominator = float(left @ jacobian @ right)
    critical_drift = float(left @ correction)
    critical_shift = (
        float(-critical_drift / denominator) if denominator != 0.0 else float("nan")
    )

    scale = max(float(np.linalg.norm(point)), 1.0)
    samples: dict[str, float] = {}
    for factor in settings.hessian_step_factors:
        step = float(factor) * scale
        if step <= 0.0 or not np.isfinite(step):
            continue
        columns = np.empty((point.size, basis.shape[1]))
        for column in range(basis.shape[1]):
            offset = step * basis[:, column]
            upper = np.asarray(
                observable.gradient(adapter, point + offset, parameter_map).gradient,
                dtype=float,
            )
            lower = np.asarray(
                observable.gradient(adapter, point - offset, parameter_map).gradient,
                dtype=float,
            )
            columns[:, column] = (upper - lower) / (2.0 * step)
        stable_hessian = basis.T @ columns
        stable_hessian = (stable_hessian + stable_hessian.T) / 2.0
        samples[f"{factor:.0e}"] = float(
            0.5 * np.sum(stable_hessian * stable_covariance)
        )
    curvature = None
    record = None
    if samples:
        values = np.asarray(list(samples.values()), dtype=float)
        median = float(np.median(values))
        spread = float(np.max(values) - np.min(values))
        denominator_spread = abs(median)
        record = DerivativeRecord(
            method="central_difference",
            order=2,
            gradient=tuple(float(value) for value in values),
            step=tuple(
                float(factor) * scale for factor in settings.hessian_step_factors
            ),
            scale=tuple(float(scale) for _ in settings.hessian_step_factors),
            condition=None,
            repeat_relative_error=(
                spread / denominator_spread if denominator_spread > 0.0 else None
            ),
            metadata={
                "quantity": "readout_curvature",
                "steps_by_factor": dict(samples),
                "spread": spread,
                "aggregate": "median",
            },
        )
        curvature = median
    return CenterManifoldCorrection(
        status=CAMOutcomeStatus.VALID,
        reason=None,
        reason_code=None,
        left=tuple(float(value) for value in left),
        right=tuple(float(value) for value in right),
        critical_eigenvalue_real=float(np.real(mode.value)),
        stable_basis=_matrix_tuple(basis),
        stable_jacobian=_matrix_tuple(stable_jacobian),
        stable_noise=_matrix_tuple(stable_noise),
        stable_covariance=_matrix_tuple(stable_covariance),
        full_stable_covariance=_matrix_tuple(full_stable),
        drift_correction=tuple(float(value) for value in correction),
        stable_mean_shift=tuple(float(value) for value in basis @ stable_shift),
        critical_drift_correction=critical_drift,
        critical_linear_shift=critical_shift,
        lyapunov_relative_residual=float(residual),
        readout_hessian_correction=curvature,
        readout_derivative=record,
        method={**method, "readout_curvature_by_step": dict(samples)},
    )
