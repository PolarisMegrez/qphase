"""qphase_cam: local effective-drift corrections
---------------------------------------------------------
The centre-manifold correction gives the local stable covariance and the Itô drift
it induces. The module offers two things: :func:`effective_drift_correction`, the
*local* second-order averages of the drift, the projected noise and the readout at one
state, and
:func:`effective_drift_curve`, the full one-dimensional Fokker-Planck curve the
historical probe computed — a coarse stable-subspace solve per point, a dense PCHIP
table and the shared stationary quadrature on it.

The covariance chain is the probe-1 centre-manifold correction and it runs exactly
once. When it fails, its status, reason and reason code are returned unchanged and
no correction is computed on top of a covariance that did not validate.

* **drift** — ``center.drift_correction`` reused verbatim, recorded as
  ``method="analytic"`` with an explicit "identical estimator" note; it is never
  presented as a fresh finite difference.
* **diffusion** — ``½ Σ_ij Σ_s[ij] ∂_i∂_j q`` with
  ``q(x) = projected_intensity(noise(x), left, left)``: the derivatives run along
  the stable basis columns and ``q`` is always projected on the **critical left**
  eigenvector exposed by the centre correction. Without an explicit ``diffusion``
  array the noise is rebuilt at every perturbed state
  (``state_matrix`` → ``cam_diffusion`` → ``canonical_sample_matrix_covariance``),
  so a state-dependent diffusion is handled exactly; an explicit array means a
  fixed canonical covariance.
* **readout** — the same average with ``observable.value`` (value based). The
  gradient-based curvature the centre correction computes is reported next to it,
  so a quadratic readout can be cross-checked instead of trusted.

Each correction carries a :class:`~qphase_cam.observable.DerivativeRecord` with its
step factors, per-step values, absolute spread and relative repeat error (``None``
when the median is zero; the spread is still reported, so no division by zero and
no fabricated zero error).

``correction_dominates_linear`` follows the centre probe's diagnostic:
``|critical_drift_correction| / (critical_rate · sigma_x)`` with
``critical_rate = -Re(critical eigenvalue)`` and
``sigma_x = sqrt(q / (2 critical_rate))``. It is ``None`` when the rate or ``q`` is
not positive; the ratio and ``sigma_x`` are reported either way.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.interpolate import PchipInterpolator
from scipy.linalg import null_space

from .center_manifold import (
    CenterManifoldConfig,
    CenterManifoldCorrection,
    center_manifold_correction,
)
from .core.noise import canonical_sample_matrix_covariance
from .nonlinear import (
    NonlinearManifold,
    ScalarGeneratorEvaluation,
    ScalarGeneratorTable,
    ScalarReconstruction,
    ScalarReconstructionConfig,
    bilinear_form,
    evaluate_generator,
    manifold_drift,
    manifold_embedding,
)
from .observable import CAMObservable, DerivativeRecord
from .outcomes import CAMOutcomeStatus
from .validity import lyapunov_covariance, projected_intensity

__all__ = [
    "EffectiveCorrectionConfig",
    "EffectiveDriftCorrection",
    "effective_drift_correction",
    "EffectiveDriftConfig",
    "EffectiveDriftCurve",
    "effective_drift_curve",
]


class EffectiveCorrectionConfig(BaseModel):
    """Step factors of the second averages plus the centre-correction controls."""

    model_config = ConfigDict(extra="forbid")

    step_factors: tuple[float, ...] = Field(
        default=(1e-3, 1e-4, 1e-5),
        min_length=1,
        description="Step factors (times max(|x|, 1)) of the second differences",
    )
    center: CenterManifoldConfig = Field(default_factory=CenterManifoldConfig)

    @field_validator("step_factors")
    @classmethod
    def _check_step_factors(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        for factor in value:
            if not np.isfinite(float(factor)) or float(factor) <= 0.0:
                raise ValueError(
                    f"step factors must be finite and positive; got {factor}"
                )
        return value


class EffectiveDriftCorrection(BaseModel):
    """The local drift, projected-noise and readout corrections of one state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CAMOutcomeStatus
    reason: str | None = None
    reason_code: str | None = None
    center: CenterManifoldCorrection

    effective_drift_correction: tuple[float, ...] = ()
    effective_diffusion_correction: float | None = None
    effective_readout_correction: float | None = None
    center_readout_hessian_correction: float | None = None

    critical_rate: float | None = None
    projected_noise: float | None = None
    sigma_x: float | None = None
    dominance_ratio: float | None = None
    correction_dominates_linear: bool | None = None

    drift_derivative: DerivativeRecord | None = None
    diffusion_derivative: DerivativeRecord | None = None
    readout_derivative: DerivativeRecord | None = None
    method: dict[str, Any] = Field(default_factory=dict)


def _record(
    samples: dict[float, float], scale: float, *, quantity: str
) -> DerivativeRecord:
    values = np.asarray(list(samples.values()), dtype=float)
    median = float(np.median(values))
    spread = float(np.max(values) - np.min(values))
    return DerivativeRecord(
        method="central_difference",
        order=2,
        gradient=tuple(float(value) for value in values),
        step=tuple(float(factor) * scale for factor in samples),
        scale=tuple(float(scale) for _ in samples),
        condition=None,
        repeat_relative_error=(spread / abs(median) if median != 0.0 else None),
        metadata={
            "quantity": quantity,
            "aggregate": "median",
            "absolute_spread": spread,
            "by_step": {f"{factor:.0e}": value for factor, value in samples.items()},
        },
    )


def _second_average(
    evaluate: Any,
    basis: np.ndarray,
    covariance: np.ndarray,
    point: np.ndarray,
    step: float,
) -> float:
    """Return ``½ Σ_ij Σ_s[ij] ∂_i∂_j f`` at one step, on the stable basis."""
    center_value = float(evaluate(point))
    total = 0.0
    for first in range(basis.shape[1]):
        for second in range(first, basis.shape[1]):
            weight = 0.5 * covariance[first, second]
            if first != second:
                weight *= 2.0
            if weight == 0.0:
                continue
            offset_first = step * basis[:, first]
            offset_second = step * basis[:, second]
            if first == second:
                value = (
                    float(evaluate(point + offset_first))
                    - 2.0 * center_value
                    + float(evaluate(point - offset_first))
                ) / step**2
            else:
                value = (
                    float(evaluate(point + offset_first + offset_second))
                    - float(evaluate(point + offset_first - offset_second))
                    - float(evaluate(point - offset_first + offset_second))
                    + float(evaluate(point - offset_first - offset_second))
                ) / (4.0 * step**2)
            total += weight * value
    return total


def effective_drift_correction(
    adapter: Any,
    model: Any,
    observable: CAMObservable,
    *,
    state: Any,
    params: Mapping[str, Any],
    diffusion: Any | None = None,
    config: EffectiveCorrectionConfig | None = None,
) -> EffectiveDriftCorrection:
    """Return the local effective-drift correction of one state."""
    settings = config or EffectiveCorrectionConfig()
    point = np.asarray(state, dtype=float).reshape(-1)
    parameter_map = dict(params or {})
    center = center_manifold_correction(
        adapter,
        model,
        observable,
        state=point,
        params=parameter_map,
        diffusion=diffusion,
        config=settings.center,
    )
    method: dict[str, Any] = {
        "service": "qphase_cam.effective_drift.effective_drift_correction",
        "covariance": "qphase_cam.center_manifold.center_manifold_correction",
        "drift": "center.drift_correction (identical estimator)",
        "diffusion": "1/2 sum_ij Sigma_s[ij] d_i d_j q, q = left^T D(x) left",
        "readout": "1/2 sum_ij Sigma_s[ij] d_i d_j f, value based",
        "scope": "one local second-order average; not a Fokker-Planck curve solve",
    }
    if center.status is not CAMOutcomeStatus.VALID:
        return EffectiveDriftCorrection(
            status=center.status,
            reason=center.reason,
            reason_code=center.reason_code,
            center=center,
            method=method,
        )
    basis = np.asarray(center.stable_basis, dtype=float)
    covariance = np.asarray(center.stable_covariance, dtype=float)
    left = np.asarray(center.left, dtype=float)
    if left.size != point.size:
        raise ValueError("the centre correction did not expose a critical left vector")
    if diffusion is None:
        if model is None or not callable(getattr(model, "cam_diffusion", None)):
            raise ValueError(
                "an effective-drift correction needs the noise covariance or a model "
                "implementing cam_diffusion(state_matrix, params)"
            )

        def noise_at(vector: Any) -> np.ndarray:
            matrix = np.asarray(
                adapter.state_matrix(vector, parameter_map), dtype=complex
            )
            return canonical_sample_matrix_covariance(
                matrix, model.cam_diffusion(matrix, parameter_map)
            )

    else:
        fixed = np.asarray(diffusion, dtype=float)

        def noise_at(vector: Any) -> np.ndarray:
            del vector
            return fixed

    def projected_noise(vector: Any) -> float:
        return projected_intensity(noise_at(vector), left, left)

    def readout(vector: Any) -> float:
        return float(observable.value(adapter, vector, parameter_map))

    scale = max(float(np.linalg.norm(point)), 1.0)
    diffusion_samples: dict[float, float] = {}
    readout_samples: dict[float, float] = {}
    for factor in settings.step_factors:
        step = float(factor) * scale
        if step <= 0.0 or not np.isfinite(step):
            continue
        diffusion_samples[float(factor)] = _second_average(
            projected_noise, basis, covariance, point, step
        )
        readout_samples[float(factor)] = _second_average(
            readout, basis, covariance, point, step
        )
    if not diffusion_samples:
        return EffectiveDriftCorrection(
            status=CAMOutcomeStatus.UNIDENTIFIABLE,
            reason="no usable second-difference step",
            reason_code="no_step",
            center=center,
            method=method,
        )
    samples = list(diffusion_samples.values()) + list(readout_samples.values())
    if not all(np.isfinite(float(value)) for value in samples):
        # A second average that is not finite is a failed correction, not a valid
        # one with a NaN field; the centre diagnostics stay for inspection.
        return EffectiveDriftCorrection(
            status=CAMOutcomeStatus.STOCHASTIC_INVALID,
            reason="a second-order correction is not finite",
            reason_code="correction_not_finite",
            center=center,
            method=method,
        )
    drift = np.asarray(center.drift_correction, dtype=float)
    rate = -float(center.critical_eigenvalue_real)
    q_point = float(projected_noise(point))
    sigma_x: float | None = None
    ratio: float | None = None
    dominates: bool | None = None
    if rate > 0.0 and q_point > 0.0:
        sigma_x = float(np.sqrt(q_point / (2.0 * rate)))
        if sigma_x > 0.0:
            ratio = float(abs(center.critical_drift_correction) / (rate * sigma_x))
            dominates = bool(ratio > 1.0)
    return EffectiveDriftCorrection(
        status=CAMOutcomeStatus.VALID,
        reason=None,
        reason_code=None,
        center=center,
        effective_drift_correction=tuple(float(value) for value in drift),
        effective_diffusion_correction=float(
            np.median(list(diffusion_samples.values()))
        ),
        effective_readout_correction=float(np.median(list(readout_samples.values()))),
        center_readout_hessian_correction=center.readout_hessian_correction,
        critical_rate=rate if np.isfinite(rate) else None,
        projected_noise=q_point if np.isfinite(q_point) else None,
        sigma_x=sigma_x,
        dominance_ratio=ratio,
        correction_dominates_linear=dominates,
        drift_derivative=DerivativeRecord(
            method="analytic",
            order=2,
            gradient=tuple(float(value) for value in drift),
            metadata={
                "quantity": "effective_drift_correction",
                "source": "center.drift_correction (identical estimator)",
            },
        ),
        diffusion_derivative=_record(
            diffusion_samples, scale, quantity="effective_diffusion_correction"
        ),
        readout_derivative=_record(
            readout_samples, scale, quantity="effective_readout_correction"
        ),
        method=method,
    )


class EffectiveDriftConfig(BaseModel):
    """Grids, tolerances and quadrature of one full effective-drift curve.

    ``coarse_points_per_side`` and ``dense_points`` are the two grids the study uses:
    the coarse grid carries the per-point stable-subspace solve and the dense grid is
    the PCHIP interpolation the quadrature integrates. ``derivative_step_factor`` is the
    single step of the stable eigendirection averages, ``physical_tolerance`` and
    ``stable_covariance_tolerance`` are the curve's own selection tolerances (the source
    reconstruction keeps its own, stricter boundary tolerances), and ``quadrature``
    configures the dense :func:`~qphase_cam.nonlinear.evaluate_generator` only.
    """

    model_config = ConfigDict(extra="forbid")

    coarse_points_per_side: int = Field(default=201, ge=3)
    dense_points: int = Field(default=6001, ge=11)
    derivative_step_factor: float = Field(default=3.0e-7, gt=0.0)
    physical_tolerance: float = Field(default=1.0e-9, ge=0.0)
    stable_covariance_tolerance: float = Field(default=1.0e-7, ge=0.0)
    quadrature: ScalarReconstructionConfig = Field(
        default_factory=lambda: ScalarReconstructionConfig(
            integration_split="nearest_zero"
        )
    )


class EffectiveDriftCurve(BaseModel):
    """The full one-dimensional effective-drift curve of one reconstruction.

    ``coarse`` holds the per-point arrays (including the invalid points' NaN and the
    validity mask), ``dense`` the interpolated table the quadrature integrates, and
    ``evaluation`` the shared stationary evaluation of that table. ``source_status`` is
    the reconstruction's own status: a source that was not valid is never reported as a
    valid curve.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CAMOutcomeStatus
    reason: str | None = None
    reason_code: str | None = None
    source_status: CAMOutcomeStatus

    manifold: NonlinearManifold
    generator: ScalarGeneratorTable | None = None
    evaluation: ScalarGeneratorEvaluation | None = None

    coarse: dict[str, Any] = Field(default_factory=dict)
    dense: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    method: dict[str, Any] = Field(default_factory=dict)


def _eigendirection_average(
    function: Any,
    vector: np.ndarray,
    stable_basis: np.ndarray,
    stable_covariance: np.ndarray,
    step_factor: float,
) -> float:
    """Return ``½ Σ_k λ_k d²f/du_k²`` along the stable covariance's eigendirections.

    The step is a single ``step_factor * max(‖vector‖, 1)`` for every direction and the
    second difference is central: this is the historical probe's convention, and it is
    deliberately *not* the multi-step median of
    :func:`~qphase_cam.effective_drift._second_average` — the two do not agree and the
    published curve was produced with this one.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(stable_covariance)
    step = float(step_factor) * max(float(np.linalg.norm(vector)), 1.0)
    base = float(function(vector))
    correction = 0.0
    for variance, coordinate in zip(eigenvalues, eigenvectors.T, strict=True):
        if float(variance) <= 0.0:
            continue
        direction = stable_basis @ coordinate
        second = (
            float(function(vector + step * direction))
            - 2.0 * base
            + float(function(vector - step * direction))
        ) / step**2
        correction += 0.5 * float(variance) * second
    return float(correction)


def effective_drift_curve(
    adapter: Any,
    model: Any,
    observable: CAMObservable,
    *,
    state: Any,
    params: Mapping[str, Any],
    reconstruction: ScalarReconstruction,
    parameter_direction: Any | None = None,
    config: EffectiveDriftConfig | None = None,
) -> EffectiveDriftCurve:
    """Return the full effective-drift curve of one reconstruction.

    The reconstruction supplies the manifold, the physical boundaries and the soft-mode
    width, so the curve is the one-dimensional problem the caller already defined: this
    function never resolves its own boundaries. A reconstruction that is not ``valid``
    keeps its status — an uncontrolled approximation is not turned into a curve by being
    evaluated on a finer grid.

    Each coarse point solves the stable subspace exactly as the reference probe does:
    ``P = I − tangent·left``, the stable basis from ``null_space(left)``, the stable
    Jacobian ``Bᵀ P J B``, the stable noise ``Bᵀ P Σ Pᵀ B`` and its Lyapunov covariance;
    a point is kept only when the state is physical, the stable margin is positive and
    the covariance is positive semidefinite. The dense grid interpolates the kept
    interval and the shared :func:`~qphase_cam.nonlinear.evaluate_generator` evaluates
    it.

    Without ``parameter_direction`` there is no forcing: the per-point forcing stays
    NaN (it is unavailable, not zero) and the evaluation runs unforced, so the
    path-information rate and the stationary response are ``None``.
    """
    settings = config or EffectiveDriftConfig()
    point = np.asarray(state, dtype=float).reshape(-1)
    parameter_map = dict(params or {})
    method: dict[str, Any] = {
        "service": "qphase_cam.effective_drift.effective_drift_curve",
        "source": "the caller's scalar_reconstruction (manifold, boundaries, sigma)",
        "coarse_grid": (
            f"{settings.coarse_points_per_side} points per side from the "
            "reconstruction's boundaries, deduplicated at zero"
        ),
        "stable_projection": "P = I - tangent.left; stable basis = null_space(left)",
        "stable_jacobian": "B^T P J(x) B",
        "stable_noise": "B^T P Sigma P^T B",
        "stable_covariance": "continuous Lyapunov solve (shared public path)",
        "drift_correction": (
            "1/2 sum_ij Sigma_s[ij] B(e_i, e_j), central Hessian columns"
        ),
        "stable_mean": "B (-J_s^-1 B^T P drift_correction)",
        "second_average": (
            "half sum_k lambda_k d2f/du_k^2 along the stable covariance's "
            "eigendirections, one central step step_factor*max(|x|,1)"
        ),
        "quadrature": "evaluate_generator on the dense table",
        "integration_split": settings.quadrature.integration_split,
        "derivative_step_factor": float(settings.derivative_step_factor),
        "physical_tolerance": float(settings.physical_tolerance),
        "stable_covariance_tolerance": float(settings.stable_covariance_tolerance),
        "parameter_direction": parameter_direction is not None,
    }
    diagnostics: dict[str, Any] = {}
    manifold = reconstruction.manifold

    def failed(status: CAMOutcomeStatus, code: str, reason: str) -> EffectiveDriftCurve:
        return EffectiveDriftCurve(
            status=status,
            reason=reason,
            reason_code=code,
            source_status=reconstruction.status,
            manifold=manifold,
            method=dict(method),
            summary=dict(diagnostics),
        )

    if reconstruction.status is not CAMOutcomeStatus.VALID:
        return failed(
            reconstruction.status,
            "source_reconstruction_not_valid",
            "the source reconstruction is not valid "
            f"({reconstruction.status.value}: {reconstruction.reason or 'no reason'})",
        )
    negative = reconstruction.x_min
    positive = reconstruction.x_max
    sigma = reconstruction.sigma
    if negative is None or positive is None or sigma is None:
        raise ValueError(
            "a valid reconstruction must carry x_min, x_max and sigma; the curve "
            "cannot place its grid without them"
        )
    if not (negative < 0.0 < positive):
        raise ValueError(
            f"the reconstruction interval [{negative}, {positive}] does not straddle "
            "zero; the curve needs both sides"
        )
    if not np.all(np.isfinite(point)):
        raise ValueError("the state must be finite")

    left = np.asarray(manifold.left, dtype=float)
    left_points = (int(settings.coarse_points_per_side) + 1) // 2
    x_coarse = np.concatenate(
        (
            np.linspace(negative, 0.0, left_points),
            np.linspace(0.0, positive, left_points)[1:],
        )
    )
    vectors = manifold_embedding(manifold, point, x_coarse)
    # The tangent is the manifold's own first derivative, so the projection is the one
    # the reference probe builds.
    tangent = np.broadcast_to(np.asarray(manifold.embeddings[0]), vectors.shape).copy()
    for degree, coefficient in enumerate(manifold.embeddings[1:], start=2):
        tangent += degree * x_coarse[:, None] ** (degree - 1) * np.asarray(coefficient)
    stable_basis = np.asarray(null_space(left[None, :]), dtype=float)
    if stable_basis.shape != (point.size, point.size - 1):
        raise ValueError(
            f"the stable basis has shape {stable_basis.shape} for a "
            f"{point.size}-coordinate state"
        )
    identity = np.eye(point.size)
    hessian = np.empty((point.size, point.size, point.size), dtype=float)
    for first in range(point.size):
        for second in range(point.size):
            hessian[:, first, second] = bilinear_form(
                adapter,
                point,
                parameter_map,
                identity[:, first],
                identity[:, second],
            )
    direction = (
        None
        if parameter_direction is None
        else np.asarray(parameter_direction, dtype=float).reshape(-1)
    )

    f_deterministic = manifold_drift(manifold, x_coarse)
    f_effective = np.full(x_coarse.size, np.nan)
    q_effective = np.full(x_coarse.size, np.nan)
    omega_effective = np.full(x_coarse.size, np.nan)
    forcing = np.full(x_coarse.size, np.nan)
    stable_covariance_trace = np.full(x_coarse.size, np.nan)
    stable_margin = np.full(x_coarse.size, np.nan)
    physical_minimum = np.full(x_coarse.size, np.nan)
    residual = np.full(x_coarse.size, np.nan)
    valid = np.zeros(x_coarse.size, dtype=bool)

    def covariance_at(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(adapter.state_matrix(vector, parameter_map), dtype=complex)
        diffusion = np.asarray(
            model.cam_diffusion(matrix, parameter_map), dtype=complex
        )
        return matrix, canonical_sample_matrix_covariance(matrix, diffusion)

    def q_at(vector: np.ndarray) -> float:
        _, covariance = covariance_at(vector)
        return projected_intensity(covariance, left, left)

    def omega_at(vector: np.ndarray) -> float:
        return float(observable.value(adapter, vector, parameter_map))

    for index, (vector, tangent_vector) in enumerate(
        zip(vectors, tangent, strict=True)
    ):
        matrix, covariance = covariance_at(vector)
        physical_minimum[index] = float(
            np.min(np.linalg.eigvalsh((matrix + matrix.conjugate().T) / 2.0))
        )
        projection = identity - np.outer(tangent_vector, left)
        local_jacobian = np.asarray(
            adapter.jacobian(vector, parameter_map), dtype=float
        )
        stable_jacobian = stable_basis.T @ projection @ local_jacobian @ stable_basis
        stable_margin[index] = -float(
            np.max(np.real(np.linalg.eigvals(stable_jacobian)))
        )
        if (
            physical_minimum[index] < -settings.physical_tolerance
            or stable_margin[index] <= 0.0
        ):
            continue
        stable_noise = (
            stable_basis.T @ projection @ covariance @ projection.T @ stable_basis
        )
        stable_covariance, lyapunov_residual = lyapunov_covariance(
            stable_jacobian, stable_noise
        )
        stable_covariance = (stable_covariance + stable_covariance.T) / 2.0
        residual[index] = float(lyapunov_residual)
        if np.min(np.linalg.eigvalsh(stable_covariance)) < (
            -settings.stable_covariance_tolerance
        ):
            continue
        full_stable_covariance = stable_basis @ stable_covariance @ stable_basis.T
        drift_correction = 0.5 * np.einsum(
            "kij,ij->k", hessian, full_stable_covariance, optimize=True
        )
        stable_forcing = stable_basis.T @ projection @ drift_correction
        stable_mean = stable_basis @ (-np.linalg.solve(stable_jacobian, stable_forcing))
        shifted = vector + stable_mean
        critical_mean = float(
            left
            @ (adapter.rhs(shifted, parameter_map) - adapter.rhs(vector, parameter_map))
        )
        f_effective[index] = (
            f_deterministic[index] + critical_mean + float(left @ drift_correction)
        )
        q_effective[index] = q_at(shifted) + _eigendirection_average(
            q_at,
            vector,
            stable_basis,
            stable_covariance,
            settings.derivative_step_factor,
        )
        omega_effective[index] = omega_at(shifted) + _eigendirection_average(
            omega_at,
            vector,
            stable_basis,
            stable_covariance,
            settings.derivative_step_factor,
        )
        forcing_is_finite = True
        if direction is not None:
            parameter_jacobian = np.asarray(
                adapter.parameter_jacobian(vector, parameter_map), dtype=float
            )
            forcing[index] = float(left @ parameter_jacobian @ direction)
            forcing_is_finite = bool(np.isfinite(forcing[index]))
        stable_covariance_trace[index] = float(np.trace(stable_covariance))
        valid[index] = bool(
            np.isfinite(f_effective[index])
            and np.isfinite(q_effective[index])
            and q_effective[index] > 0.0
            and np.isfinite(omega_effective[index])
            and forcing_is_finite
        )

    zero_index = int(np.argmin(np.abs(x_coarse)))
    left_valid = np.flatnonzero(valid[: zero_index + 1])
    right_valid = np.flatnonzero(valid[zero_index:]) + zero_index

    def span(values: np.ndarray) -> list[float] | None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        return [float(np.min(finite)), float(np.max(finite))]

    if left_valid.size == 0 or right_valid.size == 0:
        diagnostics.update(
            {
                "valid_coarse_count": int(np.count_nonzero(valid)),
                "stable_margin_range": span(stable_margin),
                "physical_minimum_range": span(physical_minimum),
            }
        )
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "no_stable_interval",
            "no stable physical interval contains the critical point",
        )
    start = int(left_valid[0])
    stop = int(right_valid[-1]) + 1
    interval = slice(start, stop)
    if not np.all(valid[interval]):
        diagnostics.update(
            {
                "valid_coarse_count": int(np.count_nonzero(valid)),
                "invalid_inside_interval_count": int(
                    np.count_nonzero(~valid[interval])
                ),
                "stable_margin_range": span(stable_margin),
                "physical_minimum_range": span(physical_minimum),
            }
        )
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "disconnected_interval",
            "the stable physical interval is disconnected",
        )

    x_dense = np.linspace(
        x_coarse[start], x_coarse[stop - 1], int(settings.dense_points)
    )
    coarse_interval = x_coarse[interval]

    def interpolate(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            PchipInterpolator(coarse_interval, values[interval])(x_dense), dtype=float
        )

    f_dense = interpolate(f_effective)
    q_dense = interpolate(q_effective)
    omega_dense = interpolate(omega_effective)
    forcing_dense = None if direction is None else interpolate(forcing)
    stable_margin_dense = interpolate(stable_margin)
    table = ScalarGeneratorTable(
        x=tuple(float(value) for value in x_dense),
        drift=tuple(float(value) for value in f_dense),
        diffusion=tuple(float(value) for value in q_dense),
        readout=tuple(float(value) for value in omega_dense),
    )
    evaluation = evaluate_generator(
        table,
        forcing=forcing_dense,
        config=settings.quadrature,
    )
    if not evaluation.normalizable or evaluation.scalar_intensity is None:
        diagnostics.update(
            {
                "density_normalization_raw": evaluation.density_normalization,
                "minimum_q_effective": float(np.min(q_dense)),
                "coarse_interval": [
                    float(x_coarse[start]),
                    float(x_coarse[stop - 1]),
                ],
            }
        )
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "dense_density_not_normalizable",
            "the dense stationary density does not normalize",
        )
    density = np.asarray(evaluation.density, dtype=float)
    rate = float(manifold.soft_rate)
    stable_mass = {
        factor: float(
            np.trapezoid(density * (stable_margin_dense <= factor * rate), x_dense)
        )
        for factor in (1.0, 3.0, 5.0, 10.0)
    }
    information = (
        None
        if evaluation.stationary_response is None
        else float(evaluation.stationary_response) ** 2
        / float(evaluation.scalar_intensity)
    )
    path_information = evaluation.path_information_rate
    summary: dict[str, Any] = {
        "valid_coarse_count": int(np.count_nonzero(valid)),
        "coarse_interval": [float(x_coarse[start]), float(x_coarse[stop - 1])],
        "stationary_mean": evaluation.stationary_mean,
        "stationary_mean_x": evaluation.stationary_mean_x,
        "stationary_response": evaluation.stationary_response,
        "allan_intensity": evaluation.scalar_intensity,
        "information_rate": information,
        "path_information_rate": path_information,
        "information_over_path": (
            None
            if (information is None or not path_information)
            else float(information / path_information)
        ),
        "edge_mass_fraction": evaluation.edge_mass_fraction,
        "endpoint_density_ratio": evaluation.endpoint_density_ratio,
        "minimum_q_effective": float(np.min(q_dense)),
        "minimum_stable_margin": float(np.min(stable_margin[interval])),
        "probability_stable_margin_le_1x_soft_rate": stable_mass[1.0],
        "probability_stable_margin_le_3x_soft_rate": stable_mass[3.0],
        "probability_stable_margin_le_5x_soft_rate": stable_mass[5.0],
        "probability_stable_margin_le_10x_soft_rate": stable_mass[10.0],
        "negative_boundary": float(negative),
        "positive_boundary": float(positive),
        "negative_physical_boundary": bool(reconstruction.negative_physical_boundary),
        "positive_physical_boundary": bool(reconstruction.positive_physical_boundary),
        "sigma_x": float(sigma),
        "lyapunov_residual_max": (
            float(np.nanmax(residual)) if np.any(np.isfinite(residual)) else None
        ),
        "dense_point_count": int(x_dense.size),
        "coarse_point_count": int(x_coarse.size),
    }
    method["coarse_points_per_side"] = int(settings.coarse_points_per_side)
    method["dense_points"] = int(settings.dense_points)
    method["core_point_count"] = evaluation.core_point_count
    method["edge_point_count"] = evaluation.edge_point_count
    method["density_normalization_raw"] = evaluation.density_normalization
    coarse = {
        "x": x_coarse,
        "f_deterministic": f_deterministic,
        "f_effective": f_effective,
        "q_effective": q_effective,
        "omega_effective": omega_effective,
        "forcing": forcing,
        "stable_covariance_trace": stable_covariance_trace,
        "stable_margin": stable_margin,
        "physical_minimum": physical_minimum,
        "lyapunov_residual": residual,
        "valid": valid,
    }
    dense = {
        "x": x_dense,
        "density": density,
        "f_effective": f_dense,
        "q_effective": q_dense,
        "omega_effective": omega_dense,
        "forcing": forcing_dense,
        "stable_margin": stable_margin_dense,
    }
    return EffectiveDriftCurve(
        status=CAMOutcomeStatus.VALID,
        source_status=reconstruction.status,
        manifold=manifold,
        generator=table,
        evaluation=evaluation,
        coarse=coarse,
        dense=dense,
        summary=summary,
        method=dict(method),
    )
