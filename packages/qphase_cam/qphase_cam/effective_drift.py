"""qphase_cam: local effective-drift corrections
---------------------------------------------------------
The centre-manifold correction gives the local stable covariance and the Itô drift
it induces. This module adds the *local* second-order averages of the drift, the
projected noise and the readout at one state — not the probe's full
one-dimensional Fokker-Planck curve solve.

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

from .center_manifold import (
    CenterManifoldConfig,
    CenterManifoldCorrection,
    center_manifold_correction,
)
from .core.noise import canonical_sample_matrix_covariance
from .observable import CAMObservable, DerivativeRecord
from .outcomes import CAMOutcomeStatus
from .validity import projected_intensity

__all__ = [
    "EffectiveCorrectionConfig",
    "EffectiveDriftCorrection",
    "effective_drift_correction",
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
