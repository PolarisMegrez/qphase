"""Short-delay coherence numerical core."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from ...contracts.quantities import (
    DEFAULT_FREQUENCY_ORIENTATION,
    FrequencyOrientation,
    orientation_sign,
)

__all__ = ["CoherenceCarrierEstimate", "estimate_coherence_carrier"]


@dataclass(frozen=True)
class CoherenceCarrierEstimate:
    """One zero-delay phase-derivative estimate and its diagnostics.

    ``conditional_std`` is the conditional uncertainty of the selected
    slope 鈥?the local phase-fit residual variance propagated through the
    normal equations, i.e. the error at the computed estimate given the
    ensemble correlation 鈥?while ``frequency_sem`` carries the sampling
    scope over independent trajectories. ``confidence_lower``/
    ``confidence_upper`` are the conditional normal-approximation interval
    at the configured confidence level.
    """

    frequency: float = math.nan
    frequency_sem: float = math.nan
    conditional_std: float = math.nan
    confidence_lower: float = math.nan
    confidence_upper: float = math.nan
    selected_lag_points: int = 0
    selected_lag_time: float = math.nan
    phase_fit_rms: float = math.nan
    first_lag_coherence: float = math.nan
    first_lag_phase: float = math.nan
    nyquist_fraction: float = math.nan
    status: str = "failed"
    error: str = ""


@dataclass(frozen=True)
class _CandidateFit:
    lag_points: int
    frequency: float
    frequency_sem: float
    conditional_std: float
    phase_fit_rms: float


def _phase_slope(
    correlation: np.ndarray,
    dt: float,
    lag_points: int,
    order: int,
) -> tuple[float, float, float]:
    c0 = complex(correlation[0])
    if not np.isfinite(c0.real) or abs(c0) <= np.finfo(float).tiny:
        return math.nan, math.nan, math.nan
    normalized = np.asarray(correlation[: lag_points + 1], dtype=complex) / c0
    phase = np.unwrap(np.angle(normalized))
    x = np.arange(1, lag_points + 1, dtype=float) / float(lag_points)
    design = np.column_stack([x**power for power in range(1, order + 1)])
    coefficients, *_ = np.linalg.lstsq(design, phase[1:], rcond=None)
    fitted = design @ coefficients
    slope = float(coefficients[0] / (lag_points * dt))
    residuals = phase[1:] - fitted
    rms = float(np.sqrt(np.mean(residuals**2)))
    # Conditional standard error of the slope: the residual variance of the
    # local phase fit propagated through the normal equations 鈥?the error at
    # the computed estimate given the ensemble correlation, not the scatter
    # over trajectories (that stays with the jackknife SEM).
    gram_inverse = np.linalg.inv(design.T @ design)
    conditional = float(
        math.sqrt(
            max(
                float(np.sum(residuals**2)) / (lag_points - order) * gram_inverse[0, 0],
                0.0,
            )
        )
        / (lag_points * dt)
    )
    return slope, rms, conditional


def _jackknife_sem(
    per_trajectory: np.ndarray,
    dt: float,
    lag_points: int,
    order: int,
    sign: float,
) -> float:
    n_traj = int(per_trajectory.shape[0])
    if n_traj < 2:
        return math.nan
    total = np.sum(per_trajectory, axis=0)
    estimates = np.empty(n_traj, dtype=float)
    for index in range(n_traj):
        leave_one = (total - per_trajectory[index]) / float(n_traj - 1)
        slope, _, _ = _phase_slope(leave_one, dt, lag_points, order)
        estimates[index] = sign * slope
    if not np.all(np.isfinite(estimates)):
        return math.nan
    center = float(np.mean(estimates))
    return float(np.sqrt((n_traj - 1.0) / n_traj * np.sum((estimates - center) ** 2)))


def _consistent(
    candidate: _CandidateFit,
    references: list[_CandidateFit],
    sigma: float,
) -> bool:
    for reference in references:
        combined = math.hypot(candidate.frequency_sem, reference.frequency_sem)
        if not np.isfinite(combined):
            return False
        numerical = (
            64.0
            * np.finfo(float).eps
            * max(1.0, abs(candidate.frequency), abs(reference.frequency))
        )
        if (
            abs(candidate.frequency - reference.frequency)
            > sigma * combined + numerical
        ):
            return False
    return True


def estimate_coherence_carrier(
    per_trajectory_correlation: np.ndarray,
    dt: float,
    *,
    orientation: FrequencyOrientation = DEFAULT_FREQUENCY_ORIENTATION,
    polynomial_order: int = 2,
    minimum_lag_points: int = 4,
    maximum_lag_points: int | None = None,
    consistency_sigma: float = 2.0,
    intensity_floor: float = 1.0e-14,
    confidence_level: float = 0.95,
) -> tuple[CoherenceCarrierEstimate, list[_CandidateFit]]:
    """Estimate ``d arg C(tau)/d tau`` at ``tau=0+``.

    The input contains one independently time-averaged correlation row per
    trajectory. The point estimate is formed from the ensemble correlation;
    leave-one-trajectory-out estimates are used only for uncertainty and
    nested-window selection. ``confidence_level`` sets the level of the
    conditional normal-approximation interval around the selected
    frequency.
    """
    values = np.asarray(per_trajectory_correlation, dtype=complex)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 3:
        return (
            CoherenceCarrierEstimate(
                error="correlations must have shape (n_traj, n_lags>=3)"
            ),
            [],
        )
    if not np.isfinite(dt) or dt <= 0.0:
        return CoherenceCarrierEstimate(error="dt must be positive"), []
    correlation = np.mean(values, axis=0)
    intensity = float(np.real(correlation[0]))
    if not np.isfinite(intensity) or intensity <= intensity_floor:
        return CoherenceCarrierEstimate(error="readout has non-positive intensity"), []

    available = values.shape[1] - 1
    maximum = (
        available if maximum_lag_points is None else min(available, maximum_lag_points)
    )
    minimum = max(minimum_lag_points, polynomial_order + 1)
    if maximum < minimum:
        return CoherenceCarrierEstimate(error="too few lag points for local fit"), []

    sign = orientation_sign(orientation)
    candidates: list[_CandidateFit] = []
    for lag_points in range(minimum, maximum + 1):
        slope, residual, conditional = _phase_slope(
            correlation, dt, lag_points, polynomial_order
        )
        sem = _jackknife_sem(values, dt, lag_points, polynomial_order, sign)
        candidates.append(
            _CandidateFit(
                lag_points=lag_points,
                frequency=sign * slope,
                frequency_sem=sem,
                conditional_std=conditional,
                phase_fit_rms=residual,
            )
        )

    selected = candidates[0]
    if values.shape[0] > 1:
        for candidate in candidates[1:]:
            references = candidates[: candidate.lag_points - minimum]
            if _consistent(candidate, references, consistency_sigma):
                selected = candidate
    else:
        selected = min(candidates, key=lambda candidate: candidate.phase_fit_rms)

    normalized = correlation / correlation[0]
    first_phase = float(np.angle(normalized[1]))
    first_coherence = float(abs(normalized[1]))
    status = "ok" if selected.lag_points > minimum else "minimum_window"
    z = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    if np.isfinite(selected.conditional_std):
        confidence_lower = selected.frequency - z * selected.conditional_std
        confidence_upper = selected.frequency + z * selected.conditional_std
    else:
        confidence_lower = confidence_upper = math.nan
    return (
        CoherenceCarrierEstimate(
            frequency=selected.frequency,
            frequency_sem=selected.frequency_sem,
            conditional_std=selected.conditional_std,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            selected_lag_points=selected.lag_points,
            selected_lag_time=selected.lag_points * dt,
            phase_fit_rms=selected.phase_fit_rms,
            first_lag_coherence=first_coherence,
            first_lag_phase=first_phase,
            nyquist_fraction=abs(first_phase) / math.pi,
            status=status,
        ),
        candidates,
    )
