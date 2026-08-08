"""Stationarity, coherence, and Allan diagnostics for complex trajectories.

Beyond raw summaries this analyser fits exponential/Gaussian/Kubo line-shape
models to the magnitude of the mean normalized first-order coherence (with an
interleaved train/holdout split) and can emit compact per-trajectory
per-block spectral features instead of full PSD cubes. Two opt-in whole-state
diagnostics use all recorded modes jointly: ``stationarity_details`` adjudicates
stationarity per trajectory from block-mean canonical ``R = alpha alpha^dagger``
features (block covariance, radial quantiles, head-tail drift, change-point
score), and ``matrix_projection`` projects the canonical R coordinates onto an
explicit real left vector against an explicit reference. Every output records
its units and conventions explicitly (see the ``*_unit``, ``sidedness`` and
``definition`` keys).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar, cast

import numpy as np
from pydantic import Field, model_validator
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy
from qphase.core.protocols import PluginConfigBase
from scipy.optimize import least_squares

from ..coordinates import (
    CANONICAL_COORDINATE_LAYOUT,
    R_CONVENTION,
    canonical_r_coordinates,
    canonical_vector,
)
from ..utils import resolve_mode_columns
from .base import (
    Analyzer,
    AnalyzerExecutionCapabilities,
    AnalyzerWorkspaceEstimate,
    AnalyzerWorkspaceRequest,
)
from .result import AnalysisResult
from .trajectory_diagnostic import (
    TrajectoryDiagnosticChild,
    TrajectoryDiagnosticContext,
)

__all__ = ["TrajectoryDiagnostics", "TrajectoryDiagnosticsConfig"]

# Minimum lag statistics required before any g1 line-shape model is fitted.
_G1_MIN_FINITE_LAGS = 8
_G1_MIN_SIGNAL_LAGS = 6
# A lag counts as signal while |g1| stays above this multiple of its SEM.
_G1_NOISE_FLOOR_FACTOR = 2.0
# Holdout RMS larger than factor * train RMS (plus an absolute guard) marks a
# fit as "holdout_mismatch"; the guard avoids flagging numerical noise.
_G1_HOLDOUT_MISMATCH_FACTOR = 3.0
_G1_HOLDOUT_MISMATCH_ATOL = 1.0e-6
# Holdout RMS values within this relative band of the best one are treated as
# ties and resolved toward the model with fewer parameters.
_G1_PREFERRED_RELATIVE_TOL = 0.02

# A block group must clear both thresholds to yield spectral features.
_BLOCK_SPECTRUM_MIN_SAMPLES = 8
_BLOCK_SPECTRUM_MIN_BLOCKS = 2

# Canonical R = alpha alpha^dagger coordinates (real, length n_modes**2) are
# shared with the observer subpackage via ``qphase_sde.coordinates``.

# Fewer blocks than this cannot support a meaningful covariance across blocks.
_STATIONARITY_MIN_BLOCKS = 4
_STATIONARITY_RADIAL_QUANTILES = (0.1, 0.5, 0.9)
# Above this covariance condition number the radial distance falls back to
# Euclidean (the Mahalanobis metric would be dominated by numerical noise).
_STATIONARITY_MAX_CONDITION = 1.0e6


class TrajectoryDiagnosticsConfig(PluginConfigBase):
    """Configuration for time-domain trajectory diagnostics."""

    modes: list[int] = Field(..., min_length=1, description="Physical mode indices")
    block_durations: list[float] = Field(
        default_factory=list,
        description="Non-overlapping block durations used for stationarity summaries",
    )
    coherence: bool = Field(True, description="Compute complex first-order coherence")
    coherence_max_lag: float | None = Field(
        None,
        gt=0.0,
        description="Largest coherence lag; None uses at most 4096 saved samples",
    )
    allan: bool = Field(True, description="Compute angular-frequency Allan variance")
    allan_taus: list[float] | None = Field(
        None,
        description="Requested Allan averaging times; None selects logarithmic times",
    )
    allan_points: int = Field(24, ge=2, le=256)
    allan_min_windows: int = Field(8, ge=1)
    amplitude_floor: float = Field(
        0.0,
        ge=0.0,
        description="Exclude Allan phase differences touching lower amplitudes",
    )
    center_coherence: bool = Field(
        False,
        description="Subtract each trajectory mean before computing coherence",
    )
    keep_coherence_per_trajectory: bool = Field(
        False,
        description="Retain per-trajectory coherence arrays in addition to summaries",
    )
    block_spectrum: bool = Field(
        False,
        description=(
            "Compute per-trajectory per-block periodogram features (peak, "
            "local HWHM, integrated power, wing fraction) for block_durations"
        ),
    )
    block_spectrum_wing_factor: float = Field(
        3.0,
        gt=1.0,
        description=(
            "Wings start outside the peak core halfwidth defined as "
            "max(wing_factor * resolution_angular, local_hwhm_angular)"
        ),
    )
    stationarity_details: bool = Field(
        False,
        description=(
            "Adjudicate stationarity per trajectory from block-mean canonical "
            "R features over block_durations: block covariance, radial "
            "quantiles, head-tail drift, and a change-point score"
        ),
    )
    matrix_projection: bool = Field(
        False,
        description=(
            "Project canonical R = alpha alpha^dagger coordinates (built from "
            "all recorded modes) onto matrix_projection_left_vector against "
            "matrix_projection_reference"
        ),
    )
    matrix_projection_reference: list[float] | None = Field(
        None,
        description=(
            "Flat real canonical R coordinates of length n_modes**2 "
            "subtracted before projection; None uses the zero matrix"
        ),
    )
    matrix_projection_left_vector: list[float] | None = Field(
        None,
        description=(
            "Flat real weights of length n_modes**2 applied to the canonical "
            "coordinates; None emits coordinate summaries without a projection"
        ),
    )
    matrix_projection_keep_coordinates: bool = Field(
        False,
        description=(
            "Retain full per-trajectory canonical coordinate trajectories "
            "(n_traj, n_time, n_modes**2) in the matrix_projection output"
        ),
    )

    @model_validator(mode="after")
    def validate_durations(self) -> TrajectoryDiagnosticsConfig:
        if any(value <= 0.0 for value in self.block_durations):
            raise ValueError("block_durations must contain positive values")
        if self.allan_taus is not None and any(
            value <= 0.0 for value in self.allan_taus
        ):
            raise ValueError("allan_taus must contain positive values")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("modes must not contain duplicates")
        vectors = (
            ("matrix_projection_reference", self.matrix_projection_reference),
            ("matrix_projection_left_vector", self.matrix_projection_left_vector),
        )
        for label, vector in vectors:
            if vector is None:
                continue
            if not self.matrix_projection:
                raise ValueError(f"{label} requires matrix_projection=True")
            root = math.isqrt(len(vector))
            if root < 1 or root * root != len(vector):
                raise ValueError(f"{label} must have length n_modes**2")
        reference = self.matrix_projection_reference
        left_vector = self.matrix_projection_left_vector
        if (
            reference is not None
            and left_vector is not None
            and len(reference) != len(left_vector)
        ):
            raise ValueError(
                "matrix_projection_reference and matrix_projection_left_vector "
                "must have matching lengths"
            )
        return self


class TrajectoryDiagnostics(Analyzer):
    """Analyze saved complex trajectories without assuming a spectral line shape."""

    name: ClassVar[str] = "trajectory_diagnostics"
    description: ClassVar[str] = "Stationarity, coherence, and Allan diagnostics"
    config_schema: ClassVar[type[TrajectoryDiagnosticsConfig]] = (
        TrajectoryDiagnosticsConfig
    )

    def capabilities(self) -> AnalyzerExecutionCapabilities:
        """Declare the current full-record host implementation."""
        return AnalyzerExecutionCapabilities(
            execution_location="host",
            requires_full_trajectory=True,
            supports_trajectory_batching=False,
            supports_time_streaming=False,
        )

    def estimate_workspace(
        self, request: AnalyzerWorkspaceRequest
    ) -> AnalyzerWorkspaceEstimate:
        """Estimate D2H, canonical-coordinate, and diagnostic temporaries."""
        materialization = (
            request.trajectory_bytes if request.backend_name == "cupy" else 0
        )
        coordinate_bytes = (
            request.trajectory_bytes * request.n_record_modes // 2
        )
        scratch = request.trajectory_bytes
        return AnalyzerWorkspaceEstimate(
            host_bytes=materialization + coordinate_bytes + scratch
        )

    def analyze(self, data: Any, backend: BackendBase) -> AnalysisResult:
        config = cast(TrajectoryDiagnosticsConfig, self.config)
        array = _trajectory_array(data)
        values = np.asarray(convert_to_numpy(array))
        if values.ndim != 3:
            raise ValueError(
                "trajectory diagnostics expects shape (n_traj, n_time, n_modes)"
            )
        if values.shape[1] < 3:
            raise ValueError("trajectory diagnostics requires at least three samples")
        if not np.iscomplexobj(values):
            raise ValueError("trajectory diagnostics requires complex mode amplitudes")

        dt = float(getattr(data, "dt", 1.0))
        t0 = float(getattr(data, "t0", 0.0))
        if dt <= 0.0:
            raise ValueError("trajectory sample spacing must be positive")

        columns = resolve_mode_columns(data, config.modes)
        diagnostic_context = TrajectoryDiagnosticContext(
            values=values,
            dt=dt,
            t0=t0,
            modes=tuple(config.modes),
            mode_columns=tuple(columns),
            coordinate_builder=canonical_r_coordinates,
        )
        result: dict[str, Any] = {
            "modes": list(config.modes),
            "t0": t0,
            "dt": dt,
            "n_traj": int(values.shape[0]),
            "n_samples": int(values.shape[1]),
            "mode_results": {},
        }
        children: list[TrajectoryDiagnosticChild] = [_ModeSummaryChild()]
        if config.coherence:
            children.append(_CoherenceChild())
        if config.allan:
            children.append(_AllanChild())
        if config.block_spectrum:
            children.append(_BlockSpectrumChild())
        if config.matrix_projection:
            children.append(_MatrixProjectionChild())
        if config.stationarity_details:
            children.append(_StationarityChild())
        for child in children:
            child.apply(diagnostic_context, result, config)
        return AnalysisResult(
            data_dict=result,
            meta={
                "modes": list(config.modes),
                "t0": t0,
                "dt": dt,
                "n_traj": int(values.shape[0]),
                "n_samples": int(values.shape[1]),
            },
        )


class _ModeSummaryChild:
    name = "mode_summary"

    def apply(
        self,
        context: TrajectoryDiagnosticContext,
        result: dict[str, Any],
        config: TrajectoryDiagnosticsConfig,
    ) -> None:
        for mode in context.modes:
            series = context.series(mode)
            result["mode_results"][mode] = {
                "mean_amplitude_per_trajectory": np.mean(np.abs(series), axis=1),
                "mean_power_per_trajectory": np.mean(np.abs(series) ** 2, axis=1),
                "phase_increment": _phase_increment_summary(
                    series, context.dt, config.amplitude_floor
                ),
                "block_statistics": _block_statistics(
                    series,
                    context.dt,
                    config.block_durations,
                    config.amplitude_floor,
                ),
            }


class _CoherenceChild:
    name = "coherence"

    def apply(
        self,
        context: TrajectoryDiagnosticContext,
        result: dict[str, Any],
        config: TrajectoryDiagnosticsConfig,
    ) -> None:
        for mode in context.modes:
            result["mode_results"][mode]["coherence"] = _coherence(
                context.series(mode),
                context.dt,
                max_lag=config.coherence_max_lag,
                center=config.center_coherence,
                keep_per_trajectory=config.keep_coherence_per_trajectory,
            )


class _AllanChild:
    name = "allan"

    def apply(
        self,
        context: TrajectoryDiagnosticContext,
        result: dict[str, Any],
        config: TrajectoryDiagnosticsConfig,
    ) -> None:
        for mode in context.modes:
            result["mode_results"][mode]["allan"] = _allan_variance(
                context.series(mode),
                context.dt,
                taus=config.allan_taus,
                points=config.allan_points,
                min_windows=config.allan_min_windows,
                amplitude_floor=config.amplitude_floor,
            )


class _BlockSpectrumChild:
    name = "block_spectrum"

    def apply(
        self,
        context: TrajectoryDiagnosticContext,
        result: dict[str, Any],
        config: TrajectoryDiagnosticsConfig,
    ) -> None:
        for mode in context.modes:
            result["mode_results"][mode]["block_spectrum"] = _block_spectrum(
                context.series(mode),
                context.dt,
                config.block_durations,
                wing_factor=config.block_spectrum_wing_factor,
            )


class _MatrixProjectionChild:
    name = "matrix_projection"

    def apply(
        self,
        context: TrajectoryDiagnosticContext,
        result: dict[str, Any],
        config: TrajectoryDiagnosticsConfig,
    ) -> None:
        result[self.name] = _matrix_projection(
            context.values,
            context.canonical_coordinates(),
            context.dt,
            config,
        )


class _StationarityChild:
    name = "stationarity"

    def apply(
        self,
        context: TrajectoryDiagnosticContext,
        result: dict[str, Any],
        config: TrajectoryDiagnosticsConfig,
    ) -> None:
        result[self.name] = _stationarity_details(
            context.canonical_coordinates(),
            context.dt,
            config.block_durations,
            n_modes=context.values.shape[-1],
        )


def _trajectory_array(data: Any) -> Any:
    if hasattr(data, "ndim") and hasattr(data, "shape"):
        return data
    wrapped = getattr(data, "data", None)
    if hasattr(wrapped, "ndim") and hasattr(wrapped, "shape"):
        return wrapped
    raise TypeError("trajectory diagnostics requires array-like trajectory data")


def _duration_samples(duration: float, dt: float, label: str) -> int:
    raw = float(duration) / dt
    samples = round(raw)
    if samples < 1 or not math.isclose(raw, samples, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError(
            f"{label}={duration:.12g} must be a positive multiple of dt={dt:.12g}"
        )
    return int(samples)


def _block_statistics(
    series: np.ndarray,
    dt: float,
    durations: list[float],
    amplitude_floor: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    n_traj, n_time = series.shape
    for duration in durations:
        block_samples = _duration_samples(duration, dt, "block duration")
        n_blocks = n_time // block_samples
        if n_blocks < 1:
            raise ValueError(
                f"block duration {duration:.12g} exceeds the observed trajectory"
            )
        trimmed = series[:, : n_blocks * block_samples]
        blocks = trimmed.reshape(n_traj, n_blocks, block_samples)
        amplitude = np.abs(blocks)
        if block_samples > 1:
            increments = np.angle(blocks[:, :, 1:] * np.conj(blocks[:, :, :-1]))
            increments = increments / dt
            valid = (amplitude[:, :, 1:] > amplitude_floor) & (
                amplitude[:, :, :-1] > amplitude_floor
            )
            frequency = _masked_mean(increments, valid, axis=2)
        else:
            frequency = np.full((n_traj, n_blocks), np.nan)
        results.append(
            {
                "duration": block_samples * dt,
                "samples": block_samples,
                "n_blocks": n_blocks,
                "mean_complex": np.mean(blocks, axis=2),
                "mean_amplitude": np.mean(amplitude, axis=2),
                "mean_power": np.mean(amplitude**2, axis=2),
                "mean_angular_frequency": frequency,
                "frequency_unit": "angular",
            }
        )
    return results


def _phase_increment_summary(
    series: np.ndarray,
    dt: float,
    amplitude_floor: float,
) -> dict[str, Any]:
    amplitude = np.abs(series)
    increments = np.angle(series[:, 1:] * np.conj(series[:, :-1]))
    valid = (amplitude[:, 1:] > amplitude_floor) & (amplitude[:, :-1] > amplitude_floor)
    absolute = np.abs(increments)
    return {
        "mean_angular_frequency_per_trajectory": _masked_mean(
            increments / dt, valid, axis=1
        ),
        "max_abs_phase_step_per_trajectory": np.max(
            np.where(valid, absolute, np.nan), axis=1
        ),
        "near_nyquist_fraction_per_trajectory": _masked_mean(
            (absolute >= 0.9 * np.pi).astype(float), valid, axis=1
        ),
        "near_nyquist_threshold": 0.9 * np.pi,
    }


def _coherence(
    series: np.ndarray,
    dt: float,
    *,
    max_lag: float | None,
    center: bool,
    keep_per_trajectory: bool,
) -> dict[str, Any]:
    n_traj, n_time = series.shape
    if max_lag is None:
        n_lags = min(n_time, 4097)
    else:
        n_lags = min(n_time, _duration_samples(max_lag, dt, "coherence_max_lag") + 1)
    n_fft = 1 << (2 * n_time - 1).bit_length()
    per_traj = np.empty((n_traj, n_lags), dtype=np.complex128)
    denominator = np.arange(n_time, n_time - n_lags, -1, dtype=float)
    for index, values in enumerate(series):
        work = values - np.mean(values) if center else values
        spectrum = np.fft.fft(work, n=n_fft)
        correlation = np.fft.ifft(spectrum * np.conj(spectrum))[:n_lags]
        per_traj[index] = correlation / denominator

    mean = np.mean(per_traj, axis=0)
    normalizer = mean[0]
    normalized = (
        mean / normalizer
        if abs(normalizer) > np.finfo(float).tiny
        else np.full_like(mean, np.nan)
    )
    if n_traj > 1:
        sem = np.sqrt(
            np.sum(np.abs(per_traj - mean[None, :]) ** 2, axis=0) / (n_traj - 1)
        ) / math.sqrt(n_traj)
    else:
        sem = np.full(n_lags, np.nan)
    lag = np.arange(n_lags, dtype=float) * dt
    result: dict[str, Any] = {
        "quantity": "first_order_coherence",
        "normalization": "g1(0)",
        "lag_unit": "seconds",
        "lag": lag,
        "g1": mean,
        "g1_normalized": normalized,
        "g1_sem_magnitude": sem,
        "centered": center,
        "model_fits": _fit_g1_models(lag, normalized, sem),
    }
    if keep_per_trajectory:
        result["g1_per_trajectory"] = per_traj
    return result


def _allan_variance(
    series: np.ndarray,
    dt: float,
    *,
    taus: list[float] | None,
    points: int,
    min_windows: int,
    amplitude_floor: float,
) -> dict[str, Any]:
    n_traj, n_time = series.shape
    max_m = (n_time - min_windows) // 2
    if max_m < 1:
        raise ValueError("trajectory is too short for the requested allan_min_windows")
    if taus is None:
        candidates = np.geomspace(1, max_m, num=points)
        averaging_samples = np.unique(np.rint(candidates).astype(int))
    else:
        averaging_samples = np.unique(
            np.asarray(
                [_duration_samples(value, dt, "Allan tau") for value in taus],
                dtype=int,
            )
        )
        if averaging_samples[-1] > max_m:
            raise ValueError(
                "an Allan tau leaves fewer than allan_min_windows second differences"
            )

    amplitude = np.abs(series)
    phase_increments = np.angle(series[:, 1:] * np.conj(series[:, :-1]))
    valid_increments = (amplitude[:, 1:] > amplitude_floor) & (
        amplitude[:, :-1] > amplitude_floor
    )
    phase = np.concatenate(
        (
            np.zeros((n_traj, 1), dtype=float),
            np.cumsum(np.where(valid_increments, phase_increments, 0.0), axis=1),
        ),
        axis=1,
    )
    valid_cumulative = np.concatenate(
        (
            np.zeros((n_traj, 1), dtype=int),
            np.cumsum(valid_increments, axis=1),
        ),
        axis=1,
    )
    per_traj = np.full((n_traj, len(averaging_samples)), np.nan, dtype=float)
    valid_counts = np.zeros((n_traj, len(averaging_samples)), dtype=int)
    for column, m in enumerate(averaging_samples):
        delta = phase[:, 2 * m :] - 2.0 * phase[:, m:-m] + phase[:, : -2 * m]
        valid = valid_cumulative[:, 2 * m :] - valid_cumulative[:, : -2 * m]
        valid = valid == 2 * m
        valid_counts[:, column] = np.sum(valid, axis=1)
        mean_square = _masked_mean(delta**2, valid, axis=1)
        tau = m * dt
        per_traj[:, column] = mean_square / (2.0 * tau**2)

    finite = np.isfinite(per_traj)
    sample_count = np.sum(finite, axis=0)
    mean = _masked_mean(per_traj, finite, axis=0)
    sem = np.full(len(averaging_samples), np.nan, dtype=float)
    for column, count in enumerate(sample_count):
        if count > 1:
            standard_deviation = np.std(per_traj[finite[:, column], column], ddof=1)
            sem[column] = standard_deviation / math.sqrt(count)
    return {
        "quantity": "allan_variance",
        "variable": "angular_frequency",
        "tau_unit": "seconds",
        "tau": averaging_samples.astype(float) * dt,
        "angular_frequency_variance": mean,
        "angular_frequency_variance_sem": sem,
        "per_trajectory": per_traj,
        "valid_second_differences": valid_counts,
        "trajectory_sample_count": sample_count,
        "definition": "overlapping_phase_second_difference",
    }


def _g1_exponential(lag: np.ndarray, gamma: float) -> np.ndarray:
    """White frequency noise limit: |g1(tau)| = exp(-gamma * tau)."""
    return np.exp(-gamma * lag)


def _g1_gaussian(lag: np.ndarray, gamma: float) -> np.ndarray:
    """Fast modulation/static disorder limit: |g1(tau)| = exp(-(gamma*tau)^2)."""
    return np.exp(-((gamma * lag) ** 2))


def _g1_kubo(lag: np.ndarray, delta: float, tau_c: float) -> np.ndarray:
    """Exact Kubo |g1| for Ornstein-Uhlenbeck frequency modulation.

    |g1(tau)| = exp(-delta^2 tau_c^2 (exp(-tau/tau_c) + tau/tau_c - 1));
    the short-time limit is Gaussian with rate delta/sqrt(2) and the
    long-time limit is exponential with rate delta^2 * tau_c.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = lag / tau_c
        exponent = -((delta * tau_c) ** 2) * (np.expm1(-scaled) + scaled)
    return np.exp(exponent)


_G1_MODEL_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "exponential": _g1_exponential,
    "gaussian": _g1_gaussian,
    "kubo": _g1_kubo,
}
_G1_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "exponential": ("gamma",),
    "gaussian": ("gamma",),
    "kubo": ("delta", "tau_c"),
}
_G1_PARAM_UNITS: dict[str, dict[str, str]] = {
    "exponential": {"gamma": "per_second"},
    "gaussian": {"gamma": "per_second"},
    "kubo": {"delta": "per_second", "tau_c": "seconds"},
}


def _g1_e_fold_rate(lag: np.ndarray, magnitude: np.ndarray) -> float:
    """Estimate an initial decay rate from the 1/e crossing of |g1|."""
    target = 1.0 / math.e
    below = np.nonzero(magnitude < target)[0]
    if below.size and below[0] > 0:
        index = int(below[0])
        upper = float(magnitude[index - 1])
        lower = float(magnitude[index])
        fraction = (upper - target) / (upper - lower) if upper != lower else 0.0
        crossing = float(lag[index - 1]) + fraction * float(lag[index] - lag[index - 1])
        if crossing > 0.0:
            return 1.0 / crossing
    span = float(lag[-1]) if lag[-1] > 0.0 else 1.0
    return 1.0 / span


def _g1_initial_guess(
    name: str,
    lag: np.ndarray,
    magnitude: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Heuristic starting parameters and positivity bounds for one model."""
    rate = _g1_e_fold_rate(lag, magnitude)
    if name == "kubo":
        delta = rate
        usable = np.nonzero(
            (lag > 0.0) & (magnitude > 1.0e-3) & (magnitude < 1.0 - 1.0e-9)
        )[0]
        if usable.size:
            first = int(usable[0])
            # Short-time Kubo limit: |g1| ~ exp(-(delta * tau)^2 / 2).
            delta = math.sqrt(-2.0 * math.log(float(magnitude[first]))) / float(
                lag[first]
            )
        delta = max(delta, 1.0e-12)
        # Long-time Kubo limit decays at delta^2 * tau_c.
        tau_c = max(rate / (delta * delta), np.finfo(float).eps)
        return np.array([delta, tau_c]), (
            np.array([0.0, np.finfo(float).tiny]),
            np.array([np.inf, np.inf]),
        )
    return np.array([rate]), (np.array([0.0]), np.array([np.inf]))


def _fit_g1_model(
    name: str,
    lag_train: np.ndarray,
    magnitude_train: np.ndarray,
    lag_holdout: np.ndarray,
    magnitude_holdout: np.ndarray,
) -> dict[str, Any]:
    """Least-squares fit of one line-shape model to |g1|; never raises."""
    entry: dict[str, Any] = {
        "method": "least_squares|g1|",
        "lag_unit": "seconds",
        "param_units": _G1_PARAM_UNITS[name],
        "n_train": int(lag_train.size),
        "n_holdout": int(lag_holdout.size),
    }
    model = _G1_MODEL_FUNCTIONS[name]
    initial, bounds = _g1_initial_guess(name, lag_train, magnitude_train)
    try:
        result = least_squares(
            lambda params: model(lag_train, *params) - magnitude_train,
            initial,
            bounds=bounds,
            method="trf",
            max_nfev=10_000,
        )
        params = np.asarray(result.x, dtype=float)
        if not result.success or not np.all(np.isfinite(params)):
            raise RuntimeError(str(result.message))
    except Exception as exc:
        entry["status"] = "fit_failed"
        entry["message"] = str(exc)
        return entry

    train_residual = model(lag_train, *params) - magnitude_train
    holdout_residual = model(lag_holdout, *params) - magnitude_holdout
    train_rms = float(np.sqrt(np.mean(train_residual**2)))
    holdout_rms = float(np.sqrt(np.mean(holdout_residual**2)))
    mismatch = (
        holdout_rms
        > _G1_HOLDOUT_MISMATCH_FACTOR * train_rms + _G1_HOLDOUT_MISMATCH_ATOL
    )
    entry.update(
        {
            "status": "holdout_mismatch" if mismatch else "ok",
            "params": {
                key: float(value)
                for key, value in zip(_G1_PARAM_NAMES[name], params, strict=True)
            },
            "train_rms_residual": train_rms,
            "train_max_residual": float(np.max(np.abs(train_residual))),
            "holdout_rms_residual": holdout_rms,
            "holdout_max_residual": float(np.max(np.abs(holdout_residual))),
        }
    )
    return entry


def _fit_g1_models(
    lag: np.ndarray,
    g1_normalized: np.ndarray,
    sem: np.ndarray,
) -> dict[str, Any]:
    """Fit exponential/Gaussian/Kubo decays to |g1_normalized| with holdout.

    The finite lags are split by interleaving (even positions train, odd
    positions holdout) so both subsets cover the full decay range. Models are
    compared by holdout RMS; near-ties resolve toward fewer parameters.
    """
    magnitude = np.abs(g1_normalized)
    finite = np.isfinite(magnitude)
    indices = np.nonzero(finite)[0]
    floor = np.where(np.isfinite(sem), _G1_NOISE_FLOOR_FACTOR * sem, 0.0)
    n_signal = int(np.count_nonzero(finite & (magnitude > floor)))
    train = indices[0::2]
    holdout = indices[1::2]
    summary: dict[str, Any] = {
        "fit_target": "abs(g1_normalized)",
        "holdout_split": "interleaved_even_train",
        "lag_unit": "seconds",
        "lag_step_seconds": float(lag[1] - lag[0]) if lag.size > 1 else math.nan,
        "lag_window_seconds": [float(lag[0]), float(lag[-1])],
        "n_lags": int(lag.size),
        "n_finite": int(indices.size),
        "n_signal": n_signal,
        "min_finite_lags": _G1_MIN_FINITE_LAGS,
        "min_signal_lags": _G1_MIN_SIGNAL_LAGS,
        "noise_floor": f"{_G1_NOISE_FLOOR_FACTOR}*g1_sem_magnitude",
        "holdout_mismatch_factor": _G1_HOLDOUT_MISMATCH_FACTOR,
    }
    insufficient = indices.size < _G1_MIN_FINITE_LAGS or n_signal < _G1_MIN_SIGNAL_LAGS
    models: dict[str, dict[str, Any]] = {}
    for name in _G1_MODEL_FUNCTIONS:
        if insufficient:
            models[name] = {
                "status": "insufficient_data",
                "method": "least_squares|g1|",
                "lag_unit": "seconds",
                "param_units": _G1_PARAM_UNITS[name],
                "n_train": int(train.size),
                "n_holdout": int(holdout.size),
            }
        else:
            models[name] = _fit_g1_model(
                name,
                lag[train],
                magnitude[train],
                lag[holdout],
                magnitude[holdout],
            )
    summary["models"] = models

    converged = [
        (name, entry["holdout_rms_residual"])
        for name, entry in models.items()
        if entry["status"] in ("ok", "holdout_mismatch")
    ]
    if converged:
        best = min(rms for _, rms in converged)
        # Keep the fixed model order so equal-parameter ties are deterministic.
        tied = [
            name
            for name, rms in converged
            if rms <= best * (1.0 + _G1_PREFERRED_RELATIVE_TOL)
        ]
        summary["preferred_model"] = min(
            tied, key=lambda name: len(_G1_PARAM_NAMES[name])
        )
        summary["preferred_model_rule"] = (
            "minimum holdout RMS; ties within "
            f"{_G1_PREFERRED_RELATIVE_TOL:.0%} relative resolve toward "
            "fewer parameters"
        )
    return summary


def _parabolic_offset(values: np.ndarray, index: int) -> float:
    """Return the sub-bin parabolic-interpolation offset of a local maximum."""
    size = values.size
    left = float(values[(index - 1) % size])
    center = float(values[index])
    right = float(values[(index + 1) % size])
    denominator = left - 2.0 * center + right
    if denominator >= 0.0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))


def _linear_crossing(
    f0: float,
    v0: float,
    f1: float,
    v1: float,
    target: float,
) -> float:
    """Frequency at which the linear segment (f0,v0)-(f1,v1) reaches target."""
    return f0 + (target - v0) * (f1 - f0) / (v1 - v0)


def _half_power_hwhm(
    frequencies: np.ndarray,
    row: np.ndarray,
    peak_index: int,
) -> float:
    """Half width at half maximum from linearly interpolated crossings.

    If a half-power crossing is not found before the band edge, the edge is
    used, so the returned width is a lower bound in that case.
    """
    target = 0.5 * float(row[peak_index])
    left = peak_index
    while left > 0 and row[left] >= target:
        left -= 1
    if row[left] >= target:
        cross_left = float(frequencies[0])
    else:
        cross_left = _linear_crossing(
            float(frequencies[left]),
            float(row[left]),
            float(frequencies[left + 1]),
            float(row[left + 1]),
            target,
        )
    last = row.size - 1
    right = peak_index
    while right < last and row[right] >= target:
        right += 1
    if row[right] >= target:
        cross_right = float(frequencies[last])
    else:
        cross_right = _linear_crossing(
            float(frequencies[right - 1]),
            float(row[right - 1]),
            float(frequencies[right]),
            float(row[right]),
            target,
        )
    return 0.5 * (cross_right - cross_left)


def _block_spectrum_features(
    blocks: np.ndarray,
    dt: float,
    wing_factor: float,
) -> dict[str, Any]:
    """Compact rectangular-window periodogram features per trajectory/block.

    The PSD normalization ``abs(fft(x))**2 * dt / (2*pi*n_samples)`` makes the
    bin sum times the resolution equal to the mean squared amplitude
    (Parseval), which is reported as ``integrated_power``.
    """
    n_traj, n_blocks, n_samples = blocks.shape
    resolution = 2.0 * np.pi / (n_samples * dt)
    transformed = np.fft.fft(blocks, axis=-1)
    psd = np.abs(transformed) ** 2 * (dt / (2.0 * np.pi * n_samples))
    psd = np.fft.fftshift(psd, axes=-1)
    frequencies = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(n_samples, d=dt))
    integrated = np.mean(np.abs(blocks) ** 2, axis=-1)

    shape = (n_traj, n_blocks)
    peak_frequency = np.full(shape, np.nan)
    peak_power = np.full(shape, np.nan)
    hwhm = np.full(shape, np.nan)
    core_halfwidth = np.full(shape, np.nan)
    wing_fraction = np.full(shape, np.nan)
    resolution_limited = np.zeros(shape, dtype=bool)
    for trajectory in range(n_traj):
        for block in range(n_blocks):
            row = psd[trajectory, block]
            total = float(np.sum(row))
            if integrated[trajectory, block] <= 0.0 or total <= 0.0:
                continue
            peak_index = int(np.argmax(row))
            peak_f = (
                float(frequencies[peak_index])
                + _parabolic_offset(row, peak_index) * resolution
            )
            width = _half_power_hwhm(frequencies, row, peak_index)
            core = max(wing_factor * resolution, width)
            outside = np.abs(frequencies - peak_f) > core
            peak_frequency[trajectory, block] = peak_f
            peak_power[trajectory, block] = float(row[peak_index])
            hwhm[trajectory, block] = width
            core_halfwidth[trajectory, block] = core
            wing_fraction[trajectory, block] = float(np.sum(row[outside]) / total)
            resolution_limited[trajectory, block] = width <= resolution
    return {
        "status": "ok",
        "peak_angular_frequency": peak_frequency,
        "peak_power": peak_power,
        "local_hwhm_angular": hwhm,
        "resolution_limited": resolution_limited,
        "integrated_power": integrated,
        "wing_fraction": wing_fraction,
        "core_halfwidth_angular": core_halfwidth,
        "frequency_unit": "angular",
        "sidedness": "two-sided",
        "window": "rectangular",
        "resolution_angular": resolution,
        "psd_normalization": "abs(fft(x))**2 * dt / (2*pi*n_samples)",
        "peak_interpolation": "parabolic",
        "wing_factor": float(wing_factor),
        "wing_core_definition": (
            "max(wing_factor * resolution_angular, local_hwhm_angular)"
        ),
        "resolution_limited_definition": ("local_hwhm_angular <= resolution_angular"),
    }


def _block_spectrum(
    series: np.ndarray,
    dt: float,
    durations: list[float],
    *,
    wing_factor: float,
) -> dict[str, Any]:
    """Per-trajectory per-block spectral features for each requested duration."""
    if not durations:
        return {"status": "no_blocks_configured"}
    n_traj, n_time = series.shape
    entries: list[dict[str, Any]] = []
    for duration in durations:
        block_samples = _duration_samples(duration, dt, "block duration")
        n_blocks = n_time // block_samples
        entry: dict[str, Any] = {
            "duration": block_samples * dt,
            "samples": block_samples,
            "n_blocks": n_blocks,
            "min_samples": _BLOCK_SPECTRUM_MIN_SAMPLES,
            "min_blocks": _BLOCK_SPECTRUM_MIN_BLOCKS,
        }
        if (
            block_samples < _BLOCK_SPECTRUM_MIN_SAMPLES
            or n_blocks < _BLOCK_SPECTRUM_MIN_BLOCKS
        ):
            entry["status"] = "insufficient_data"
        else:
            trimmed = series[:, : n_blocks * block_samples]
            blocks = trimmed.reshape(n_traj, n_blocks, block_samples)
            entry.update(_block_spectrum_features(blocks, dt, wing_factor))
        entries.append(entry)
    return {"status": "ok", "entries": entries}


def _matrix_projection(
    values: np.ndarray,
    coordinates: np.ndarray,
    dt: float,
    config: TrajectoryDiagnosticsConfig,
) -> dict[str, Any]:
    """Whole-state projection c(t) = left_vector . (vec(R(t)) - vec(reference)).

    Canonical R coordinates are built from all recorded modes (never sliced
    per mode). With no ``left_vector`` only coordinate summaries are emitted;
    with no ``reference`` the zero matrix is used. When ``block_spectrum`` is
    also enabled the real projection series is fed through the same
    per-trajectory per-block periodogram features as the mode series.
    """
    n_traj, n_time, n_modes = values.shape
    n_coordinates = n_modes * n_modes
    reference = np.zeros(n_coordinates)
    if config.matrix_projection_reference is not None:
        reference = canonical_vector(
            config.matrix_projection_reference,
            n_coordinates,
            "matrix_projection_reference",
        )
    left_vector = None
    if config.matrix_projection_left_vector is not None:
        left_vector = canonical_vector(
            config.matrix_projection_left_vector,
            n_coordinates,
            "matrix_projection_left_vector",
        )
    result: dict[str, Any] = {
        "status": "ok",
        "quantity": "matrix_projection",
        "coordinate_layout": CANONICAL_COORDINATE_LAYOUT,
        "convention": R_CONVENTION,
        "n_modes": n_modes,
        "n_coordinates": n_coordinates,
        "reference_vector": reference,
        "reference_norm": float(np.linalg.norm(reference)),
        "time_unit": "seconds",
        "dt": dt,
        "mean_coordinates_per_trajectory": np.mean(coordinates, axis=1),
    }
    if left_vector is not None:
        projection = (coordinates - reference[None, None, :]) @ left_vector
        result.update(
            {
                "left_vector": left_vector,
                "left_vector_norm": float(np.linalg.norm(left_vector)),
                "projection_definition": (
                    "c(t) = left_vector . (vec(R(t)) - vec(reference))"
                ),
                "projection_per_trajectory": projection,
                "projection_mean_per_trajectory": np.mean(projection, axis=1),
                "projection_std_per_trajectory": np.std(projection, axis=1),
                "projection_min_per_trajectory": np.min(projection, axis=1),
                "projection_max_per_trajectory": np.max(projection, axis=1),
            }
        )
        if config.block_spectrum:
            result["block_spectrum"] = _block_spectrum(
                projection,
                dt,
                config.block_durations,
                wing_factor=config.block_spectrum_wing_factor,
            )
    if config.matrix_projection_keep_coordinates:
        result["coordinates_per_trajectory"] = coordinates
    return result


def _stationarity_details(
    coordinates: np.ndarray,
    dt: float,
    durations: list[float],
    *,
    n_modes: int,
) -> dict[str, Any]:
    """Per-trajectory stationarity adjudication for each requested duration.

    The feature vector is the within-block mean of the canonical R
    coordinates (richer than the block-mean complex amplitude: it carries
    second moments), trimmed to whole non-overlapping blocks. The trajectory
    axis is preserved everywhere; entries with too few blocks report a
    structured ``insufficient_data`` status instead of raising.
    """
    if not durations:
        return {"status": "no_blocks_configured"}
    n_traj, n_time, n_coordinates = coordinates.shape
    entries: list[dict[str, Any]] = []
    for duration in durations:
        block_samples = _duration_samples(duration, dt, "block duration")
        n_blocks = n_time // block_samples
        entry: dict[str, Any] = {
            "duration": block_samples * dt,
            "samples": block_samples,
            "n_blocks": n_blocks,
            "min_blocks": _STATIONARITY_MIN_BLOCKS,
        }
        if n_blocks < _STATIONARITY_MIN_BLOCKS:
            entry["status"] = "insufficient_data"
        else:
            trimmed = coordinates[:, : n_blocks * block_samples]
            blocks = trimmed.reshape(
                n_traj, n_blocks, block_samples, n_coordinates
            )
            features = np.mean(blocks, axis=2)
            entry.update(_stationarity_features(features))
        entries.append(entry)
    return {
        "status": "ok",
        "feature": "block_mean_canonical_r",
        "coordinate_layout": CANONICAL_COORDINATE_LAYOUT,
        "convention": R_CONVENTION,
        "n_features": n_modes * n_modes,
        "entries": entries,
    }


def _stationarity_features(features: np.ndarray) -> dict[str, Any]:
    """Covariance, radial quantiles, drift, and change-point per trajectory.

    ``features`` has shape ``(n_traj, n_blocks, n_features)``. Radial
    distances from the per-trajectory median block feature use the
    Mahalanobis metric of the block covariance when its condition number is
    below ``_STATIONARITY_MAX_CONDITION`` and Euclidean distances otherwise.
    The change-point statistic is a two-sample separation with per-split
    pooled variance, combined in quadrature across features and maximized
    over split points.
    """
    n_traj, n_blocks, n_features = features.shape
    covariance = np.empty((n_traj, n_features, n_features))
    median = np.median(features, axis=1)
    centered = features - median[:, None, :]
    metric: list[str] = []
    radial = np.empty((n_traj, n_blocks))
    for trajectory in range(n_traj):
        cov = np.atleast_2d(np.cov(features[trajectory], rowvar=False, ddof=1))
        covariance[trajectory] = cov
        condition = float(np.linalg.cond(cov))
        if np.isfinite(condition) and condition < _STATIONARITY_MAX_CONDITION:
            solved = np.linalg.solve(np.linalg.cholesky(cov), centered[trajectory].T)
            radial[trajectory] = np.sqrt(np.sum(solved**2, axis=0))
            metric.append("mahalanobis")
        else:
            radial[trajectory] = np.sqrt(np.sum(centered[trajectory] ** 2, axis=1))
            metric.append("euclidean")
    quantiles = np.quantile(
        radial, list(_STATIONARITY_RADIAL_QUANTILES), axis=1
    ).transpose()

    side = max(1, n_blocks // 4)
    head = features[:, :side]
    tail = features[:, -side:]
    delta = np.mean(head, axis=1) - np.mean(tail, axis=1)
    if side > 1:
        pooled_variance = 0.5 * (
            np.var(head, axis=1, ddof=1) + np.var(tail, axis=1, ddof=1)
        )
    else:
        pooled_variance = np.var(features, axis=1, ddof=1)
    standard_error = np.sqrt(pooled_variance * 2.0 / side)
    ratio = np.zeros_like(delta)
    np.divide(np.abs(delta), standard_error, out=ratio, where=standard_error > 0.0)

    counts = np.arange(1, n_blocks, dtype=float)
    right_counts = n_blocks - counts
    cum1 = np.cumsum(features, axis=1)
    cum2 = np.cumsum(features**2, axis=1)
    left_mean = cum1[:, :-1] / counts[None, :, None]
    right_mean = (cum1[:, -1:] - cum1[:, :-1]) / right_counts[None, :, None]
    left_var = cum2[:, :-1] / counts[None, :, None] - left_mean**2
    right_var = (cum2[:, -1:] - cum2[:, :-1]) / right_counts[
        None, :, None
    ] - right_mean**2
    pooled = (
        (counts - 1.0)[None, :, None] * left_var
        + (right_counts - 1.0)[None, :, None] * right_var
    ) / (n_blocks - 2.0)
    weight = counts * right_counts / n_blocks
    scaled = np.zeros_like(left_mean)
    np.divide(
        (left_mean - right_mean) ** 2,
        pooled,
        out=scaled,
        where=pooled > np.finfo(float).tiny,
    )
    separation = np.sqrt(np.sum(scaled * weight[None, :, None], axis=2))
    split = np.argmax(separation, axis=1)
    score = separation[np.arange(n_traj), split]

    return {
        "status": "ok",
        "block_covariance": covariance,
        "covariance_estimator": "sample covariance across blocks (ddof=1)",
        "radial_quantiles": quantiles,
        "radial_quantile_levels": list(_STATIONARITY_RADIAL_QUANTILES),
        "radial_center": "per-trajectory median of block features",
        "radial_distance_metric_per_trajectory": metric,
        "radial_distance_definition": (
            "mahalanobis when the covariance condition number is below "
            f"{_STATIONARITY_MAX_CONDITION:.3g}, euclidean otherwise"
        ),
        "head_tail_drift": {
            "delta": delta,
            "standard_error": standard_error,
            "significance_ratio": np.max(ratio, axis=1),
            "blocks_per_side": side,
            "definition": (
                "mean(first K blocks) - mean(last K blocks) with K = "
                "max(1, n_blocks // 4); significance_ratio is the max over "
                "features of |delta| / standard_error with "
                "standard_error**2 = pooled head/tail variance * 2/K "
                "(all-block variance when K = 1)"
            ),
        },
        "change_point": {
            "score": score,
            "block_index": (split + 1).astype(int),
            "statistic": "two_sample_t_quadrature_max",
            "definition": (
                "max over splits of sqrt(sum_features (left_mean - "
                "right_mean)**2 * (n_left*n_right/n_blocks) / "
                "pooled_variance) with per-split pooled variance; "
                "block_index is the first block of the right segment"
            ),
        },
    }


def _masked_mean(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    axis: int,
) -> np.ndarray:
    count = np.sum(mask, axis=axis)
    total = np.sum(np.where(mask, values, 0.0), axis=axis)
    result = np.full(np.shape(total), np.nan, dtype=np.result_type(total, float))
    np.divide(total, count, out=result, where=count > 0)
    return result
