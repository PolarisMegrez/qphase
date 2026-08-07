"""Stationarity, coherence, and Allan diagnostics for complex trajectories."""

from __future__ import annotations

import math
from typing import Any, ClassVar, cast

import numpy as np
from pydantic import Field, model_validator
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy
from qphase.core.protocols import PluginConfigBase

from ..utils import resolve_mode_columns
from .base import Analyzer
from .result import AnalysisResult

__all__ = ["TrajectoryDiagnostics", "TrajectoryDiagnosticsConfig"]


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
        return self


class TrajectoryDiagnostics(Analyzer):
    """Analyze saved complex trajectories without assuming a spectral line shape."""

    name: ClassVar[str] = "trajectory_diagnostics"
    description: ClassVar[str] = "Stationarity, coherence, and Allan diagnostics"
    config_schema: ClassVar[type[TrajectoryDiagnosticsConfig]] = (
        TrajectoryDiagnosticsConfig
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
        mode_results: dict[int, dict[str, Any]] = {}
        for mode, column in zip(config.modes, columns, strict=True):
            series = values[:, :, column]
            payload: dict[str, Any] = {
                "mean_amplitude_per_trajectory": np.mean(np.abs(series), axis=1),
                "mean_power_per_trajectory": np.mean(np.abs(series) ** 2, axis=1),
                "phase_increment": _phase_increment_summary(
                    series, dt, config.amplitude_floor
                ),
                "block_statistics": _block_statistics(
                    series,
                    dt,
                    config.block_durations,
                    config.amplitude_floor,
                ),
            }
            if config.coherence:
                payload["coherence"] = _coherence(
                    series,
                    dt,
                    max_lag=config.coherence_max_lag,
                    center=config.center_coherence,
                    keep_per_trajectory=config.keep_coherence_per_trajectory,
                )
            if config.allan:
                payload["allan"] = _allan_variance(
                    series,
                    dt,
                    taus=config.allan_taus,
                    points=config.allan_points,
                    min_windows=config.allan_min_windows,
                    amplitude_floor=config.amplitude_floor,
                )
            mode_results[int(mode)] = payload

        result = {
            "modes": list(config.modes),
            "t0": t0,
            "dt": dt,
            "n_traj": int(values.shape[0]),
            "n_samples": int(values.shape[1]),
            "mode_results": mode_results,
        }
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
    valid = (amplitude[:, 1:] > amplitude_floor) & (
        amplitude[:, :-1] > amplitude_floor
    )
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
            np.sum(np.abs(per_traj - mean[None, :]) ** 2, axis=0)
            / (n_traj - 1)
        ) / math.sqrt(n_traj)
    else:
        sem = np.full(n_lags, np.nan)
    result: dict[str, Any] = {
        "lag": np.arange(n_lags, dtype=float) * dt,
        "g1": mean,
        "g1_normalized": normalized,
        "g1_sem_magnitude": sem,
        "centered": center,
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
        raise ValueError(
            "trajectory is too short for the requested allan_min_windows"
        )
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
            standard_deviation = np.std(
                per_traj[finite[:, column], column], ddof=1
            )
            sem[column] = standard_deviation / math.sqrt(count)
    return {
        "tau": averaging_samples.astype(float) * dt,
        "angular_frequency_variance": mean,
        "angular_frequency_variance_sem": sem,
        "per_trajectory": per_traj,
        "valid_second_differences": valid_counts,
        "trajectory_sample_count": sample_count,
        "definition": "overlapping_phase_second_difference",
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
