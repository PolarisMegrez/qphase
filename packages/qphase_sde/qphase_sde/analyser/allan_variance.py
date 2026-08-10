"""Focused, trajectory-batchable Allan variance analyser."""

from __future__ import annotations

import math
from typing import Any, ClassVar, cast

import numpy as np
from pydantic import Field, model_validator
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy
from qphase.core.protocols import PluginConfigBase

from ..utils import resolve_mode_columns
from .allan_statistics import calculate_allan_variance, summarize_trajectories
from .base import (
    Analyzer,
    AnalyzerExecutionCapabilities,
    AnalyzerWorkspaceEstimate,
    AnalyzerWorkspaceRequest,
)
from .result import AnalysisResult

__all__ = ["AllanVarianceAnalyzer", "AllanVarianceConfig"]


class AllanVarianceConfig(PluginConfigBase):
    """Configuration for focused angular-frequency Allan statistics."""

    modes: list[int] = Field(..., min_length=1)
    taus: list[float] | None = Field(
        None, description="Requested averaging times; None selects a logarithmic grid"
    )
    points: int = Field(40, ge=2, le=256)
    min_windows: int = Field(8, ge=1)
    min_independent_windows: int = Field(
        4,
        ge=1,
        description="Minimum nominal non-overlapping windows per trajectory",
    )
    amplitude_floor: float = Field(0.0, ge=0.0)
    transfer_chunk_samples: int = Field(
        8192,
        ge=1,
        description="Maximum saved samples copied from a device at once",
    )

    @model_validator(mode="after")
    def validate_values(self) -> AllanVarianceConfig:
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("modes must not contain duplicates")
        if self.taus is not None and any(value <= 0.0 for value in self.taus):
            raise ValueError("taus must contain positive values")
        return self


class AllanVarianceAnalyzer(Analyzer):
    """Compute compact Allan statistics without retaining complex trajectories."""

    name: ClassVar[str] = "allan_variance"
    description: ClassVar[str] = (
        "Trajectory-batchable angular-frequency Allan variance statistics"
    )
    config_schema: ClassVar[type[AllanVarianceConfig]] = AllanVarianceConfig

    def capabilities(self) -> AnalyzerExecutionCapabilities:
        return AnalyzerExecutionCapabilities(
            execution_location="host",
            requires_full_trajectory=True,
            supports_trajectory_batching=True,
            supports_time_streaming=False,
        )

    def estimate_workspace(
        self, request: AnalyzerWorkspaceRequest
    ) -> AnalyzerWorkspaceEstimate:
        mode_bytes = request.trajectory_bytes // max(request.n_record_modes, 1)
        chunk = min(
            cast(AllanVarianceConfig, self.config).transfer_chunk_samples,
            request.saved_samples,
        )
        transfer_bytes = (
            request.n_traj * chunk * 2 * request.real_itemsize
            if request.backend_name == "cupy"
            else 0
        )
        return AnalyzerWorkspaceEstimate(
            device_bytes=transfer_bytes,
            host_bytes=5 * mode_bytes,
        )

    def analyze(self, data: Any, backend: BackendBase) -> AnalysisResult:
        config = cast(AllanVarianceConfig, self.config)
        array = getattr(data, "data", data)
        if (
            not hasattr(array, "ndim")
            or array.ndim != 3
            or not _is_complex_array(array)
        ):
            raise ValueError(
                "allan_variance expects complex shape (n_traj, n_time, n_modes)"
            )
        if array.shape[1] < 3:
            raise ValueError("allan_variance requires at least three samples")
        dt = float(getattr(data, "dt", 1.0))
        t0 = float(getattr(data, "t0", 0.0))
        if dt <= 0.0:
            raise ValueError("trajectory sample spacing must be positive")
        columns = resolve_mode_columns(data, config.modes)
        mode_results: dict[int, dict[str, Any]] = {}
        for mode, column in zip(config.modes, columns, strict=True):
            series = _copy_mode_to_host(
                array, column, config.transfer_chunk_samples
            )
            mode_results[mode] = {
                "allan": calculate_allan_variance(
                    series,
                    dt,
                    taus=config.taus,
                    points=config.points,
                    min_windows=config.min_windows,
                    min_independent_windows=config.min_independent_windows,
                    amplitude_floor=config.amplitude_floor,
                ),
                "phase_increment": _phase_increment_summary(
                    series, dt, config.amplitude_floor
                ),
            }
        payload = {
            "modes": list(config.modes),
            "t0": t0,
            "dt": dt,
            "n_traj": int(array.shape[0]),
            "n_samples": int(array.shape[1]),
            "mode_results": mode_results,
        }
        return AnalysisResult(data_dict=payload, meta=dict(payload, mode_results=None))

    def create_result_accumulator(self) -> AllanResultAccumulator:
        return AllanResultAccumulator()


def _copy_mode_to_host(
    array: Any, column: int, chunk_samples: int
) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return np.asarray(array[:, :, column])
    n_traj, n_samples = int(array.shape[0]), int(array.shape[1])
    first_stop = min(n_samples, chunk_samples)
    first = np.asarray(convert_to_numpy(array[:, :first_stop, column]))
    result = np.empty((n_traj, n_samples), dtype=first.dtype)
    result[:, :first_stop] = first
    for start in range(first_stop, n_samples, chunk_samples):
        stop = min(n_samples, start + chunk_samples)
        result[:, start:stop] = convert_to_numpy(array[:, start:stop, column])
    return result


def _is_complex_array(array: Any) -> bool:
    try:
        return bool(np.issubdtype(array.dtype, np.complexfloating))
    except TypeError:
        is_complex = getattr(array, "is_complex", None)
        return bool(is_complex()) if callable(is_complex) else False


def _masked_mean(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    count = np.sum(mask, axis=axis)
    total = np.sum(np.where(mask, values, 0.0), axis=axis)
    result = np.full(np.shape(total), np.nan, dtype=float)
    np.divide(total, count, out=result, where=count > 0)
    return result


def _phase_increment_summary(
    series: np.ndarray, dt: float, amplitude_floor: float
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


class AllanResultAccumulator:
    """Concatenate independent trajectory-batch Allan summaries."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def update(self, payload: dict[str, Any]) -> None:
        if self.payloads:
            first = self.payloads[0]
            if first["modes"] != payload["modes"]:
                raise ValueError("Allan trajectory batches use different modes")
            if first["n_samples"] != payload["n_samples"] or not math.isclose(
                first["dt"], payload["dt"]
            ):
                raise ValueError("Allan trajectory batches use different time grids")
            for mode in first["modes"]:
                first_tau = first["mode_results"][mode]["allan"]["tau"]
                next_tau = payload["mode_results"][mode]["allan"]["tau"]
                if not np.array_equal(first_tau, next_tau):
                    raise ValueError("Allan trajectory batches use different tau grids")
        self.payloads.append(payload)

    def finalize(self) -> dict[str, Any]:
        if not self.payloads:
            raise RuntimeError("cannot finalize an empty Allan accumulator")
        first = self.payloads[0]
        mode_results: dict[int, dict[str, Any]] = {}
        for mode in first["modes"]:
            allan_payloads = [
                item["mode_results"][mode]["allan"] for item in self.payloads
            ]
            allan = dict(allan_payloads[0])
            per_trajectory = np.concatenate(
                [np.asarray(item["per_trajectory"]) for item in allan_payloads], axis=0
            )
            valid = np.concatenate(
                [
                    np.asarray(item["valid_second_differences"])
                    for item in allan_payloads
                ],
                axis=0,
            )
            nonoverlap_per_trajectory = np.concatenate(
                [
                    np.asarray(item["nonoverlap_per_trajectory"])
                    for item in allan_payloads
                ],
                axis=0,
            )
            nonoverlap_valid = np.concatenate(
                [
                    np.asarray(item["nonoverlap_valid_second_differences"])
                    for item in allan_payloads
                ],
                axis=0,
            )
            mean, sem, sample_count = summarize_trajectories(per_trajectory)
            nonoverlap_mean, nonoverlap_sem, nonoverlap_sample_count = (
                summarize_trajectories(nonoverlap_per_trajectory)
            )
            allan.update(
                {
                    "angular_frequency_variance": mean,
                    "angular_frequency_variance_sem": sem,
                    "per_trajectory": per_trajectory,
                    "valid_second_differences": valid,
                    "trajectory_sample_count": sample_count,
                    "nonoverlap_angular_frequency_variance": nonoverlap_mean,
                    "nonoverlap_angular_frequency_variance_sem": nonoverlap_sem,
                    "nonoverlap_per_trajectory": nonoverlap_per_trajectory,
                    "nonoverlap_valid_second_differences": nonoverlap_valid,
                    "nonoverlap_trajectory_sample_count": nonoverlap_sample_count,
                    "total_independent_window_count": np.sum(
                        nonoverlap_valid, axis=0, dtype=np.int64
                    ),
                }
            )
            phase_payloads = [
                item["mode_results"][mode]["phase_increment"] for item in self.payloads
            ]
            phase = dict(phase_payloads[0])
            for key in (
                "mean_angular_frequency_per_trajectory",
                "max_abs_phase_step_per_trajectory",
                "near_nyquist_fraction_per_trajectory",
            ):
                phase[key] = np.concatenate(
                    [np.asarray(item[key]) for item in phase_payloads], axis=0
                )
            mode_results[mode] = {"allan": allan, "phase_increment": phase}

        return {
            "modes": list(first["modes"]),
            "t0": first["t0"],
            "dt": first["dt"],
            "n_traj": sum(int(item["n_traj"]) for item in self.payloads),
            "n_samples": first["n_samples"],
            "mode_results": mode_results,
        }
