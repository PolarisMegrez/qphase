"""Cross-scan white-FM window detection and Allan-noise scaling fits."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import numpy as np
from pydantic import BaseModel, Field, model_validator
from qphase.backend.base import BackendBase
from qphase.core.aggregation import AggregateResult, write_table_csv
from qphase.core.dataset import DatasetResultProtocol
from qphase.core.errors import QPhaseError
from qphase.core.protocols import PluginConfigBase, ResultProtocol
from scipy.optimize import curve_fit

from ..result import SDEResult
from .base import Analyzer
from .frequency_orientation import (
    FrequencyOrientation,
    orientation_metadata,
    resolve_frequency_orientation,
)
from .result import AnalysisResult

__all__ = [
    "AllanScalingAnalyzer",
    "AllanScalingConfig",
    "NormalFormExpectation",
]


class NormalFormExpectation(BaseModel):
    """Expected exponents for ``epsilon**k * x**m + x**n = 0``."""

    n: int = Field(..., ge=2)
    k: int = Field(1, ge=1)
    m: int = Field(0, ge=0)
    observable_order: int = Field(
        1,
        ge=1,
        description="Leading order of the measured frequency shift in x",
    )

    @model_validator(mode="after")
    def validate_orders(self) -> NormalFormExpectation:
        if self.m >= self.n:
            raise ValueError("normal_form.m must be smaller than normal_form.n")
        return self

    @property
    def frequency_exponent(self) -> float:
        return self.observable_order * self.k / (self.n - self.m)

    @property
    def allan_exponent(self) -> float:
        return 2.0 * (self.n - self.observable_order) * self.k / (self.n - self.m)


class AllanScalingConfig(PluginConfigBase):
    """Configuration for automated white-FM and scan-scaling analysis."""

    scan_param: str = Field(..., description="Scanned model parameter")
    critical_value: float = Field(..., description="Critical scan-parameter value")
    mode: int = Field(0, ge=0)
    source_key: str = Field("allan_variance")
    legacy_source_key: str | None = Field("trajectory_diagnostics")
    side: Literal["absolute", "positive", "negative"] = Field("absolute")
    white_slope_min: float = Field(-1.2)
    white_slope_max: float = Field(-0.8)
    local_window_points: int = Field(5, ge=3)
    min_local_r2: float = Field(0.95, ge=0.0, le=1.0)
    min_tau_decades: float = Field(0.25, ge=0.0)
    min_averaging_samples: int = Field(
        32,
        ge=1,
        description="Reject microscopic tau windows shorter than this many records",
    )
    window_selection: Literal["latest", "widest"] = Field(
        "latest",
        description=(
            "Prefer the asymptotically latest or logarithmically widest white window"
        ),
    )
    tau_min: float | None = Field(None, gt=0.0)
    tau_max: float | None = Field(None, gt=0.0)
    max_relative_sem: float = Field(0.2, gt=0.0)
    min_independent_windows: int = Field(32, ge=1)
    min_scaling_points: int = Field(5, ge=3)
    target_scaling_decades: float = Field(1.0, ge=0.0)
    max_scaling_reduced_chi2: float = Field(10.0, gt=0.0)
    bootstrap_samples: int = Field(2000, ge=0, le=100000)
    bootstrap_seed: int = Field(20260808, ge=0)
    fit_frequency: bool = Field(True)
    min_frequency_rss_improvement: float = Field(0.1, ge=0.0, lt=1.0)
    normal_form: NormalFormExpectation | None = Field(None)
    exponent_relative_tolerance: float = Field(0.1, gt=0.0)
    exponent_sigma_tolerance: float = Field(2.0, gt=0.0)
    export: list[str] = Field(
        default_factory=lambda: ["allan_points.csv", "allan_scaling.json"]
    )
    output_dir: str | None = Field(None)

    @model_validator(mode="after")
    def validate_ranges(self) -> AllanScalingConfig:
        if self.white_slope_min >= self.white_slope_max:
            raise ValueError("white_slope_min must be smaller than white_slope_max")
        if self.tau_min is not None and self.tau_max is not None:
            if self.tau_min >= self.tau_max:
                raise ValueError("tau_min must be smaller than tau_max")
        return self


@dataclass
class _Point:
    scan_value: float
    epsilon: float
    allan: dict[str, Any]
    phase: dict[str, Any]
    orientation: FrequencyOrientation


@dataclass
class _WhiteWindow:
    accepted: bool
    reason: str
    tau_start: float = math.nan
    tau_end: float = math.nan
    tau_decades: float = 0.0
    slope: float = math.nan
    r2: float = math.nan
    intensity: float = math.nan
    intensity_sem: float = math.nan
    relative_sem: float = math.nan
    independent_windows: int = 0
    independent_window_count_source: str = "measured_nonoverlap"
    intensity_tau_start: float = math.nan
    intensity_tau_end: float = math.nan
    per_trajectory_intensity: np.ndarray | None = None


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    design = np.column_stack((np.ones(x.size), x))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = y - fitted
    total = y - np.mean(y)
    ss_res = float(residual @ residual)
    ss_tot = float(total @ total)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return float(coefficients[0]), float(coefficients[1]), r2


def _mode_payload(payload: dict[str, Any], mode: int) -> dict[str, Any]:
    modes = payload.get("mode_results")
    if not isinstance(modes, dict):
        raise QPhaseError("Allan payload has no mode_results mapping")
    result = modes.get(mode, modes.get(str(mode)))
    if not isinstance(result, dict):
        raise QPhaseError(f"Allan payload has no mode {mode}")
    return result


def _extract_source(
    result: SDEResult, config: AllanScalingConfig
) -> tuple[dict[str, Any], dict[str, Any], FrequencyOrientation]:
    payload = result.analysis.get(config.source_key)
    if not isinstance(payload, dict) and config.legacy_source_key is not None:
        payload = result.analysis.get(config.legacy_source_key)
    if not isinstance(payload, dict):
        raise QPhaseError(
            f"SDE result has neither {config.source_key!r} nor a usable legacy source"
        )
    mode_payload = _mode_payload(payload, config.mode)
    allan = mode_payload.get("allan")
    phase = mode_payload.get("phase_increment", {})
    if not isinstance(allan, dict) or not isinstance(phase, dict):
        raise QPhaseError("Allan mode payload is incomplete")
    allan = dict(allan)
    allan["sample_spacing"] = float(payload.get("dt", math.nan))
    if "total_independent_window_count" not in allan:
        _estimate_legacy_independent_counts(allan, payload)
    return allan, phase, resolve_frequency_orientation(payload)


def _estimate_legacy_independent_counts(
    allan: dict[str, Any], payload: dict[str, Any]
) -> None:
    """Add an auditable conservative count estimate for legacy Allan payloads."""
    tau = np.asarray(allan.get("tau"), dtype=float)
    valid = np.asarray(allan.get("valid_second_differences"), dtype=float)
    dt = float(payload.get("dt", math.nan))
    n_samples = int(payload.get("n_samples", 0))
    if (
        tau.ndim != 1
        or valid.ndim != 2
        or valid.shape[1] != tau.size
        or not np.isfinite(dt)
        or dt <= 0.0
        or n_samples < 3
    ):
        allan["total_independent_window_count"] = np.zeros(tau.shape, dtype=int)
        allan["independent_window_count_source"] = "unavailable_legacy_payload"
        return
    averaging_samples = np.rint(tau / dt).astype(int)
    overlapping_nominal = np.maximum(n_samples - 2 * averaging_samples, 1)
    independent_nominal = np.maximum((n_samples - 1) // (2 * averaging_samples), 0)
    valid_fraction = np.clip(valid / overlapping_nominal[None, :], 0.0, 1.0)
    estimated = np.floor(valid_fraction * independent_nominal[None, :]).astype(int)
    allan["total_independent_window_count"] = np.sum(estimated, axis=0, dtype=np.int64)
    allan["independent_window_count_source"] = "estimated_from_legacy_overlap_counts"


def _iter_results(data: Any) -> list[SDEResult]:
    if isinstance(data, DatasetResultProtocol):
        return [
            point
            for index in np.ndindex(data.shape)
            if isinstance((point := data.point_view(index)), SDEResult)
        ]
    if isinstance(data, AggregateResult):
        data = data.results
    if isinstance(data, dict):
        return [value for value in data.values() if isinstance(value, SDEResult)]
    if isinstance(data, SDEResult):
        return [data]
    raise QPhaseError(
        f"AllanScalingAnalyzer received unsupported input type: {type(data).__name__}"
    )


def _epsilon(scan_value: float, config: AllanScalingConfig) -> float | None:
    signed = scan_value - config.critical_value
    if config.side == "positive" and signed <= 0.0:
        return None
    if config.side == "negative" and signed >= 0.0:
        return None
    return abs(signed)


def _collect_points(data: Any, config: AllanScalingConfig) -> list[_Point]:
    points: list[_Point] = []
    for result in _iter_results(data):
        params = result.meta.get("params", {})
        if config.scan_param not in params:
            raise QPhaseError(
                f"SDE result meta.params is missing {config.scan_param!r}"
            )
        scan_value = float(params[config.scan_param])
        epsilon = _epsilon(scan_value, config)
        if epsilon is None or epsilon <= 0.0:
            continue
        allan, phase, orientation = _extract_source(result, config)
        points.append(_Point(scan_value, epsilon, allan, phase, orientation))
    points.sort(key=lambda point: point.epsilon)
    if not points:
        raise QPhaseError(
            "AllanScalingAnalyzer received no nonzero perturbation points"
        )
    orientations = {point.orientation for point in points}
    if len(orientations) != 1:
        raise QPhaseError("Allan scan points use different frequency orientations")
    return points


def _white_window(point: _Point, config: AllanScalingConfig) -> _WhiteWindow:
    allan = point.allan
    tau = np.asarray(allan.get("tau"), dtype=float)
    variance = np.asarray(allan.get("angular_frequency_variance"), dtype=float)
    sem = np.asarray(allan.get("angular_frequency_variance_sem"), dtype=float)
    independent = np.asarray(
        allan.get("total_independent_window_count", np.zeros(tau.shape)), dtype=int
    )
    per_trajectory = np.asarray(
        allan.get("nonoverlap_per_trajectory", allan.get("per_trajectory")),
        dtype=float,
    )
    if per_trajectory.ndim != 2 or per_trajectory.shape[1] != tau.size:
        return _WhiteWindow(False, "missing per-trajectory Allan statistics")
    valid = (
        np.isfinite(tau)
        & np.isfinite(variance)
        & np.isfinite(sem)
        & (tau > 0.0)
        & (variance > 0.0)
        & (sem >= 0.0)
        & (independent >= config.min_independent_windows)
        & (sem / variance <= config.max_relative_sem)
    )
    if config.tau_min is not None:
        valid &= tau >= config.tau_min
    sample_spacing = float(allan.get("sample_spacing", math.nan))
    if np.isfinite(sample_spacing) and sample_spacing > 0.0:
        valid &= tau >= config.min_averaging_samples * sample_spacing
    if config.tau_max is not None:
        valid &= tau <= config.tau_max

    candidates: list[tuple[int, int, float, float]] = []
    width = config.local_window_points
    for start in range(0, tau.size - width + 1):
        stop = start + width
        if not np.all(valid[start:stop]):
            continue
        _, slope, r2 = _linear_fit(
            np.log(tau[start:stop]), np.log(variance[start:stop])
        )
        if (
            config.white_slope_min <= slope <= config.white_slope_max
            and r2 >= config.min_local_r2
        ):
            candidates.append((start, stop, slope, r2))
    if not candidates:
        return _WhiteWindow(False, "no local white-FM tau window")

    groups: list[list[tuple[int, int, float, float]]] = []
    for candidate in candidates:
        if not groups or candidate[0] > groups[-1][-1][1]:
            groups.append([candidate])
        else:
            groups[-1].append(candidate)
    bounds = [(group[0][0], group[-1][1]) for group in groups]
    if config.window_selection == "latest":
        start, stop = max(
            bounds,
            key=lambda item: (
                tau[item[1] - 1],
                math.log10(tau[item[1] - 1] / tau[item[0]]),
            ),
        )
    else:
        start, stop = max(
            bounds,
            key=lambda item: math.log10(tau[item[1] - 1] / tau[item[0]]),
        )
    _, slope, r2 = _linear_fit(np.log(tau[start:stop]), np.log(variance[start:stop]))
    tau_decades = math.log10(tau[stop - 1] / tau[start])
    if tau_decades < config.min_tau_decades:
        return _WhiteWindow(False, "white-FM tau window is too narrow")

    trajectory_intensity = np.nanmean(
        per_trajectory[:, start:stop] * tau[None, start:stop], axis=1
    )
    finite = np.isfinite(trajectory_intensity)
    if np.count_nonzero(finite) < 2:
        return _WhiteWindow(False, "too few trajectory-level Allan intensities")
    trajectory_intensity = trajectory_intensity[finite]
    intensity = float(np.mean(trajectory_intensity))
    intensity_sem = float(
        np.std(trajectory_intensity, ddof=1) / math.sqrt(trajectory_intensity.size)
    )
    relative_sem = intensity_sem / intensity if intensity > 0.0 else math.inf
    return _WhiteWindow(
        accepted=relative_sem <= config.max_relative_sem,
        reason="ok"
        if relative_sem <= config.max_relative_sem
        else "intensity SEM too large",
        tau_start=float(tau[start]),
        tau_end=float(tau[stop - 1]),
        tau_decades=tau_decades,
        slope=slope,
        r2=r2,
        intensity=intensity,
        intensity_sem=intensity_sem,
        relative_sem=relative_sem,
        independent_windows=int(np.min(independent[start:stop])),
        independent_window_count_source=str(
            allan.get("independent_window_count_source", "measured_nonoverlap")
        ),
        intensity_tau_start=float(tau[start]),
        intensity_tau_end=float(tau[stop - 1]),
        per_trajectory_intensity=trajectory_intensity,
    )


def _accepted_runs(windows: list[_WhiteWindow]) -> list[list[int]]:
    runs: list[list[int]] = []
    for index, window in enumerate(windows):
        if not window.accepted:
            continue
        if not runs or index != runs[-1][-1] + 1:
            runs.append([index])
        else:
            runs[-1].append(index)
    return runs


def _candidate_runs(
    points: list[_Point],
    windows: list[_WhiteWindow],
    config: AllanScalingConfig,
) -> list[tuple[list[int], tuple[float, float], float, float]]:
    candidates: list[tuple[list[int], tuple[float, float], float, float]] = []
    for run in _accepted_runs(windows):
        for start in range(len(run)):
            for stop in range(start + config.min_scaling_points, len(run) + 1):
                subset = run[start:stop]
                common_start = max(windows[index].tau_start for index in subset)
                common_end = min(windows[index].tau_end for index in subset)
                if common_end <= common_start:
                    continue
                tau_decades = math.log10(common_end / common_start)
                if tau_decades < config.min_tau_decades:
                    continue
                epsilon_decades = math.log10(
                    points[subset[-1]].epsilon / points[subset[0]].epsilon
                )
                candidates.append(
                    (subset, (common_start, common_end), epsilon_decades, tau_decades)
                )
    return candidates


def _set_common_intensity(
    point: _Point, window: _WhiteWindow, tau_start: float, tau_end: float
) -> None:
    tau = np.asarray(point.allan.get("tau"), dtype=float)
    per_trajectory = np.asarray(
        point.allan.get("nonoverlap_per_trajectory", point.allan.get("per_trajectory")),
        dtype=float,
    )
    selected = (tau >= tau_start) & (tau <= tau_end)
    if per_trajectory.ndim != 2 or not np.any(selected):
        window.accepted = False
        window.reason = "common white-FM tau window has no samples"
        return
    values = np.nanmean(per_trajectory[:, selected] * tau[None, selected], axis=1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        window.accepted = False
        window.reason = "common white-FM tau window has too few trajectories"
        return
    window.intensity = float(np.mean(values))
    window.intensity_sem = float(np.std(values, ddof=1) / math.sqrt(values.size))
    window.relative_sem = (
        window.intensity_sem / window.intensity if window.intensity > 0.0 else math.inf
    )
    window.intensity_tau_start = tau_start
    window.intensity_tau_end = tau_end
    window.per_trajectory_intensity = values


def _fit_noise(
    points: list[_Point],
    windows: list[_WhiteWindow],
    indices: list[int],
    config: AllanScalingConfig,
) -> dict[str, Any]:
    epsilon = np.asarray([points[index].epsilon for index in indices])
    intensity = np.asarray([windows[index].intensity for index in indices])
    sem = np.asarray([windows[index].intensity_sem for index in indices])
    log_epsilon = np.log(epsilon)
    log_intensity = np.log(intensity)
    sigma_log = np.maximum(sem / intensity, np.finfo(float).eps)
    design = np.column_stack((np.ones(epsilon.size), log_epsilon))
    weighted = design / sigma_log[:, None]
    target = log_intensity / sigma_log
    coefficients, *_ = np.linalg.lstsq(weighted, target, rcond=None)
    residual = log_intensity - design @ coefficients
    dof = max(epsilon.size - 2, 1)
    covariance = np.linalg.inv(weighted.T @ weighted)
    covariance *= float(np.sum((residual / sigma_log) ** 2) / dof)
    exponent = -float(coefficients[1])
    exponent_sem = math.sqrt(max(float(covariance[1, 1]), 0.0))

    bootstrap: list[float] = []
    if config.bootstrap_samples:
        rng = np.random.default_rng(config.bootstrap_seed)
        for _ in range(config.bootstrap_samples):
            sampled = []
            for index in indices:
                values = cast(np.ndarray, windows[index].per_trajectory_intensity)
                sampled.append(
                    float(np.mean(rng.choice(values, values.size, replace=True)))
                )
            if np.all(np.asarray(sampled) > 0.0):
                bootstrap_target = np.log(sampled) / sigma_log
                bootstrap_coefficients, *_ = np.linalg.lstsq(
                    weighted, bootstrap_target, rcond=None
                )
                bootstrap.append(-float(bootstrap_coefficients[1]))
    ci = (
        np.percentile(bootstrap, [2.5, 97.5]).tolist()
        if bootstrap
        else [math.nan, math.nan]
    )
    return {
        "model": "N_A = amplitude * abs(epsilon) ** (-q)",
        "amplitude": float(math.exp(coefficients[0])),
        "exponent": exponent,
        "exponent_sem": exponent_sem,
        "exponent_ci95": ci,
        "reduced_chi2": float(np.sum((residual / sigma_log) ** 2) / dof),
        "points": len(indices),
    }


def _frequency_summary(point: _Point) -> tuple[float, float]:
    values = np.asarray(
        point.phase.get("mean_angular_frequency_per_trajectory"), dtype=float
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    sem = (
        float(np.std(values, ddof=1) / math.sqrt(values.size))
        if values.size > 1
        else math.nan
    )
    return float(np.mean(values)), sem


def _fit_frequency(
    points: list[_Point], indices: list[int], config: AllanScalingConfig
) -> dict[str, Any]:
    epsilon = np.asarray([points[index].epsilon for index in indices])
    summaries = [_frequency_summary(points[index]) for index in indices]
    frequency = np.asarray([item[0] for item in summaries])
    sem = np.asarray([item[1] for item in summaries])
    valid = np.isfinite(frequency)
    epsilon = epsilon[valid]
    frequency = frequency[valid]
    sem = sem[valid]
    if epsilon.size < 5:
        return {"status": "insufficient_points", "points": int(epsilon.size)}

    sigma = np.where(np.isfinite(sem) & (sem > 0.0), sem, np.nan)
    if not np.any(np.isfinite(sigma)):
        sigma_arg = None
    else:
        fallback = float(np.nanmedian(sigma))
        sigma_arg = np.where(np.isfinite(sigma), sigma, fallback)

    def model(
        x: np.ndarray, offset: float, amplitude: float, exponent: float
    ) -> np.ndarray:
        return offset + amplitude * x**exponent

    expected = (
        config.normal_form.frequency_exponent if config.normal_form else 1.0 / 3.0
    )
    denominator = float(epsilon[-1] ** expected - epsilon[0] ** expected)
    amplitude = (
        float((frequency[-1] - frequency[0]) / denominator)
        if abs(denominator) > np.finfo(float).eps
        else max(float(np.ptp(frequency)), np.finfo(float).eps)
    )
    offset = float(frequency[0] - amplitude * epsilon[0] ** expected)
    initial = [offset, amplitude, expected]
    try:
        params, covariance = curve_fit(
            model,
            epsilon,
            frequency,
            p0=initial,
            sigma=sigma_arg,
            absolute_sigma=sigma_arg is not None,
            bounds=([-np.inf, -np.inf, 0.02], [np.inf, np.inf, 1.5]),
            maxfev=50000,
        )
    except (RuntimeError, ValueError) as exc:
        return {"status": "fit_failed", "error": str(exc), "points": int(epsilon.size)}
    fitted = model(epsilon, *params)
    linear_design = np.column_stack((np.ones(epsilon.size), epsilon))
    linear_params, *_ = np.linalg.lstsq(linear_design, frequency, rcond=None)
    linear_residual = frequency - linear_design @ linear_params
    nonlinear_residual = frequency - fitted
    exponent_sem = math.sqrt(max(float(covariance[2, 2]), 0.0))
    expected_fit: dict[str, Any] | None = None
    if config.normal_form is not None:
        expected = config.normal_form.frequency_exponent
        expected_design = np.column_stack((np.ones(epsilon.size), epsilon**expected))
        if sigma_arg is None:
            expected_params, *_ = np.linalg.lstsq(
                expected_design, frequency, rcond=None
            )
        else:
            expected_params, *_ = np.linalg.lstsq(
                expected_design / sigma_arg[:, None],
                frequency / sigma_arg,
                rcond=None,
            )
        expected_residual = frequency - expected_design @ expected_params
        expected_fit = {
            "exponent": expected,
            "offset": float(expected_params[0]),
            "amplitude": float(expected_params[1]),
            "rss": float(expected_residual @ expected_residual),
        }
    return {
        "status": "ok",
        "model": "omega0 + A * abs(epsilon) ** p",
        "offset": float(params[0]),
        "amplitude": float(params[1]),
        "exponent": float(params[2]),
        "exponent_sem": exponent_sem,
        "nonlinear_rss": float(nonlinear_residual @ nonlinear_residual),
        "linear_rss": float(linear_residual @ linear_residual),
        "expected_exponent_fit": expected_fit,
        "points": int(epsilon.size),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _exponent_matches(
    observed: float | None,
    expected: float,
    config: AllanScalingConfig,
    *,
    sem: float | None = None,
    ci95: list[float] | None = None,
) -> bool:
    if observed is None or not np.isfinite(observed):
        return False
    difference = abs(observed - expected)
    if difference > config.exponent_relative_tolerance * abs(expected):
        return False
    if ci95 is not None and len(ci95) == 2 and np.all(np.isfinite(ci95)):
        return bool(ci95[0] <= expected <= ci95[1])
    if sem is not None and np.isfinite(sem):
        return difference <= max(config.exponent_sigma_tolerance * sem, 1e-8)
    return True


class AllanScalingAnalyzer(Analyzer):
    """Detect white-FM windows and fit Allan-noise scaling across a scan."""

    name: ClassVar[str] = "allan_scaling"
    description: ClassVar[str] = "Cross-scan white-FM detection and scaling fits"
    config_schema: ClassVar[type[AllanScalingConfig]] = AllanScalingConfig

    def analyze(self, data: Any, backend: BackendBase) -> ResultProtocol:
        del backend
        config = cast(AllanScalingConfig, self.config)
        points = _collect_points(data, config)
        windows = [_white_window(point, config) for point in points]
        candidates = _candidate_runs(points, windows, config)
        selected: list[int] = []
        common_tau: tuple[float, float] | None = None
        if candidates:
            trial_config = config.model_copy(update={"bootstrap_samples": 0})
            ranked: list[
                tuple[
                    tuple[bool, float, float, int],
                    list[int],
                    tuple[float, float],
                ]
            ] = []
            for subset, common, epsilon_span, tau_span in candidates:
                trial_windows = [replace(window) for window in windows]
                for index in subset:
                    _set_common_intensity(
                        points[index],
                        trial_windows[index],
                        common[0],
                        common[1],
                    )
                trial_fit = _fit_noise(points, trial_windows, subset, trial_config)
                compatible = (
                    trial_fit["reduced_chi2"] <= config.max_scaling_reduced_chi2
                )
                if config.normal_form is not None:
                    expected = config.normal_form.allan_exponent
                    compatible &= abs(trial_fit["exponent"] - expected) <= (
                        config.exponent_relative_tolerance * expected
                    )
                ranked.append(
                    (
                        (compatible, epsilon_span, tau_span, len(subset)),
                        subset,
                        common,
                    )
                )
            _, selected, common_tau = max(ranked, key=lambda item: item[0])
        if common_tau is not None:
            for index in selected:
                _set_common_intensity(
                    points[index], windows[index], common_tau[0], common_tau[1]
                )
        epsilon_decades = (
            math.log10(points[selected[-1]].epsilon / points[selected[0]].epsilon)
            if selected
            else 0.0
        )
        noise_fit = _fit_noise(points, windows, selected, config) if selected else None
        frequency_fit = (
            _fit_frequency(points, selected, config)
            if selected and config.fit_frequency
            else None
        )
        normal_form: dict[str, Any] | None = None
        if config.normal_form is not None:
            expected_p = config.normal_form.frequency_exponent
            expected_q = config.normal_form.allan_exponent
            observed_p = frequency_fit.get("exponent") if frequency_fit else None
            observed_q = noise_fit.get("exponent") if noise_fit else None
            free_frequency_match = _exponent_matches(
                observed_p,
                expected_p,
                config,
                sem=frequency_fit.get("exponent_sem") if frequency_fit else None,
            )
            expected_frequency_fit = (
                frequency_fit.get("expected_exponent_fit") if frequency_fit else None
            )
            fixed_frequency_match = bool(
                expected_frequency_fit
                and frequency_fit
                and expected_frequency_fit["rss"]
                <= (1.0 - config.min_frequency_rss_improvement)
                * frequency_fit["linear_rss"]
                and observed_p is not None
                and abs(observed_p - expected_p)
                <= max(
                    config.exponent_sigma_tolerance
                    * frequency_fit.get("exponent_sem", math.nan),
                    1e-8,
                )
            )
            normal_form = {
                "n": config.normal_form.n,
                "k": config.normal_form.k,
                "m": config.normal_form.m,
                "observable_order": config.normal_form.observable_order,
                "expected_frequency_exponent": expected_p,
                "expected_allan_exponent": expected_q,
                "frequency_match": free_frequency_match or fixed_frequency_match,
                "frequency_match_basis": (
                    "free_exponent"
                    if free_frequency_match
                    else "fixed_expected_exponent"
                    if fixed_frequency_match
                    else "none"
                ),
                "allan_match": _exponent_matches(
                    observed_q,
                    expected_q,
                    config,
                    sem=noise_fit.get("exponent_sem") if noise_fit else None,
                    ci95=noise_fit.get("exponent_ci95") if noise_fit else None,
                ),
            }

        rows: list[dict[str, Any]] = []
        selected_set = set(selected)
        for index, (point, window) in enumerate(zip(points, windows, strict=True)):
            frequency, frequency_sem = _frequency_summary(point)
            rows.append(
                {
                    config.scan_param: point.scan_value,
                    "epsilon": point.epsilon,
                    "accepted": window.accepted,
                    "selected": index in selected_set,
                    "reason": window.reason,
                    "tau_start": window.tau_start,
                    "tau_end": window.tau_end,
                    "tau_decades": window.tau_decades,
                    "allan_slope": window.slope,
                    "allan_r2": window.r2,
                    "allan_intensity": window.intensity,
                    "allan_intensity_sem": window.intensity_sem,
                    "relative_sem": window.relative_sem,
                    "independent_windows": window.independent_windows,
                    "independent_window_count_source": (
                        window.independent_window_count_source
                    ),
                    "intensity_tau_start": window.intensity_tau_start,
                    "intensity_tau_end": window.intensity_tau_end,
                    "mean_angular_frequency": frequency,
                    "mean_angular_frequency_sem": frequency_sem,
                    "orientation": point.orientation,
                }
            )
        gate_failures: list[str] = []
        if epsilon_decades < config.target_scaling_decades:
            gate_failures.append("epsilon_window_too_narrow")
        if noise_fit is None:
            gate_failures.append("allan_scaling_unavailable")
        elif noise_fit["reduced_chi2"] > config.max_scaling_reduced_chi2:
            gate_failures.append("allan_scaling_poor_fit")
        if config.fit_frequency:
            if not frequency_fit or frequency_fit.get("status") != "ok":
                gate_failures.append("frequency_fit_unavailable")
            elif (
                frequency_fit["nonlinear_rss"]
                > (1.0 - config.min_frequency_rss_improvement)
                * frequency_fit["linear_rss"]
            ):
                gate_failures.append("frequency_nonlinearity_not_resolved")
        if normal_form is not None:
            if config.fit_frequency and not normal_form["frequency_match"]:
                gate_failures.append("frequency_exponent_mismatch")
            if not normal_form["allan_match"]:
                gate_failures.append("allan_exponent_mismatch")
        status = "ok" if not gate_failures else "target_not_met"
        summary = {
            "status": status,
            "gate_failures": gate_failures,
            "scan_param": config.scan_param,
            "critical_value": config.critical_value,
            "mode": config.mode,
            "accepted_points": int(sum(window.accepted for window in windows)),
            "selected_points": len(selected),
            "epsilon_min": points[selected[0]].epsilon if selected else math.nan,
            "epsilon_max": points[selected[-1]].epsilon if selected else math.nan,
            "epsilon_window_decades": epsilon_decades,
            "common_tau_start": common_tau[0] if common_tau else math.nan,
            "common_tau_end": common_tau[1] if common_tau else math.nan,
            "common_tau_decades": (
                math.log10(common_tau[1] / common_tau[0]) if common_tau else 0.0
            ),
            "target_scaling_decades": config.target_scaling_decades,
            "noise_fit": noise_fit,
            "frequency_fit": frequency_fit,
            "normal_form": normal_form,
            **orientation_metadata(points[0].orientation),
        }
        written = self._export(config, rows, summary)
        return AnalysisResult(
            data_dict={"rows": rows, "summary": summary, "written": written},
            meta={
                "status": status,
                "scan_param": config.scan_param,
                "mode": config.mode,
                "count": len(points),
                **orientation_metadata(points[0].orientation),
            },
        )

    def _export(
        self,
        config: AllanScalingConfig,
        rows: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> dict[str, str]:
        output_dir = getattr(self, "output_dir", None) or config.output_dir
        if output_dir is None or not config.export:
            return {}
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}
        if "allan_points.csv" in config.export:
            path = write_table_csv(rows, root / "allan_points.csv")
            written["allan_points"] = str(path)
        if "allan_scaling.json" in config.export:
            path = root / "allan_scaling.json"
            path.write_text(
                json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
            )
            written["allan_scaling"] = str(path)
        return written
