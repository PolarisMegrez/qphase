"""Adaptive carrier estimation from a band-limited power spectrum."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import numpy as np
from pydantic import Field, field_validator, model_validator
from qphase.backend.base import BackendBase
from qphase.core.aggregation import write_table_csv
from qphase.core.errors import QPhaseError
from qphase.core.protocols import PluginConfigBase, ResultProtocol

from .base import Analyzer, AnalyzerExecutionCapabilities
from .frequency_orientation import (
    FrequencyOrientation,
    orientation_metadata,
    resolve_frequency_orientation,
)
from .result import AnalysisResult
from .result_input import load_sde_results

__all__ = [
    "BandLimitedCarrierAnalyzer",
    "BandLimitedCarrierCandidate",
    "BandLimitedCarrierConfig",
    "BandLimitedCarrierEstimate",
    "estimate_band_limited_carrier",
]


class BandLimitedCarrierConfig(PluginConfigBase):
    """Configuration for adaptive, spectrally filtered carrier estimation."""

    scan_param: str = Field(..., description="Parameter used as the scan axis")
    psd_key: str = Field("psd", description="Analysis key containing the PSD")
    readout: int | Literal["trace"] = Field(
        "trace",
        description="Physical mode index or incoherent trace over recorded modes",
    )
    freq_min: float | None = Field(None, description="Lower carrier search bound")
    freq_max: float | None = Field(None, description="Upper carrier search bound")
    bandwidth_multipliers: list[float] = Field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0],
        description=(
            "Nested half-bandwidths relative to the spectral concentration width"
        ),
    )
    bandwidth_min: float | None = Field(
        None, gt=0.0, description="Optional minimum candidate half-bandwidth"
    )
    bandwidth_max: float | None = Field(
        None, gt=0.0, description="Optional maximum candidate half-bandwidth"
    )
    baseline_quantile: float = Field(0.05, ge=0.0, lt=0.5)
    concentration_clip_sigma: float = Field(5.0, gt=1.0)
    concentration_iterations: int = Field(2, ge=0, le=8)
    peak_smoothing_bins: int = Field(5, ge=1, le=101)
    taper_fraction: float = Field(0.2, gt=0.0, lt=1.0)
    maximum_lag: float = Field(256.0, gt=0.0)
    coherence_floor: float = Field(0.05, gt=0.0, lt=1.0)
    minimum_lag_points: int = Field(8, ge=3)
    consensus_count: int = Field(3, ge=2, le=16)
    stability_fraction: float = Field(
        0.015,
        gt=0.0,
        description="Allowed carrier drift as a fraction of candidate bandwidth",
    )
    stability_sigma: float = Field(2.0, gt=0.0)
    export: list[str] = Field(
        default_factory=lambda: ["carrier_results.csv", "carrier_candidates.csv"]
    )
    output_dir: str | None = Field(None, description="Usually injected by the engine")
    pattern: str = Field("*.npz", description="Glob for saved result inputs")

    @field_validator("bandwidth_multipliers")
    @classmethod
    def validate_bandwidth_multipliers(cls, values: list[float]) -> list[float]:
        if len(values) < 2 or any(
            not np.isfinite(value) or value <= 0 for value in values
        ):
            raise ValueError(
                "bandwidth_multipliers must contain at least two positives"
            )
        if any(right <= left for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("bandwidth_multipliers must be strictly increasing")
        return values

    @model_validator(mode="after")
    def validate_ranges(self) -> BandLimitedCarrierConfig:
        if (
            self.freq_min is not None
            and self.freq_max is not None
            and self.freq_min >= self.freq_max
        ):
            raise ValueError("freq_min must be smaller than freq_max")
        if (
            self.bandwidth_min is not None
            and self.bandwidth_max is not None
            and self.bandwidth_min > self.bandwidth_max
        ):
            raise ValueError("bandwidth_min cannot exceed bandwidth_max")
        if self.consensus_count > len(self.bandwidth_multipliers):
            raise ValueError("consensus_count cannot exceed the number of bandwidths")
        return self


@dataclass(frozen=True)
class BandLimitedCarrierCandidate:
    """One nested-bandwidth phase-regression result."""

    half_bandwidth: float
    frequency: float = math.nan
    regression_std: float = math.nan
    phase_fit_rms: float = math.nan
    lag_points: int = 0
    lag_end: float = math.nan
    status: str = "failed"
    error: str = ""


@dataclass(frozen=True)
class BandLimitedCarrierEstimate:
    """Selected carrier and auditable bandwidth diagnostics."""

    frequency: float = math.nan
    regression_std: float = math.nan
    bandwidth_std: float = math.nan
    diagnostic_uncertainty: float = math.nan
    peak_center: float = math.nan
    peak_hwhm: float = math.nan
    spectral_width: float = math.nan
    selected_half_bandwidth: float = math.nan
    consensus_count: int = 0
    phase_fit_rms: float = math.nan
    lag_points: int = 0
    lag_end: float = math.nan
    status: str = "failed"
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_spectrum(
    axis: np.ndarray, spectrum: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    frequency = np.asarray(axis, dtype=float).reshape(-1)
    power = np.asarray(spectrum, dtype=float).reshape(-1)
    if frequency.size < 64 or frequency.size != power.size:
        raise ValueError("axis and spectrum must have the same length of at least 64")
    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(power)):
        raise ValueError("axis and spectrum must be finite")
    spacing = np.diff(frequency)
    step = float(np.median(spacing))
    if step <= 0.0 or not np.allclose(spacing, step, rtol=1.0e-6, atol=0.0):
        raise ValueError("frequency axis must be uniformly increasing")
    if np.max(power) <= 0.0:
        raise ValueError("spectrum must contain positive power")
    return frequency, power, step


def _concentration_moments(
    axis: np.ndarray,
    spectrum: np.ndarray,
    *,
    baseline_quantile: float,
    clip_sigma: float,
    iterations: int,
) -> tuple[float, float, float]:
    selected = np.ones(axis.shape, dtype=bool)
    center = float(np.mean(axis))
    width = float(np.std(axis))
    baseline = float(np.quantile(spectrum, baseline_quantile))
    for _ in range(iterations + 1):
        baseline = float(np.quantile(spectrum[selected], baseline_quantile))
        excess = np.maximum(spectrum - baseline, 0.0)
        weights = np.where(selected, excess * excess, 0.0)
        total = float(np.sum(weights))
        if total <= np.finfo(float).tiny:
            raise ValueError("baseline subtraction removed all spectral power")
        center = float(np.sum(axis * weights) / total)
        variance = float(np.sum((axis - center) ** 2 * weights) / total)
        width = math.sqrt(max(variance, 0.0))
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("spectral concentration width is not resolved")
        selected = np.abs(axis - center) <= clip_sigma * width
        if np.count_nonzero(selected) < 16:
            break
    return center, width, baseline


def _moving_average(values: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 1:
        return values
    if bins % 2 == 0:
        bins += 1
    kernel = np.full(bins, 1.0 / bins)
    return np.convolve(values, kernel, mode="same")


def _connected_peak(
    axis: np.ndarray,
    spectrum: np.ndarray,
    *,
    concentration_center: float,
    concentration_width: float,
    baseline: float,
    smoothing_bins: int,
    spacing: float,
) -> tuple[float, float]:
    neighborhood = np.abs(axis - concentration_center) <= max(
        2.0 * concentration_width, 16.0 * spacing
    )
    indices = np.flatnonzero(neighborhood)
    if indices.size < 3:
        raise ValueError("carrier neighborhood has too few frequency bins")
    smoothed = _moving_average(spectrum, smoothing_bins)
    peak = int(indices[np.argmax(smoothed[indices])])
    half_height = baseline + 0.5 * (smoothed[peak] - baseline)
    left = peak
    while left > indices[0] and smoothed[left - 1] >= half_height:
        left -= 1
    right = peak
    while right < indices[-1] and smoothed[right + 1] >= half_height:
        right += 1
    local = slice(left, right + 1)
    weights = np.maximum(spectrum[local] - baseline, 0.0)
    if float(np.sum(weights)) > np.finfo(float).tiny:
        center = float(np.sum(axis[local] * weights) / np.sum(weights))
    else:
        center = float(axis[peak])
    hwhm = max(0.5 * float(axis[right] - axis[left]), 0.5 * spacing)
    return center, hwhm


def _hac_slope_std(
    design: np.ndarray, weights: np.ndarray, residual: np.ndarray
) -> float:
    normal = design.T @ (weights[:, None] * design)
    bread = np.linalg.pinv(normal)
    scores = design * (weights * residual)[:, None]
    count = scores.shape[0]
    bandwidth = min(
        count - 1,
        max(1, int(math.floor(4.0 * (count / 100.0) ** (2.0 / 9.0)))),
    )
    meat = scores.T @ scores
    for lag in range(1, bandwidth + 1):
        kernel = 1.0 - lag / (bandwidth + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat += kernel * (cross + cross.T)
    covariance = bread @ meat @ bread
    if count > design.shape[1]:
        covariance *= count / (count - design.shape[1])
    return math.sqrt(max(float(covariance[1, 1]), 0.0))


def _fit_candidate(
    axis: np.ndarray,
    spectrum: np.ndarray,
    *,
    center: float,
    half_bandwidth: float,
    baseline_quantile: float,
    taper_fraction: float,
    maximum_lag: float,
    coherence_floor: float,
    minimum_lag_points: int,
) -> BandLimitedCarrierCandidate:
    selected = np.abs(axis - center) <= half_bandwidth
    if np.count_nonzero(selected) < 16:
        return BandLimitedCarrierCandidate(
            half_bandwidth=half_bandwidth, error="band contains fewer than 16 bins"
        )
    baseline = float(np.quantile(spectrum[selected], baseline_quantile))
    coordinate = np.abs((axis[selected] - center) / half_bandwidth)
    taper = np.ones(coordinate.shape)
    edge = coordinate > 1.0 - taper_fraction
    taper[edge] = 0.5 * (
        1.0
        + np.cos(np.pi * (coordinate[edge] - (1.0 - taper_fraction)) / taper_fraction)
    )
    filtered = np.zeros(axis.shape, dtype=float)
    filtered[selected] = np.maximum(spectrum[selected] - baseline, 0.0) * taper
    if float(np.sum(filtered)) <= np.finfo(float).tiny:
        return BandLimitedCarrierCandidate(
            half_bandwidth=half_bandwidth, error="filtered band has no excess power"
        )

    spacing = float(axis[1] - axis[0])
    lag_step = 2.0 * np.pi / (axis.size * spacing)
    lag = np.arange(axis.size, dtype=float) * lag_step
    correlation = np.fft.ifft(filtered) * np.exp(1j * axis[0] * lag)
    amplitude = np.abs(correlation) / max(
        abs(complex(correlation[0])), np.finfo(float).tiny
    )
    indices = np.flatnonzero(
        (lag > 0.0) & (lag <= maximum_lag) & (amplitude >= coherence_floor)
    )
    if indices.size:
        gaps = np.flatnonzero(np.diff(indices) > 1)
        if gaps.size:
            indices = indices[: gaps[0] + 1]
    if indices.size < minimum_lag_points:
        return BandLimitedCarrierCandidate(
            half_bandwidth=half_bandwidth,
            lag_points=int(indices.size),
            error="usable coherence interval is too short",
        )

    time = lag[indices]
    phase = np.unwrap(np.angle(correlation[indices]))
    weights = amplitude[indices] ** 2
    design = np.column_stack((np.ones(time.size), time))
    root_weight = np.sqrt(weights)
    coefficient, *_ = np.linalg.lstsq(
        design * root_weight[:, None], phase * root_weight, rcond=None
    )
    residual = phase - design @ coefficient
    rms = math.sqrt(float(np.sum(weights * residual**2) / np.sum(weights)))
    return BandLimitedCarrierCandidate(
        half_bandwidth=half_bandwidth,
        frequency=float(coefficient[1]),
        regression_std=_hac_slope_std(design, weights, residual),
        phase_fit_rms=rms,
        lag_points=int(indices.size),
        lag_end=float(time[-1]),
        status="ok",
    )


def _candidate_bandwidths(
    center: float,
    spectral_width: float,
    target_min: float,
    target_max: float,
    spacing: float,
    multipliers: list[float] | tuple[float, ...],
    bandwidth_min: float | None,
    bandwidth_max: float | None,
) -> list[float]:
    edge_limit = 0.98 * min(center - target_min, target_max - center)
    upper = edge_limit if bandwidth_max is None else min(edge_limit, bandwidth_max)
    lower = max(16.0 * spacing, bandwidth_min or 0.0)
    raw = np.clip(spectral_width * np.asarray(multipliers, dtype=float), lower, upper)
    return [
        float(value)
        for value in np.unique(raw)
        if np.isfinite(value) and lower <= value <= upper
    ]


def _select_candidate(
    candidates: list[BandLimitedCarrierCandidate],
    *,
    reference_width: float,
    consensus_count: int,
    stability_fraction: float,
    stability_sigma: float,
) -> tuple[BandLimitedCarrierCandidate, list[BandLimitedCarrierCandidate], str]:
    valid = [candidate for candidate in candidates if candidate.status == "ok"]
    if not valid:
        raise ValueError("all candidate bandwidth fits failed")
    stable_groups: list[
        tuple[float, BandLimitedCarrierCandidate, list[BandLimitedCarrierCandidate]]
    ] = []
    for end in range(consensus_count - 1, len(valid)):
        group = valid[end - consensus_count + 1 : end + 1]
        frequencies = np.asarray([item.frequency for item in group])
        regression = np.asarray(
            [
                item.regression_std if np.isfinite(item.regression_std) else 0.0
                for item in group
            ]
        )
        center = float(np.median(frequencies))
        center_std = float(np.median(regression))
        tolerance = stability_fraction * float(
            np.median([item.half_bandwidth for item in group])
        ) + stability_sigma * np.hypot(regression, center_std)
        if np.all(np.abs(frequencies - center) <= tolerance):
            log_span = math.log(group[-1].half_bandwidth / group[0].half_bandwidth)
            local_slope = float(np.ptp(frequencies)) / log_span
            selected = group[len(group) // 2]
            stable_groups.append((local_slope, selected, group))
    if stable_groups:
        _, selected, group = min(stable_groups, key=lambda item: item[0])
        return selected, group, "ok"

    selected = min(valid, key=lambda item: abs(item.half_bandwidth - reference_width))
    position = valid.index(selected)
    start = max(0, min(position - consensus_count // 2, len(valid) - consensus_count))
    group = valid[start : start + consensus_count]
    return selected, group, "unstable_bandwidth"


def estimate_band_limited_carrier(
    axis: np.ndarray,
    spectrum: np.ndarray,
    *,
    freq_min: float | None = None,
    freq_max: float | None = None,
    bandwidth_multipliers: list[float] | tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0),
    bandwidth_min: float | None = None,
    bandwidth_max: float | None = None,
    baseline_quantile: float = 0.05,
    concentration_clip_sigma: float = 5.0,
    concentration_iterations: int = 2,
    peak_smoothing_bins: int = 5,
    taper_fraction: float = 0.2,
    maximum_lag: float = 256.0,
    coherence_floor: float = 0.05,
    minimum_lag_points: int = 8,
    consensus_count: int = 3,
    stability_fraction: float = 0.015,
    stability_sigma: float = 2.0,
) -> tuple[BandLimitedCarrierEstimate, list[BandLimitedCarrierCandidate]]:
    """Estimate a filtered long-time carrier using nested bandwidths.

    Candidate windows are scaled by the standard deviation of the squared,
    baseline-subtracted spectrum. For an ideal Lorentz profile this equals the
    HWHM. A Lepski-style consistency test selects the center of the flattest
    local bandwidth plateau without using an external theoretical frequency.
    """
    try:
        frequency, power, spacing = _validate_spectrum(axis, spectrum)
        lower = float(frequency[0] if freq_min is None else freq_min)
        upper = float(frequency[-1] if freq_max is None else freq_max)
        target = (frequency >= lower) & (frequency <= upper)
        if np.count_nonzero(target) < 64:
            raise ValueError("carrier search range contains fewer than 64 bins")
        target_axis = frequency[target]
        target_power = power[target]
        concentration_center, spectral_width, baseline = _concentration_moments(
            target_axis,
            target_power,
            baseline_quantile=baseline_quantile,
            clip_sigma=concentration_clip_sigma,
            iterations=concentration_iterations,
        )
        peak_center, peak_hwhm = _connected_peak(
            target_axis,
            target_power,
            concentration_center=concentration_center,
            concentration_width=spectral_width,
            baseline=baseline,
            smoothing_bins=peak_smoothing_bins,
            spacing=spacing,
        )
        bandwidths = _candidate_bandwidths(
            peak_center,
            spectral_width,
            float(target_axis[0]),
            float(target_axis[-1]),
            spacing,
            bandwidth_multipliers,
            bandwidth_min,
            bandwidth_max,
        )
        if len(bandwidths) < consensus_count:
            raise ValueError(
                "search bounds leave too few distinct candidate bandwidths"
            )
        candidates = [
            _fit_candidate(
                frequency,
                power,
                center=peak_center,
                half_bandwidth=half_bandwidth,
                baseline_quantile=baseline_quantile,
                taper_fraction=taper_fraction,
                maximum_lag=maximum_lag,
                coherence_floor=coherence_floor,
                minimum_lag_points=minimum_lag_points,
            )
            for half_bandwidth in bandwidths
        ]
        selected, consensus, status = _select_candidate(
            candidates,
            reference_width=spectral_width,
            consensus_count=consensus_count,
            stability_fraction=stability_fraction,
            stability_sigma=stability_sigma,
        )
        frequencies = np.asarray([item.frequency for item in consensus])
        bandwidth_std = (
            float(np.std(frequencies, ddof=1)) if frequencies.size > 1 else 0.0
        )
        diagnostic = math.hypot(selected.regression_std, bandwidth_std)
        estimate = BandLimitedCarrierEstimate(
            frequency=selected.frequency,
            regression_std=selected.regression_std,
            bandwidth_std=bandwidth_std,
            diagnostic_uncertainty=diagnostic,
            peak_center=peak_center,
            peak_hwhm=peak_hwhm,
            spectral_width=spectral_width,
            selected_half_bandwidth=selected.half_bandwidth,
            consensus_count=len(consensus),
            phase_fit_rms=selected.phase_fit_rms,
            lag_points=selected.lag_points,
            lag_end=selected.lag_end,
            status=status,
        )
        return estimate, candidates
    except Exception as exc:
        return BandLimitedCarrierEstimate(error=str(exc)), []


def _extract_readout(
    loaded: Any, psd_key: str, readout: int | Literal["trace"]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if psd_key not in loaded.analysis:
        raise QPhaseError(f"{loaded.path} has no analysis[{psd_key!r}] payload")
    payload = loaded.analysis[psd_key]
    axis = np.asarray(payload["axis"], dtype=float).reshape(-1)
    matrix = np.asarray(payload["psd"], dtype=float)
    if matrix.ndim == 1:
        spectrum = matrix
    elif matrix.ndim == 2 and readout == "trace":
        spectrum = np.sum(matrix, axis=1)
    elif matrix.ndim == 2 and isinstance(readout, int):
        modes = [int(mode) for mode in payload.get("modes", [])]
        if readout in modes:
            column = modes.index(readout)
        elif 0 <= readout < matrix.shape[1]:
            column = readout
        else:
            raise QPhaseError(f"mode {readout} is absent from {loaded.path}")
        spectrum = matrix[:, column]
    else:
        raise QPhaseError(f"PSD in {loaded.path} must be one- or two-dimensional")
    return axis, np.asarray(spectrum, dtype=float).reshape(-1), payload


def _sort_key(value: Any) -> tuple[int, Any]:
    try:
        return 0, float(value)
    except (TypeError, ValueError):
        return 1, str(value)


class BandLimitedCarrierAnalyzer(Analyzer):
    """Cross-result adaptive carrier estimator for saved PSD datasets."""

    name: ClassVar[str] = "band_limited_carrier"
    description: ClassVar[str] = (
        "Adaptive band-limited long-time first-order-coherence carrier"
    )
    config_schema: ClassVar[type[BandLimitedCarrierConfig]] = BandLimitedCarrierConfig

    def capabilities(self) -> AnalyzerExecutionCapabilities:
        return AnalyzerExecutionCapabilities(
            execution_location="host", requires_full_trajectory=False
        )

    def analyze(self, data: Any, backend: BackendBase) -> ResultProtocol:
        del backend
        config = cast(BandLimitedCarrierConfig, self.config)
        loaded_results = load_sde_results(data, config.pattern)
        if not loaded_results:
            raise QPhaseError("band_limited_carrier received no input results")

        rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        reference_orientation: FrequencyOrientation | None = None
        for loaded in loaded_results:
            params = loaded.meta.get("params", {})
            if config.scan_param not in params:
                raise QPhaseError(
                    f"{loaded.path} meta.params is missing {config.scan_param!r}"
                )
            scan_value = params[config.scan_param]
            axis, spectrum, payload = _extract_readout(
                loaded, config.psd_key, config.readout
            )
            orientation = resolve_frequency_orientation(payload)
            if reference_orientation is None:
                reference_orientation = orientation
            elif reference_orientation != orientation:
                raise QPhaseError("PSD inputs use different frequency orientations")
            estimate, candidates = estimate_band_limited_carrier(
                axis,
                spectrum,
                freq_min=config.freq_min,
                freq_max=config.freq_max,
                bandwidth_multipliers=config.bandwidth_multipliers,
                bandwidth_min=config.bandwidth_min,
                bandwidth_max=config.bandwidth_max,
                baseline_quantile=config.baseline_quantile,
                concentration_clip_sigma=config.concentration_clip_sigma,
                concentration_iterations=config.concentration_iterations,
                peak_smoothing_bins=config.peak_smoothing_bins,
                taper_fraction=config.taper_fraction,
                maximum_lag=config.maximum_lag,
                coherence_floor=config.coherence_floor,
                minimum_lag_points=config.minimum_lag_points,
                consensus_count=config.consensus_count,
                stability_fraction=config.stability_fraction,
                stability_sigma=config.stability_sigma,
            )
            rows.append(
                {
                    "job_name": loaded.job_name,
                    config.scan_param: scan_value,
                    "readout": config.readout,
                    "orientation": orientation,
                    **estimate.as_dict(),
                }
            )
            for candidate in candidates:
                candidate_rows.append(
                    {
                        "job_name": loaded.job_name,
                        config.scan_param: scan_value,
                        "readout": config.readout,
                        **asdict(candidate),
                    }
                )

        rows.sort(key=lambda row: _sort_key(row[config.scan_param]))
        candidate_rows.sort(
            key=lambda row: (
                _sort_key(row[config.scan_param]),
                float(row["half_bandwidth"]),
            )
        )
        output_dir = getattr(self, "output_dir", None) or config.output_dir
        written: dict[str, str] = {}
        if output_dir is not None:
            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            if "carrier_results.csv" in config.export:
                path = write_table_csv(rows, destination / "carrier_results.csv")
                written["carrier_results"] = str(path)
            if "carrier_candidates.csv" in config.export:
                path = write_table_csv(
                    candidate_rows, destination / "carrier_candidates.csv"
                )
                written["carrier_candidates"] = str(path)
        metadata = orientation_metadata(
            reference_orientation or resolve_frequency_orientation(None)
        )
        return AnalysisResult(
            data_dict={
                "carrier_rows": rows,
                "candidate_rows": candidate_rows,
                "written": written,
                **metadata,
            },
            meta={
                "scan_param": config.scan_param,
                "readout": config.readout,
                "count": len(rows),
                "uncertainty_note": (
                    "regression_std is HAC phase-regression uncertainty; "
                    "bandwidth_std is estimator sensitivity, not trajectory SEM"
                ),
                **metadata,
            },
        )
