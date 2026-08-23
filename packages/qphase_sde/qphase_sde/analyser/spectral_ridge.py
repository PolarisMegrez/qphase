"""Scale-space spectral-ridge estimation from saved power spectra."""

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
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal import find_peaks

from .base import Analyzer, AnalyzerExecutionCapabilities
from .frequency_orientation import (
    FrequencyOrientation,
    orientation_metadata,
    resolve_frequency_orientation,
)
from .result import AnalysisResult
from .result_input import load_sde_results

Readout = int | Literal["trace"]

__all__ = [
    "SpectralRidgeAnalyzer",
    "SpectralRidgeConfig",
    "SpectralRidgeEstimate",
    "estimate_spectral_ridges",
]


def _default_readouts() -> list[Readout]:
    return ["trace"]


class SpectralRidgeConfig(PluginConfigBase):
    """Configuration for non-parametric PSD ridge extraction."""

    scan_param: str = Field(..., description="Parameter used as the scan axis")
    psd_key: str = Field("psd", description="Analysis key containing the PSD")
    readouts: list[Readout] = Field(
        default_factory=_default_readouts,
        description="Recorded bare modes and/or the incoherent trace",
    )
    freq_min: float | None = Field(None, description="Lower ridge search bound")
    freq_max: float | None = Field(None, description="Upper ridge search bound")
    maximum_profile_bins: int = Field(
        4096,
        ge=128,
        description="Maximum bins retained for scale-space analysis",
    )
    smoothing_scale_bins: list[float] = Field(
        default_factory=lambda: [2.0, 4.0, 8.0, 16.0],
        description="Gaussian scale-space standard deviations in reduced bins",
    )
    local_window_scale: float = Field(
        4.0,
        ge=2.0,
        description="Local quadratic half-window in smoothing standard deviations",
    )
    maximum_candidates: int = Field(
        4,
        ge=1,
        le=16,
        description="Strongest local maxima retained at every scale",
    )
    minimum_scale_support: int = Field(
        2,
        ge=1,
        description="Distinct smoothing scales required for a ridge candidate",
    )
    minimum_prominence_fraction: float = Field(
        0.002,
        ge=0.0,
        lt=1.0,
        description="Minimum prominence relative to baseline-subtracted peak height",
    )
    cluster_scale_factor: float = Field(
        3.0,
        gt=0.0,
        description="Frequency clustering radius relative to the largest scale",
    )
    confidence_sigma: float = Field(
        2.0,
        gt=0.0,
        description="PSD-SEM threshold used for the peak confidence interval",
    )
    frequency_bin_covariance: Literal["diagonal", "conservative"] = Field(
        "diagonal",
        description=(
            "PSD-bin covariance approximation used when reducing and smoothing SEM"
        ),
    )
    plateau_fraction: float = Field(
        0.95,
        gt=0.0,
        lt=1.0,
        description="Relative peak height defining the descriptive plateau",
    )
    tracking_enabled: bool = Field(
        True,
        description="Select a ridge path using only scan continuity and peak evidence",
    )
    tracking_frequency_scale: float | None = Field(
        None,
        gt=0.0,
        description="Frequency scale for scan-path transition cost",
    )
    tracking_gap_factor: float | None = Field(
        None,
        gt=1.0,
        description=(
            "Split tracking when a scan-axis gap exceeds this times the median gap"
        ),
    )
    tracking_weight: float = Field(1.0, ge=0.0)
    export: list[str] = Field(
        default_factory=lambda: [
            "spectral_ridge.csv",
            "spectral_ridge_candidates.csv",
        ]
    )
    output_dir: str | None = Field(None, description="Usually injected by the engine")
    pattern: str = Field("*.npz", description="Glob for saved result inputs")

    @field_validator("readouts")
    @classmethod
    def validate_readouts(cls, values: list[Readout]) -> list[Readout]:
        if not values:
            raise ValueError("readouts must not be empty")
        if any(isinstance(value, int) and value < 0 for value in values):
            raise ValueError("readout modes must be non-negative")
        keys = [str(value) for value in values]
        if len(keys) != len(set(keys)):
            raise ValueError("readouts must contain unique selections")
        return values

    @field_validator("smoothing_scale_bins")
    @classmethod
    def validate_scales(cls, values: list[float]) -> list[float]:
        if len(values) < 2 or any(
            not np.isfinite(value) or value <= 0.0 for value in values
        ):
            raise ValueError("smoothing_scale_bins must contain at least two positives")
        if any(right <= left for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("smoothing_scale_bins must be strictly increasing")
        return values

    @model_validator(mode="after")
    def validate_ranges(self) -> SpectralRidgeConfig:
        if (
            self.freq_min is not None
            and self.freq_max is not None
            and self.freq_min >= self.freq_max
        ):
            raise ValueError("freq_min must be smaller than freq_max")
        if self.minimum_scale_support > len(self.smoothing_scale_bins):
            raise ValueError(
                "minimum_scale_support cannot exceed the number of smoothing scales"
            )
        return self


@dataclass(frozen=True)
class _ScalePeak:
    scale_bins: float
    frequency: float
    value: float
    relative_height: float
    frequency_std: float
    curvature: float
    curvature_std: float
    local_r2: float


@dataclass(frozen=True)
class SpectralRidgeEstimate:
    """One scale-supported spectral-ridge candidate."""

    frequency: float = math.nan
    frequency_std: float = math.nan
    local_frequency_std: float = math.nan
    scale_frequency_std: float = math.nan
    peak_value: float = math.nan
    relative_height: float = math.nan
    curvature: float = math.nan
    curvature_std: float = math.nan
    curvature_significance: float = math.nan
    scale_support: int = 0
    confidence_lower: float = math.nan
    confidence_upper: float = math.nan
    plateau_lower: float = math.nan
    plateau_upper: float = math.nan
    score: float = math.nan
    status: str = "failed"
    error: str = ""


def _reduce_profile(
    frequency: np.ndarray,
    spectrum: np.ndarray,
    sem: np.ndarray | None,
    maximum_bins: int,
    covariance: Literal["diagonal", "conservative"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    count = frequency.size
    block = max(1, int(math.ceil(count / maximum_bins)))
    if block == 1:
        return frequency, spectrum, sem
    starts = np.arange(0, count, block)
    sizes = np.diff(np.append(starts, count))
    reduced_frequency = np.add.reduceat(frequency, starts) / sizes
    reduced_spectrum = np.add.reduceat(spectrum, starts) / sizes
    reduced_sem = None
    if sem is not None:
        if covariance == "diagonal":
            reduced_sem = np.sqrt(np.add.reduceat(sem**2, starts)) / sizes
        else:
            reduced_sem = np.add.reduceat(np.abs(sem), starts) / sizes
    return reduced_frequency, reduced_spectrum, reduced_sem


def _smooth_sem(
    sem: np.ndarray,
    scale: float,
    covariance: Literal["diagonal", "conservative"],
) -> np.ndarray:
    if covariance == "conservative":
        return gaussian_filter1d(np.abs(sem), scale, mode="nearest")
    radius = max(1, int(4.0 * scale + 0.5))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    weights = np.exp(-0.5 * (offsets / scale) ** 2)
    weights /= np.sum(weights)
    return np.sqrt(convolve1d(sem**2, weights**2, mode="nearest"))


def _quadratic_peak(
    frequency: np.ndarray,
    spectrum: np.ndarray,
    sem: np.ndarray | None,
    peak_index: int,
    scale_bins: float,
    local_window_scale: float,
) -> _ScalePeak | None:
    radius = max(3, int(math.ceil(local_window_scale * scale_bins)))
    start = max(0, peak_index - radius)
    stop = min(frequency.size, peak_index + radius + 1)
    if stop - start < 5:
        return None
    x = frequency[start:stop]
    y = spectrum[start:stop]
    origin = float(frequency[peak_index])
    dx = x - origin
    design = np.column_stack((np.ones(dx.size), dx, dx**2))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    constant, linear, quadratic = coefficients
    if not np.all(np.isfinite(coefficients)) or quadratic >= 0.0:
        return None
    offset = float(-linear / (2.0 * quadratic))
    if offset < float(dx[0]) or offset > float(dx[-1]):
        return None
    location = origin + offset
    value = float(constant + linear * offset + quadratic * offset**2)
    prediction = design @ coefficients
    residual = y - prediction
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    local_r2 = 1.0 - float(residual @ residual) / max(denominator, 1e-300)

    influence = np.linalg.pinv(design)
    if sem is not None and np.all(np.isfinite(sem[start:stop])):
        local_sem = np.asarray(sem[start:stop], dtype=float)
        covariance = (influence * local_sem[np.newaxis, :]) @ (
            influence * local_sem[np.newaxis, :]
        ).T
    else:
        dof = max(1, dx.size - design.shape[1])
        covariance = float(residual @ residual) / dof * (influence @ influence.T)
    gradient = np.asarray(
        [0.0, -1.0 / (2.0 * quadratic), linear / (2.0 * quadratic**2)]
    )
    location_variance = float(gradient @ covariance @ gradient)
    frequency_std = (
        math.sqrt(location_variance)
        if np.isfinite(location_variance) and location_variance >= 0.0
        else math.nan
    )
    curvature = float(-2.0 * quadratic)
    curvature_variance = float(4.0 * covariance[2, 2])
    curvature_std = (
        math.sqrt(curvature_variance)
        if np.isfinite(curvature_variance) and curvature_variance >= 0.0
        else math.nan
    )
    baseline = float(np.quantile(spectrum, 0.05))
    height = max(float(np.max(spectrum)) - baseline, np.finfo(float).tiny)
    relative_height = max(0.0, (value - baseline) / height)
    return _ScalePeak(
        scale_bins=scale_bins,
        frequency=location,
        value=value,
        relative_height=relative_height,
        frequency_std=frequency_std,
        curvature=curvature,
        curvature_std=curvature_std,
        local_r2=local_r2,
    )


def _contiguous_interval(
    frequency: np.ndarray, mask: np.ndarray, center_index: int
) -> tuple[float, float]:
    left = center_index
    right = center_index
    while left > 0 and bool(mask[left - 1]):
        left -= 1
    while right + 1 < mask.size and bool(mask[right + 1]):
        right += 1
    return float(frequency[left]), float(frequency[right])


def estimate_spectral_ridges(
    frequency: np.ndarray,
    spectrum: np.ndarray,
    *,
    sem: np.ndarray | None = None,
    maximum_profile_bins: int = 4096,
    smoothing_scale_bins: list[float] | tuple[float, ...] = (2.0, 4.0, 8.0, 16.0),
    local_window_scale: float = 4.0,
    maximum_candidates: int = 4,
    minimum_scale_support: int = 2,
    minimum_prominence_fraction: float = 0.002,
    cluster_scale_factor: float = 3.0,
    confidence_sigma: float = 2.0,
    plateau_fraction: float = 0.95,
    frequency_bin_covariance: Literal["diagonal", "conservative"] = "diagonal",
) -> list[SpectralRidgeEstimate]:
    """Return scale-supported local maxima without using a model frequency."""
    axis = np.asarray(frequency, dtype=float).reshape(-1)
    values = np.asarray(spectrum, dtype=float).reshape(-1)
    errors = None if sem is None else np.asarray(sem, dtype=float).reshape(-1)
    if axis.size < 16 or values.size != axis.size:
        raise ValueError("frequency and spectrum must have equal length >= 16")
    if errors is not None and errors.size != axis.size:
        raise ValueError("PSD SEM must match the spectrum length")
    spacing = np.diff(axis)
    if np.any(spacing <= 0.0):
        raise ValueError("frequency must be strictly increasing")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("spectrum must be finite and non-negative")
    axis, values, errors = _reduce_profile(
        axis,
        values,
        errors,
        maximum_profile_bins,
        frequency_bin_covariance,
    )
    step = float(np.median(np.diff(axis)))
    baseline = float(np.quantile(values, 0.05))
    height = max(float(np.max(values)) - baseline, np.finfo(float).tiny)
    prominence = minimum_prominence_fraction * height

    scale_peaks: list[_ScalePeak] = []
    profiles: dict[float, tuple[np.ndarray, np.ndarray | None]] = {}
    for scale in smoothing_scale_bins:
        smoothed = gaussian_filter1d(values, float(scale), mode="nearest")
        smoothed_sem = (
            None
            if errors is None
            else _smooth_sem(errors, float(scale), frequency_bin_covariance)
        )
        profiles[float(scale)] = (smoothed, smoothed_sem)
        indices, properties = find_peaks(smoothed, prominence=prominence)
        if indices.size == 0:
            indices = np.asarray([int(np.argmax(smoothed))])
            prominences = np.asarray([height])
        else:
            prominences = np.asarray(properties["prominences"])
        order = np.argsort(prominences)[::-1][:maximum_candidates]
        for peak_index in indices[order]:
            refined = _quadratic_peak(
                axis,
                smoothed,
                smoothed_sem,
                int(peak_index),
                float(scale),
                local_window_scale,
            )
            if refined is not None:
                scale_peaks.append(refined)
    if not scale_peaks:
        return [SpectralRidgeEstimate(error="no concave scale-space maxima")]

    cluster_radius = cluster_scale_factor * max(smoothing_scale_bins) * step
    clusters: list[list[_ScalePeak]] = []
    for peak in sorted(scale_peaks, key=lambda item: item.frequency):
        if not clusters:
            clusters.append([peak])
            continue
        centers = [
            float(np.median([item.frequency for item in group])) for group in clusters
        ]
        nearest = int(np.argmin(np.abs(np.asarray(centers) - peak.frequency)))
        if abs(centers[nearest] - peak.frequency) <= cluster_radius:
            clusters[nearest].append(peak)
        else:
            clusters.append([peak])

    estimates: list[SpectralRidgeEstimate] = []
    for cluster in clusters:
        scales = {item.scale_bins for item in cluster}
        if len(scales) < minimum_scale_support:
            continue
        locations = np.asarray([item.frequency for item in cluster])
        local_std = float(np.nanmedian([item.frequency_std for item in cluster]))
        scale_std = float(np.std(locations, ddof=1)) if locations.size > 1 else 0.0
        total_std = math.sqrt(
            max(0.0, local_std if np.isfinite(local_std) else 0.0) ** 2 + scale_std**2
        )
        frequency_estimate = float(np.median(locations))
        representative = min(
            cluster,
            key=lambda item: abs(item.frequency - frequency_estimate),
        )
        profile, profile_sem = profiles[representative.scale_bins]
        peak_index = int(np.argmin(np.abs(axis - frequency_estimate)))
        profile_peak = float(profile[peak_index])
        profile_baseline = float(np.quantile(profile, 0.05))
        profile_height = max(profile_peak - profile_baseline, np.finfo(float).tiny)
        plateau_mask = profile >= profile_baseline + plateau_fraction * profile_height
        plateau_lower, plateau_upper = _contiguous_interval(
            axis, plateau_mask, peak_index
        )
        confidence_lower = math.nan
        confidence_upper = math.nan
        if profile_sem is not None and np.all(np.isfinite(profile_sem)):
            threshold = confidence_sigma * (profile_sem[peak_index] + profile_sem)
            confidence_mask = profile_peak - profile <= threshold
            confidence_lower, confidence_upper = _contiguous_interval(
                axis, confidence_mask, peak_index
            )
        curvature = float(np.median([item.curvature for item in cluster]))
        curvature_std = float(np.nanmedian([item.curvature_std for item in cluster]))
        significance = (
            curvature / curvature_std
            if np.isfinite(curvature_std) and curvature_std > 0.0
            else math.nan
        )
        relative_height = float(np.median([item.relative_height for item in cluster]))
        support_fraction = len(scales) / len(smoothing_scale_bins)
        score = relative_height * support_fraction / (1.0 + scale_std / cluster_radius)
        status = "ok"
        if np.isfinite(significance) and significance < 2.0:
            status = "flat_top"
        estimates.append(
            SpectralRidgeEstimate(
                frequency=frequency_estimate,
                frequency_std=total_std,
                local_frequency_std=local_std,
                scale_frequency_std=scale_std,
                peak_value=float(np.median([item.value for item in cluster])),
                relative_height=relative_height,
                curvature=curvature,
                curvature_std=curvature_std,
                curvature_significance=significance,
                scale_support=len(scales),
                confidence_lower=confidence_lower,
                confidence_upper=confidence_upper,
                plateau_lower=plateau_lower,
                plateau_upper=plateau_upper,
                score=score,
                status=status,
            )
        )
    if not estimates:
        strongest = max(scale_peaks, key=lambda item: item.relative_height)
        return [
            SpectralRidgeEstimate(
                frequency=strongest.frequency,
                frequency_std=strongest.frequency_std,
                local_frequency_std=strongest.frequency_std,
                peak_value=strongest.value,
                relative_height=strongest.relative_height,
                curvature=strongest.curvature,
                curvature_std=strongest.curvature_std,
                curvature_significance=(
                    strongest.curvature / strongest.curvature_std
                    if strongest.curvature_std > 0.0
                    else math.nan
                ),
                scale_support=1,
                score=strongest.relative_height,
                status="insufficient_scale_support",
            )
        ]
    return sorted(estimates, key=lambda item: item.score, reverse=True)


def _track_candidates(
    candidates: list[list[SpectralRidgeEstimate]],
    *,
    frequency_scale: float,
    tracking_weight: float,
) -> list[int]:
    if not candidates or any(not point for point in candidates):
        return [0 for _ in candidates]
    costs = -np.log(
        np.maximum(
            np.asarray([item.score for item in candidates[0]], dtype=float),
            1e-12,
        )
    )
    backpointers: list[np.ndarray] = []
    for point in range(1, len(candidates)):
        previous = candidates[point - 1]
        current = candidates[point]
        transition = np.empty((len(previous), len(current)))
        for left, source in enumerate(previous):
            for right, target in enumerate(current):
                transition[left, right] = (
                    tracking_weight
                    * ((target.frequency - source.frequency) / frequency_scale) ** 2
                )
        total = costs[:, None] + transition
        pointer = np.argmin(total, axis=0)
        evidence = -np.log(
            np.maximum(np.asarray([item.score for item in current]), 1e-12)
        )
        costs = total[pointer, np.arange(len(current))] + evidence
        backpointers.append(pointer)
    selected = [int(np.argmin(costs))]
    for pointer in reversed(backpointers):
        selected.append(int(pointer[selected[-1]]))
    return list(reversed(selected))


def _tracking_segments(
    scan_values: np.ndarray, gap_factor: float | None
) -> list[slice]:
    if gap_factor is None or scan_values.size < 3:
        return [slice(0, scan_values.size)]
    gaps = np.abs(np.diff(scan_values))
    positive = gaps[gaps > 0.0]
    if positive.size == 0:
        return [slice(0, scan_values.size)]
    typical = float(np.median(positive))
    breaks = np.flatnonzero(gaps > gap_factor * typical) + 1
    boundaries = [0, *breaks.tolist(), scan_values.size]
    return [
        slice(left, right)
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
        if right > left
    ]


def _extract_readout(
    analysis: dict[str, Any],
    psd_key: str,
    readout: Readout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any], str]:
    if psd_key not in analysis:
        raise QPhaseError(f"analysis is missing PSD payload {psd_key!r}")
    payload = analysis[psd_key]
    axis = np.asarray(payload["axis"], dtype=float).reshape(-1)
    matrix = np.asarray(payload["psd"], dtype=float)
    sem_matrix = (
        np.asarray(payload["psd_sem"], dtype=float) if "psd_sem" in payload else None
    )
    if matrix.ndim == 1:
        return axis, matrix, sem_matrix, payload, "direct_psd_sem"
    if matrix.ndim != 2:
        raise QPhaseError("PSD payload has an unsupported shape")
    if readout == "trace":
        spectrum = np.sum(matrix, axis=1)
        sem = None if sem_matrix is None else np.sum(np.abs(sem_matrix), axis=1)
        return axis, spectrum, sem, payload, "mode_sem_upper_bound"
    modes = [int(mode) for mode in payload.get("modes", [])]
    if readout in modes:
        column = modes.index(readout)
    elif readout < matrix.shape[1]:
        column = readout
    else:
        raise QPhaseError(f"mode {readout} is absent from PSD payload")
    sem = None if sem_matrix is None else sem_matrix[:, column]
    return axis, matrix[:, column], sem, payload, "direct_psd_sem"


class SpectralRidgeAnalyzer(Analyzer):
    """Extract model-independent spectral ridges from logical PSD scans."""

    name: ClassVar[str] = "spectral_ridge"
    description: ClassVar[str] = (
        "Scale-space spectral ridges with curvature diagnostics"
    )
    config_schema: ClassVar[type[SpectralRidgeConfig]] = SpectralRidgeConfig

    def capabilities(self) -> AnalyzerExecutionCapabilities:
        return AnalyzerExecutionCapabilities(
            execution_location="host", requires_full_trajectory=False
        )

    def analyze(self, data: Any, backend: BackendBase) -> ResultProtocol:
        del backend
        config = cast(SpectralRidgeConfig, self.config)
        loaded = load_sde_results(data, config.pattern)
        if not loaded:
            raise QPhaseError("spectral_ridge received no input results")
        point_data: dict[str, list[tuple[float, Any, list[SpectralRidgeEstimate]]]] = {
            str(readout): [] for readout in config.readouts
        }
        reference_orientation: FrequencyOrientation | None = None
        uncertainty_sources: set[str] = set()
        for item in loaded:
            params = item.meta.get("params", {})
            if config.scan_param not in params:
                raise QPhaseError(
                    f"input is missing scan parameter {config.scan_param!r}"
                )
            scan_value = float(params[config.scan_param])
            for readout in config.readouts:
                axis, spectrum, sem, payload, source = _extract_readout(
                    item.analysis, config.psd_key, readout
                )
                orientation = resolve_frequency_orientation(payload)
                if reference_orientation is None:
                    reference_orientation = orientation
                elif reference_orientation != orientation:
                    raise QPhaseError("PSD inputs use different frequency orientations")
                mask = np.ones(axis.shape, dtype=bool)
                if config.freq_min is not None:
                    mask &= axis >= config.freq_min
                if config.freq_max is not None:
                    mask &= axis <= config.freq_max
                if int(np.count_nonzero(mask)) < 16:
                    raise QPhaseError(
                        "spectral ridge search window has fewer than 16 bins"
                    )
                estimates = estimate_spectral_ridges(
                    axis[mask],
                    spectrum[mask],
                    sem=None if sem is None else sem[mask],
                    maximum_profile_bins=config.maximum_profile_bins,
                    smoothing_scale_bins=config.smoothing_scale_bins,
                    local_window_scale=config.local_window_scale,
                    maximum_candidates=config.maximum_candidates,
                    minimum_scale_support=config.minimum_scale_support,
                    minimum_prominence_fraction=config.minimum_prominence_fraction,
                    cluster_scale_factor=config.cluster_scale_factor,
                    confidence_sigma=config.confidence_sigma,
                    plateau_fraction=config.plateau_fraction,
                    frequency_bin_covariance=config.frequency_bin_covariance,
                )
                point_data[str(readout)].append((scan_value, item.job_name, estimates))
                uncertainty_sources.add(
                    f"{source}+frequency_{config.frequency_bin_covariance}"
                    if sem is not None
                    else "fit_residual"
                )

        rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        for readout in config.readouts:
            readout_key = str(readout)
            points = sorted(point_data[readout_key], key=lambda item: item[0])
            candidate_sets = [item[2] for item in points]
            if config.tracking_enabled:
                frequency_scale = config.tracking_frequency_scale
                if frequency_scale is None:
                    widths = [
                        estimate.plateau_upper - estimate.plateau_lower
                        for estimates in candidate_sets
                        for estimate in estimates
                        if np.isfinite(estimate.plateau_upper)
                        and np.isfinite(estimate.plateau_lower)
                        and estimate.plateau_upper > estimate.plateau_lower
                    ]
                    # Plateau width is a full width; use its half-width as the
                    # natural transition scale for one ridge.
                    frequency_scale = (
                        0.5 * float(np.median(widths)) if widths else 1.0
                    )
                selected = []
                scan_values = np.asarray([item[0] for item in points], dtype=float)
                for segment in _tracking_segments(
                    scan_values, config.tracking_gap_factor
                ):
                    selected.extend(
                        _track_candidates(
                            candidate_sets[segment],
                            frequency_scale=max(
                                frequency_scale, np.finfo(float).eps
                            ),
                            tracking_weight=config.tracking_weight,
                        )
                    )
            else:
                selected = [0 for _ in points]
            for point_index, ((scan_value, job_name, estimates), chosen) in enumerate(
                zip(points, selected, strict=True)
            ):
                chosen = min(chosen, len(estimates) - 1)
                estimate = estimates[chosen]
                measurement_name = "trace" if readout == "trace" else f"mode_{readout}"
                row = {
                    "point_index": point_index,
                    "job_name": job_name,
                    config.scan_param: scan_value,
                    "readout": readout,
                    "measurement_name": measurement_name,
                    "measurement_kind": (
                        "incoherent_trace" if readout == "trace" else "bare_mode"
                    ),
                    "frequency": estimate.frequency,
                    "frequency_std": estimate.frequency_std,
                    "local_frequency_std": estimate.local_frequency_std,
                    "scale_frequency_std": estimate.scale_frequency_std,
                    "peak_value": estimate.peak_value,
                    "relative_height": estimate.relative_height,
                    "curvature": estimate.curvature,
                    "curvature_std": estimate.curvature_std,
                    "curvature_significance": estimate.curvature_significance,
                    "scale_support": estimate.scale_support,
                    "confidence_lower": estimate.confidence_lower,
                    "confidence_upper": estimate.confidence_upper,
                    "confidence_width": (
                        estimate.confidence_upper - estimate.confidence_lower
                    ),
                    "plateau_lower": estimate.plateau_lower,
                    "plateau_upper": estimate.plateau_upper,
                    "plateau_width": estimate.plateau_upper - estimate.plateau_lower,
                    "selected_candidate": chosen,
                    "candidate_count": len(estimates),
                    "status": estimate.status,
                    "error": estimate.error,
                }
                rows.append(row)
                for candidate_index, candidate in enumerate(estimates):
                    candidate_rows.append(
                        {
                            "point_index": point_index,
                            config.scan_param: scan_value,
                            "measurement_name": measurement_name,
                            "candidate_index": candidate_index,
                            "selected": candidate_index == chosen,
                            **asdict(candidate),
                        }
                    )

        written: dict[str, str] = {}
        output_dir = getattr(self, "output_dir", None) or config.output_dir
        if output_dir is not None:
            destination = Path(output_dir)
            if "spectral_ridge.csv" in config.export:
                path = write_table_csv(rows, destination / "spectral_ridge.csv")
                written["spectral_ridge"] = str(path)
            if "spectral_ridge_candidates.csv" in config.export:
                path = write_table_csv(
                    candidate_rows,
                    destination / "spectral_ridge_candidates.csv",
                )
                written["spectral_ridge_candidates"] = str(path)
        metadata = orientation_metadata(
            reference_orientation or resolve_frequency_orientation(None)
        )
        return AnalysisResult(
            data_dict={
                "ridge_rows": rows,
                "candidate_rows": candidate_rows,
                "written": written,
                **metadata,
            },
            meta={
                "scan_param": config.scan_param,
                "readouts": config.readouts,
                "method": "Gaussian scale-space plus local quadratic ridge",
                "selection": (
                    "peak evidence plus optional scan continuity; no model target"
                ),
                "uncertainty_sources": sorted(uncertainty_sources),
                "frequency_bin_covariance": config.frequency_bin_covariance,
                **metadata,
            },
        )
