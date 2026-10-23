"""Scale-space ridge estimation and candidate-path tracking.

The estimation entry point turns a stack of saved spectra into per-scan ridge
locations plus their fitted widths; the tracking helpers link those per-scan
candidates into continuous paths with a Huber path cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal import find_peaks


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

@dataclass(frozen=True)
class _TrackingPath:
    indices: tuple[int, ...]
    cost: float

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

def _huber_loss(value: float, delta: float) -> float:
    magnitude = abs(value)
    if magnitude <= delta:
        return 0.5 * magnitude**2
    return delta * (magnitude - 0.5 * delta)

def _track_candidate_paths(
    candidates: list[list[SpectralRidgeEstimate]],
    *,
    frequency_scale: float,
    tracking_weight: float,
    huber_delta: float,
    path_count: int,
) -> list[_TrackingPath]:
    if not candidates or any(not point for point in candidates):
        return [_TrackingPath(tuple(0 for _ in candidates), math.inf)]
    states: list[list[_TrackingPath]] = [
        [
            _TrackingPath(
                (candidate_index,),
                -math.log(max(candidate.score, 1e-12)),
            )
        ]
        for candidate_index, candidate in enumerate(candidates[0])
    ]
    for point in range(1, len(candidates)):
        previous = candidates[point - 1]
        current = candidates[point]
        next_states: list[list[_TrackingPath]] = []
        for current_index, target in enumerate(current):
            options: list[_TrackingPath] = []
            evidence = -math.log(max(target.score, 1e-12))
            for previous_index, source in enumerate(previous):
                transition = tracking_weight * _huber_loss(
                    (target.frequency - source.frequency) / frequency_scale,
                    huber_delta,
                )
                for path in states[previous_index]:
                    options.append(
                        _TrackingPath(
                            (*path.indices, current_index),
                            path.cost + transition + evidence,
                        )
                    )
            options.sort(key=lambda path: path.cost)
            unique: list[_TrackingPath] = []
            seen: set[tuple[int, ...]] = set()
            for path in options:
                if path.indices in seen:
                    continue
                seen.add(path.indices)
                unique.append(path)
                if len(unique) >= path_count:
                    break
            next_states.append(unique)
        states = next_states
    complete = sorted(
        (path for state in states for path in state),
        key=lambda path: path.cost,
    )
    output: list[_TrackingPath] = []
    seen = set()
    for path in complete:
        if path.indices in seen:
            continue
        seen.add(path.indices)
        output.append(path)
        if len(output) >= path_count:
            break
    return output

def _track_candidates(
    candidates: list[list[SpectralRidgeEstimate]],
    *,
    frequency_scale: float,
    tracking_weight: float,
    huber_delta: float = 1.0,
) -> list[int]:
    paths = _track_candidate_paths(
        candidates,
        frequency_scale=frequency_scale,
        tracking_weight=tracking_weight,
        huber_delta=huber_delta,
        path_count=1,
    )
    if not paths:
        return [0 for _ in candidates]
    return list(paths[0].indices)

def _candidate_retention_tiers(
    estimates: list[SpectralRidgeEstimate],
    accepted_path_indices: set[int],
    *,
    relative_height_fraction: float,
    minimum_scale_support: int,
    minimum_curvature_significance: float,
    continuity_rescue: bool,
) -> list[str]:
    strongest = max(candidate.relative_height for candidate in estimates)
    high = [
        candidate.relative_height >= relative_height_fraction * strongest
        for candidate in estimates
    ]
    strict = {
        index
        for index, candidate in enumerate(estimates)
        if high[index]
        and candidate.scale_support >= minimum_scale_support
        and candidate.curvature_significance >= minimum_curvature_significance
    }
    rescued = (
        {
            index
            for index in accepted_path_indices
            if index < len(estimates) and high[index]
        }
        if continuity_rescue
        else set()
    )
    retained = strict | rescued
    fallback: int | None = None
    if not retained:
        fallback = int(np.argmax([candidate.score for candidate in estimates]))
        retained.add(fallback)
    tiers: list[str] = []
    for index in range(len(estimates)):
        if index in strict:
            tiers.append("strict")
        elif index in rescued:
            tiers.append("continuity_rescued")
        elif index == fallback:
            tiers.append("fallback_low_confidence")
        else:
            tiers.append("excluded")
    return tiers

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
