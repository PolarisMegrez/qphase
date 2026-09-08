"""Band-limited coherence numerical core."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BandLimitedCarrierCandidate:
    """One nested-bandwidth phase-regression result."""

    half_bandwidth: float
    frequency: float = math.nan
    regression_std: float = math.nan
    phase_fit_rms: float = math.nan
    frequency_drift: float = math.nan
    decay_rate: float = math.nan
    lag_points: int = 0
    lag_start: float = math.nan
    lag_end: float = math.nan
    touched_maximum_lag: bool = False
    status: str = "failed"
    error: str = ""


@dataclass(frozen=True)
class BandLimitedCarrierPlatform:
    """One frequency platform supported across a finite bandwidth range."""

    frequency: float
    regression_std: float
    bandwidth_std: float
    diagnostic_uncertainty: float
    score: float
    first_candidate: int
    last_candidate: int
    candidate_count: int
    minimum_half_bandwidth: float
    maximum_half_bandwidth: float
    log_bandwidth_span: float
    phase_fit_rms: float
    frequency_drift: float
    decay_rate: float
    lag_start: float
    lag_end: float
    touched_maximum_lag: bool


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
    lag_start: float = math.nan
    lag_end: float = math.nan
    frequency_drift: float = math.nan
    decay_rate: float = math.nan
    touched_maximum_lag: bool = False
    platforms: tuple[BandLimitedCarrierPlatform, ...] = ()
    status: str = "failed"
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("platforms")
        result["platform_count"] = len(self.platforms)
        return result


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


def _ridge_local_width(
    axis: np.ndarray,
    spectrum: np.ndarray,
    *,
    center: float,
    baseline: float,
    smoothing_bins: int,
    spacing: float,
) -> float:
    """Return the half-height component width containing a fixed ridge center."""
    smoothed = _moving_average(spectrum, smoothing_bins)
    index = int(np.argmin(np.abs(axis - center)))
    height = float(smoothed[index] - baseline)
    if not np.isfinite(height) or height <= 0.0:
        return 16.0 * spacing
    half_height = baseline + 0.5 * height
    left = index
    while left > 0 and smoothed[left - 1] >= half_height:
        left -= 1
    right = index
    while right + 1 < axis.size and smoothed[right + 1] >= half_height:
        right += 1
    return max(0.5 * float(axis[right] - axis[left]), 16.0 * spacing)


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
    minimum_lag_span: float | None,
    lag_window_trials: int,
    max_phase_fit_rms: float,
    max_frequency_drift_fraction: float,
    max_negative_decay_fraction: float,
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
    baseband_correlation = correlation * np.exp(-1j * center * lag)
    amplitude = np.abs(correlation) / max(
        abs(complex(correlation[0])), np.finfo(float).tiny
    )
    usable = np.flatnonzero(
        (lag > 0.0) & (lag <= maximum_lag) & (amplitude >= coherence_floor)
    )
    segments = _contiguous_segments(usable)
    minimum_points = max(
        minimum_lag_points,
        int(math.ceil((minimum_lag_span or 0.1 * maximum_lag) / lag_step)) + 1,
    )
    segments = [segment for segment in segments if segment.size >= minimum_points]
    if not segments:
        return BandLimitedCarrierCandidate(
            half_bandwidth=half_bandwidth,
            lag_points=int(max((item.size for item in segments), default=0)),
            error="usable coherence interval is too short",
        )
    accepted: list[tuple[tuple[float, float, float], BandLimitedCarrierCandidate]] = []
    diagnostics: list[tuple[float, BandLimitedCarrierCandidate]] = []
    for segment in segments:
        windows = _lag_windows(segment, minimum_points, lag_window_trials)
        for indices in windows:
            candidate = _fit_phase_window(
                lag,
                baseband_correlation,
                amplitude,
                indices,
                half_bandwidth=half_bandwidth,
                maximum_lag=maximum_lag,
                frequency_offset=center,
            )
            duration = candidate.lag_end - candidate.lag_start
            diagnostics.append((duration, candidate))
            if (
                candidate.phase_fit_rms <= max_phase_fit_rms
                and candidate.frequency_drift
                <= max_frequency_drift_fraction * half_bandwidth
                and candidate.decay_rate
                >= -max_negative_decay_fraction * half_bandwidth
            ):
                # Prefer the longest interval, then the latest interval, then
                # the smallest phase residual. This targets a resolved
                # long-time plateau without rewarding short accidental fits.
                key = (duration, candidate.lag_end, -candidate.phase_fit_rms)
                accepted.append((key, candidate))
    if not accepted:
        _, diagnostic = max(diagnostics, key=lambda item: item[0])
        return BandLimitedCarrierCandidate(
            **{
                **asdict(diagnostic),
                "status": "nonlinear_phase",
                "error": "no lag interval passed phase residual and curvature gates",
            }
        )
    _, best = max(accepted, key=lambda item: item[0])
    return BandLimitedCarrierCandidate(**{**asdict(best), "status": "ok", "error": ""})


def _contiguous_segments(indices: np.ndarray) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    return [item for item in np.split(indices, boundaries) if item.size]


def _lag_windows(
    segment: np.ndarray, minimum_points: int, trials: int
) -> list[np.ndarray]:
    maximum_start = segment.size - minimum_points
    starts = np.unique(
        np.linspace(0, maximum_start, min(trials, maximum_start + 1), dtype=int)
    )
    windows: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()
    for start in starts:
        minimum_end = start + minimum_points
        ends = np.unique(
            np.linspace(
                minimum_end,
                segment.size,
                min(trials, segment.size - minimum_end + 1),
                dtype=int,
            )
        )
        for end in ends:
            key = (int(start), int(end))
            if key not in seen:
                windows.append(segment[start:end])
                seen.add(key)
    return windows


def _fit_phase_window(
    lag: np.ndarray,
    correlation: np.ndarray,
    amplitude: np.ndarray,
    indices: np.ndarray,
    *,
    half_bandwidth: float,
    maximum_lag: float,
    frequency_offset: float,
) -> BandLimitedCarrierCandidate:
    time = lag[indices]
    phase = np.unwrap(np.angle(correlation[indices]))
    weights = amplitude[indices] ** 2
    root_weight = np.sqrt(weights)
    linear = np.column_stack((np.ones(time.size), time))
    coefficient, *_ = np.linalg.lstsq(
        linear * root_weight[:, None], phase * root_weight, rcond=None
    )
    residual = phase - linear @ coefficient
    rms = math.sqrt(float(np.sum(weights * residual**2) / np.sum(weights)))
    centered_time = time - float(np.mean(time))
    quadratic = np.column_stack((np.ones(time.size), centered_time, centered_time**2))
    curve, *_ = np.linalg.lstsq(
        quadratic * root_weight[:, None], phase * root_weight, rcond=None
    )
    frequency_drift = abs(2.0 * float(curve[2]) * float(np.ptp(time)))
    log_amplitude = np.log(np.maximum(amplitude[indices], np.finfo(float).tiny))
    decay, *_ = np.linalg.lstsq(
        linear * root_weight[:, None], log_amplitude * root_weight, rcond=None
    )
    lag_step = float(lag[1] - lag[0])
    return BandLimitedCarrierCandidate(
        half_bandwidth=half_bandwidth,
        frequency=frequency_offset + float(coefficient[1]),
        regression_std=_hac_slope_std(linear, weights, residual),
        phase_fit_rms=rms,
        frequency_drift=frequency_drift,
        decay_rate=-float(decay[1]),
        lag_points=int(indices.size),
        lag_start=float(time[0]),
        lag_end=float(time[-1]),
        touched_maximum_lag=float(time[-1]) >= maximum_lag - 1.5 * lag_step,
        status="diagnostic",
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


def _select_platforms(
    candidates: list[BandLimitedCarrierCandidate],
    *,
    consensus_count: int,
    minimum_log_bandwidth_span: float,
    stability_fraction: float,
    stability_sigma: float,
    platform_ambiguity_delta: float,
) -> tuple[tuple[BandLimitedCarrierPlatform, ...], str]:
    valid_indices = [
        index for index, candidate in enumerate(candidates) if candidate.status == "ok"
    ]
    groups: list[tuple[set[int], BandLimitedCarrierPlatform]] = []
    for start_position, start in enumerate(valid_indices):
        for stop in valid_indices[start_position + consensus_count - 1 :]:
            indices = [index for index in valid_indices if start <= index <= stop]
            if len(indices) < consensus_count:
                continue
            group = [candidates[index] for index in indices]
            log_span = math.log(group[-1].half_bandwidth / group[0].half_bandwidth)
            if log_span < minimum_log_bandwidth_span:
                continue
            bandwidth = np.asarray([item.half_bandwidth for item in group])
            frequency = np.asarray([item.frequency for item in group])
            regression = np.asarray([item.regression_std for item in group])
            floor = stability_fraction * float(np.median(bandwidth))
            variance = np.maximum(regression, 0.0) ** 2 + floor**2
            weights = 1.0 / variance
            center = float(np.sum(weights * frequency) / np.sum(weights))
            tolerance = floor + stability_sigma * max(
                float(np.median(regression)), floor
            )
            normalized = np.abs(frequency - center) / tolerance
            if np.any(normalized > 1.0):
                continue
            bandwidth_std = float(
                np.sqrt(np.sum(weights * (frequency - center) ** 2) / np.sum(weights))
            )
            regression_std = math.sqrt(1.0 / float(np.sum(weights)))
            uncertainty = math.hypot(regression_std, bandwidth_std)
            phase_rms = float(np.median([item.phase_fit_rms for item in group]))
            score = (
                float(np.sqrt(np.mean(normalized**2)))
                + phase_rms
                + float(np.median(regression / max(floor, 1e-15)))
                - 0.15 * log_span
                - 0.03 * len(group)
            )
            groups.append(
                (
                    set(indices),
                    BandLimitedCarrierPlatform(
                        frequency=center,
                        regression_std=regression_std,
                        bandwidth_std=bandwidth_std,
                        diagnostic_uncertainty=uncertainty,
                        score=score,
                        first_candidate=indices[0],
                        last_candidate=indices[-1],
                        candidate_count=len(group),
                        minimum_half_bandwidth=float(bandwidth[0]),
                        maximum_half_bandwidth=float(bandwidth[-1]),
                        log_bandwidth_span=log_span,
                        phase_fit_rms=phase_rms,
                        frequency_drift=float(
                            np.median([item.frequency_drift for item in group])
                        ),
                        decay_rate=float(
                            np.median([item.decay_rate for item in group])
                        ),
                        lag_start=float(np.median([item.lag_start for item in group])),
                        lag_end=float(np.median([item.lag_end for item in group])),
                        touched_maximum_lag=any(
                            item.touched_maximum_lag for item in group
                        ),
                    ),
                )
            )
    if not groups:
        return (), "no_bandwidth_plateau"
    maximal = [
        (indices, platform)
        for indices, platform in groups
        if not any(indices < other for other, _ in groups)
    ]
    representatives: list[BandLimitedCarrierPlatform] = []
    for _, platform in sorted(maximal, key=lambda item: item[1].score):
        same = False
        for retained in representatives:
            tolerance = (
                platform.diagnostic_uncertainty
                + retained.diagnostic_uncertainty
                + stability_fraction
                * min(
                    platform.minimum_half_bandwidth,
                    retained.minimum_half_bandwidth,
                )
            )
            if abs(platform.frequency - retained.frequency) <= tolerance:
                same = True
                break
        if not same:
            representatives.append(platform)
    platforms = tuple(sorted(representatives, key=lambda item: item.score))
    if (
        len(platforms) > 1
        and platforms[1].score <= platforms[0].score + platform_ambiguity_delta
    ):
        return platforms, "ambiguous_multiband"
    return platforms, "ok"


def estimate_band_limited_carrier(
    axis: np.ndarray,
    spectrum: np.ndarray,
    *,
    freq_min: float | None = None,
    freq_max: float | None = None,
    bandwidth_multipliers: list[float] | tuple[float, ...] = (
        0.5,
        0.625,
        0.75,
        0.875,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
    ),
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
    minimum_lag_span: float | None = None,
    lag_window_trials: int = 16,
    max_phase_fit_rms: float = 0.05,
    max_frequency_drift_fraction: float = 0.05,
    max_negative_decay_fraction: float = 0.02,
    consensus_count: int = 3,
    minimum_log_bandwidth_span: float = 0.2,
    stability_fraction: float = 0.015,
    stability_sigma: float = 2.0,
    platform_ambiguity_delta: float = 0.5,
    fixed_center: float | None = None,
    fixed_bandwidth_max: float | None = None,
) -> tuple[BandLimitedCarrierEstimate, list[BandLimitedCarrierCandidate]]:
    """Estimate a filtered carrier from a resolved lag-bandwidth platform.

    Candidate windows are scaled by the standard deviation of the squared,
    baseline-subtracted spectrum. For an ideal Lorentz profile this equals the
    HWHM. A carrier is returned only when phase-linear lag intervals support a
    unique frequency platform across a finite logarithmic bandwidth span.
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
        if fixed_center is None:
            peak_center, peak_hwhm = _connected_peak(
                target_axis,
                target_power,
                concentration_center=concentration_center,
                concentration_width=spectral_width,
                baseline=baseline,
                smoothing_bins=peak_smoothing_bins,
                spacing=spacing,
            )
        else:
            if not float(target_axis[0]) < fixed_center < float(target_axis[-1]):
                raise ValueError("fixed ridge center lies outside the search range")
            peak_center = float(fixed_center)
            peak_hwhm = _ridge_local_width(
                target_axis,
                target_power,
                center=peak_center,
                baseline=baseline,
                smoothing_bins=peak_smoothing_bins,
                spacing=spacing,
            )
            spectral_width = peak_hwhm
            effective_max = (
                fixed_bandwidth_max
                if bandwidth_max is None
                else min(fixed_bandwidth_max, bandwidth_max)
                if fixed_bandwidth_max is not None
                else bandwidth_max
            )
            if effective_max is not None:
                spectral_width = min(
                    spectral_width,
                    effective_max / max(bandwidth_multipliers),
                )
        bandwidths = _candidate_bandwidths(
            peak_center,
            spectral_width,
            float(target_axis[0]),
            float(target_axis[-1]),
            spacing,
            bandwidth_multipliers,
            bandwidth_min,
            (
                fixed_bandwidth_max
                if bandwidth_max is None
                else min(fixed_bandwidth_max, bandwidth_max)
                if fixed_bandwidth_max is not None
                else bandwidth_max
            ),
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
                minimum_lag_span=minimum_lag_span,
                lag_window_trials=lag_window_trials,
                max_phase_fit_rms=max_phase_fit_rms,
                max_frequency_drift_fraction=max_frequency_drift_fraction,
                max_negative_decay_fraction=max_negative_decay_fraction,
            )
            for half_bandwidth in bandwidths
        ]
        platforms, status = _select_platforms(
            candidates,
            consensus_count=consensus_count,
            minimum_log_bandwidth_span=minimum_log_bandwidth_span,
            stability_fraction=stability_fraction,
            stability_sigma=stability_sigma,
            platform_ambiguity_delta=platform_ambiguity_delta,
        )
        selected = (
            min(platforms, key=lambda item: abs(item.frequency - fixed_center))
            if fixed_center is not None and platforms
            else platforms[0]
            if platforms
            else None
        )
        if fixed_center is not None and selected is not None:
            status = "ok"
        group = (
            candidates[selected.first_candidate : selected.last_candidate + 1]
            if selected is not None
            else []
        )
        group = [item for item in group if item.status == "ok"]
        estimate = BandLimitedCarrierEstimate(
            frequency=(
                selected.frequency
                if selected is not None and status == "ok"
                else math.nan
            ),
            regression_std=(selected.regression_std if selected else math.nan),
            bandwidth_std=(selected.bandwidth_std if selected else math.nan),
            diagnostic_uncertainty=(
                selected.diagnostic_uncertainty if selected else math.nan
            ),
            peak_center=peak_center,
            peak_hwhm=peak_hwhm,
            spectral_width=spectral_width,
            selected_half_bandwidth=(
                math.sqrt(
                    selected.minimum_half_bandwidth * selected.maximum_half_bandwidth
                )
                if selected
                else math.nan
            ),
            consensus_count=(selected.candidate_count if selected else 0),
            phase_fit_rms=(selected.phase_fit_rms if selected else math.nan),
            lag_points=(
                int(np.median([item.lag_points for item in group])) if group else 0
            ),
            lag_start=(selected.lag_start if selected else math.nan),
            lag_end=(selected.lag_end if selected else math.nan),
            frequency_drift=(
                float(np.median([item.frequency_drift for item in group]))
                if group
                else math.nan
            ),
            decay_rate=(
                float(np.median([item.decay_rate for item in group]))
                if group
                else math.nan
            ),
            touched_maximum_lag=any(item.touched_maximum_lag for item in group),
            platforms=platforms,
            status=status,
            error=(
                ""
                if status == "ok"
                else "carrier is not uniquely resolved across lag and bandwidth"
            ),
        )
        return estimate, candidates
    except Exception as exc:
        return BandLimitedCarrierEstimate(error=str(exc)), []


def track_band_limited_carrier(
    scan_values: np.ndarray,
    estimates: list[BandLimitedCarrierEstimate],
    *,
    frequency_scale: float | None = None,
    curvature_weight: float = 1.0,
    ambiguity_delta: float = 1.0,
    max_normalized_curvature: float = 3.0,
) -> list[dict[str, Any]]:
    """Track a data-supported carrier platform across an ordered scan.

    The path cost uses local platform quality and divided-difference curvature.
    It does not use a theoretical carrier, target exponent, or CAM result.
    """
    values = np.asarray(scan_values, dtype=float).reshape(-1)
    if values.size != len(estimates):
        raise ValueError("scan values and carrier estimates have different lengths")
    if values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError("scan values must be strictly increasing for tracking")
    output = [
        {
            "tracked_frequency": math.nan,
            "tracked_diagnostic_uncertainty": math.nan,
            "tracked_platform_index": -1,
            "tracked_status": "unavailable",
            "tracked_path_cost_gap": math.nan,
        }
        for _ in estimates
    ]
    available = np.asarray([bool(item.platforms) for item in estimates], dtype=bool)
    indices = np.flatnonzero(available)
    for segment in _contiguous_segments(indices):
        segment_estimates = [estimates[int(index)] for index in segment]
        widths = [item.spectral_width for item in segment_estimates]
        uncertainties = [
            platform.diagnostic_uncertainty
            for item in segment_estimates
            for platform in item.platforms
            if np.isfinite(platform.diagnostic_uncertainty)
        ]
        scale = frequency_scale
        if scale is None:
            width_scale = 0.25 * float(np.median(widths))
            uncertainty_scale = (
                3.0 * float(np.median(uncertainties)) if uncertainties else 0.0
            )
            scale = max(width_scale, uncertainty_scale, np.finfo(float).eps)
        path, cost_gap, normalized_curvature = _track_platform_segment(
            values[segment],
            segment_estimates,
            frequency_scale=float(scale),
            curvature_weight=curvature_weight,
        )
        path_status = "ok" if cost_gap > ambiguity_delta else "ambiguous_path"
        for source_index, platform_index in zip(segment, path, strict=True):
            path_position = int(np.flatnonzero(segment == source_index)[0])
            if normalized_curvature[path_position] > max_normalized_curvature:
                output[int(source_index)]["tracked_status"] = "discontinuous_path"
                output[int(source_index)]["tracked_path_cost_gap"] = cost_gap
                continue
            platform = estimates[int(source_index)].platforms[platform_index]
            output[int(source_index)] = {
                "tracked_frequency": platform.frequency,
                "tracked_diagnostic_uncertainty": platform.diagnostic_uncertainty,
                "tracked_platform_index": int(platform_index),
                "tracked_status": path_status,
                "tracked_path_cost_gap": cost_gap,
            }
    return output


def _track_platform_segment(
    scan_values: np.ndarray,
    estimates: list[BandLimitedCarrierEstimate],
    *,
    frequency_scale: float,
    curvature_weight: float,
) -> tuple[list[int], float, np.ndarray]:
    count = len(estimates)
    local_costs = []
    for estimate in estimates:
        scores = np.asarray([item.score for item in estimate.platforms])
        local_costs.append(scores - float(np.min(scores)))
    if count == 1:
        order = np.argsort(local_costs[0])
        gap = (
            float(local_costs[0][order[1]] - local_costs[0][order[0]])
            if order.size > 1
            else math.inf
        )
        return [int(order[0])], gap, np.zeros(1)

    states: dict[tuple[int, int], list[tuple[float, list[int]]]] = {}
    step = float(scan_values[1] - scan_values[0])
    for first, first_platform in enumerate(estimates[0].platforms):
        for second, second_platform in enumerate(estimates[1].platforms):
            jump = (
                second_platform.frequency - first_platform.frequency
            ) / frequency_scale
            cost = float(local_costs[0][first] + local_costs[1][second])
            cost += 0.02 * jump**2 * min(abs(step), 1.0) / max(abs(step), 1e-15)
            states.setdefault((first, second), []).append((cost, [first, second]))
    for point in range(2, count):
        current: dict[tuple[int, int], list[tuple[float, list[int]]]] = {}
        left_step = float(scan_values[point - 1] - scan_values[point - 2])
        right_step = float(scan_values[point] - scan_values[point - 1])
        for (previous, middle), histories in states.items():
            previous_frequency = estimates[point - 2].platforms[previous].frequency
            middle_frequency = estimates[point - 1].platforms[middle].frequency
            slope = (middle_frequency - previous_frequency) / left_step
            predicted = middle_frequency + slope * right_step
            for cost, path in histories:
                for following, platform in enumerate(estimates[point].platforms):
                    curvature = (platform.frequency - predicted) / frequency_scale
                    candidate_cost = (
                        cost
                        + float(local_costs[point][following])
                        + curvature_weight * curvature**2
                    )
                    key = (middle, following)
                    retained = current.setdefault(key, [])
                    retained.append((candidate_cost, [*path, following]))
                    retained.sort(key=lambda item: item[0])
                    del retained[2:]
        states = current
    ranked = sorted(
        (item for histories in states.values() for item in histories),
        key=lambda item: item[0],
    )
    gap = ranked[1][0] - ranked[0][0] if len(ranked) > 1 else math.inf
    path = ranked[0][1]
    normalized_curvature = np.zeros(count, dtype=float)
    for point in range(2, count):
        left_step = float(scan_values[point - 1] - scan_values[point - 2])
        right_step = float(scan_values[point] - scan_values[point - 1])
        previous_frequency = estimates[point - 2].platforms[path[point - 2]].frequency
        middle_frequency = estimates[point - 1].platforms[path[point - 1]].frequency
        following_frequency = estimates[point].platforms[path[point]].frequency
        predicted = (
            middle_frequency
            + (middle_frequency - previous_frequency) / left_step * right_step
        )
        normalized_curvature[point] = (
            abs(following_frequency - predicted) / frequency_scale
        )
    return path, float(gap), normalized_curvature
