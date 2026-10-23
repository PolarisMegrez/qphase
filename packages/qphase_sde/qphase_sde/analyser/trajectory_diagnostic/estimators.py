"""Numeric estimators behind the trajectory diagnostics analyser.

Every function takes arrays (plus the analyser configuration where a knob
matters) and returns plain payload dictionaries; none of them touch the
product layer, so the children in :mod:`..diagnostics` stay thin.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from ...contracts.quantities import FrequencyOrientation, orientation_sign
from ...numerics.coordinates import (
    CANONICAL_COORDINATE_LAYOUT,
    R_CONVENTION,
    canonical_vector,
)
from ..frequency_orientation import orient_spectrum, orientation_metadata
from .config import (
    _BLOCK_SPECTRUM_MIN_BLOCKS,
    _BLOCK_SPECTRUM_MIN_SAMPLES,
    _G1_HOLDOUT_MISMATCH_ATOL,
    _G1_HOLDOUT_MISMATCH_FACTOR,
    _G1_MIN_FINITE_LAGS,
    _G1_MIN_SIGNAL_LAGS,
    _G1_NOISE_FLOOR_FACTOR,
    _G1_PREFERRED_RELATIVE_TOL,
    _STATIONARITY_MAX_CONDITION,
    _STATIONARITY_MIN_BLOCKS,
    _STATIONARITY_RADIAL_QUANTILES,
    TrajectoryDiagnosticsConfig,
)


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
    orientation: FrequencyOrientation,
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
            frequency = orientation_sign(orientation) * _masked_mean(
                increments, valid, axis=2
            )
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
                **orientation_metadata(orientation),
            }
        )
    return results


def _phase_increment_summary(
    series: np.ndarray,
    dt: float,
    amplitude_floor: float,
    orientation: FrequencyOrientation,
) -> dict[str, Any]:
    amplitude = np.abs(series)
    increments = np.angle(series[:, 1:] * np.conj(series[:, :-1]))
    valid = (amplitude[:, 1:] > amplitude_floor) & (amplitude[:, :-1] > amplitude_floor)
    absolute = np.abs(increments)
    return {
        "mean_angular_frequency_per_trajectory": _masked_mean(
            orientation_sign(orientation) * increments / dt, valid, axis=1
        ),
        "max_abs_phase_step_per_trajectory": np.max(
            np.where(valid, absolute, np.nan), axis=1
        ),
        "near_nyquist_fraction_per_trajectory": _masked_mean(
            (absolute >= 0.9 * np.pi).astype(float), valid, axis=1
        ),
        "near_nyquist_threshold": 0.9 * np.pi,
        **orientation_metadata(orientation),
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
    orientation: FrequencyOrientation,
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
    frequencies, psd = orient_spectrum(frequencies, psd, orientation=orientation)
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
        **orientation_metadata(orientation),
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
    orientation: FrequencyOrientation,
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
            entry.update(_block_spectrum_features(blocks, dt, wing_factor, orientation))
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
                orientation=config.orientation,
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
            blocks = trimmed.reshape(n_traj, n_blocks, block_samples, n_coordinates)
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
