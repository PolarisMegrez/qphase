"""Finite-delay carrier numerical core."""

from __future__ import annotations

import math

import numpy as np


def finite_delay_carrier_from_spectrum(
    axis: np.ndarray,
    spectrum: np.ndarray,
    detector_rates: np.ndarray,
    *,
    maximum_lag: float,
    tail_time_constants: float,
) -> dict[str, np.ndarray | float]:
    """Evaluate the coherence-weighted phase velocity from a full PSD."""
    frequency = np.asarray(axis, dtype=float).reshape(-1)
    power = np.asarray(spectrum, dtype=float).reshape(-1)
    rates = np.asarray(detector_rates, dtype=float).reshape(-1)
    if frequency.size < 4 or power.size != frequency.size:
        raise ValueError("axis and spectrum must have equal length of at least four")
    spacing = np.diff(frequency)
    step = float(np.median(spacing))
    if step <= 0.0 or not np.allclose(spacing, step, rtol=1e-6, atol=0.0):
        raise ValueError("frequency axis must be uniformly increasing")
    if not np.all(np.isfinite(power)) or np.any(power < 0.0):
        raise ValueError("spectrum must be finite and non-negative")
    if np.sum(power) <= np.finfo(float).tiny:
        raise ValueError("spectrum has no positive power")

    lag_step = 2.0 * np.pi / (frequency.size * step)
    lag = np.arange(frequency.size, dtype=float) * lag_step
    phase_shift = np.exp(1j * frequency[0] * lag)
    coherence = np.fft.ifft(power) * phase_shift
    derivative = np.fft.ifft(1j * frequency * power) * phase_shift
    denominator_zero = abs(complex(coherence[0])) ** 2
    instantaneous = float(
        np.imag(np.conj(coherence[0]) * derivative[0]) / denominator_zero
    )

    carrier = np.full(rates.shape, np.nan)
    coherent_weight = np.full(rates.shape, np.nan)
    lag_end = np.full(rates.shape, np.nan)
    sample_count = np.zeros(rates.shape, dtype=int)
    for index, rate in enumerate(rates):
        limit = min(maximum_lag, tail_time_constants / (2.0 * rate))
        count = min(
            frequency.size,
            max(2, int(math.floor(limit / lag_step)) + 1),
        )
        time = lag[:count]
        weight = np.exp(-2.0 * rate * time)
        denominator = float(np.trapezoid(weight * np.abs(coherence[:count]) ** 2, time))
        numerator = float(
            np.trapezoid(
                weight * np.imag(np.conj(coherence[:count]) * derivative[:count]),
                time,
            )
        )
        if denominator > np.finfo(float).tiny:
            carrier[index] = numerator / denominator
            coherent_weight[index] = denominator
        lag_end[index] = float(time[-1])
        sample_count[index] = count
    return {
        "instantaneous_frequency": instantaneous,
        "frequency": carrier,
        "coherent_weight": coherent_weight,
        "lag_end": lag_end,
        "sample_count": sample_count,
        "sample_lag": lag_step,
    }
