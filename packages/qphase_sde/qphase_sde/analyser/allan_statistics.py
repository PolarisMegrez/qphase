"""Shared angular-frequency Allan statistics."""

from __future__ import annotations

import math

import numpy as np

__all__ = ["calculate_allan_variance", "summarize_trajectories"]


def _masked_mean(values: np.ndarray, mask: np.ndarray, *, axis: int) -> np.ndarray:
    count = np.sum(mask, axis=axis)
    total = np.sum(np.where(mask, values, 0.0), axis=axis)
    result = np.full(np.shape(total), np.nan, dtype=float)
    np.divide(total, count, out=result, where=count > 0)
    return result


def _masked_square_mean(
    values: np.ndarray, mask: np.ndarray, *, axis: int
) -> np.ndarray:
    count = np.sum(mask, axis=axis)
    np.square(values, out=values)
    values[~mask] = 0.0
    total = np.sum(values, axis=axis)
    result = np.full(np.shape(total), np.nan, dtype=float)
    np.divide(total, count, out=result, where=count > 0)
    return result


def _duration_samples(value: float, dt: float) -> int:
    samples = int(round(value / dt))
    if samples < 1 or not math.isclose(
        samples * dt, value, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(
            f"Allan tau {value:.12g} is not aligned with sample spacing {dt:.12g}"
        )
    return samples


def summarize_trajectories(
    per_trajectory: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ensemble mean, SEM, and finite trajectory count by tau."""
    finite = np.isfinite(per_trajectory)
    sample_count = np.sum(finite, axis=0)
    mean = _masked_mean(per_trajectory, finite, axis=0)
    sem = np.full(per_trajectory.shape[1], np.nan, dtype=float)
    for column, count in enumerate(sample_count):
        if count > 1:
            standard_deviation = np.std(
                per_trajectory[finite[:, column], column], ddof=1
            )
            sem[column] = standard_deviation / math.sqrt(count)
    return mean, sem, sample_count


def calculate_allan_variance(
    series: np.ndarray,
    dt: float,
    *,
    taus: list[float] | None,
    points: int,
    min_windows: int,
    min_independent_windows: int,
    amplitude_floor: float,
) -> dict[str, object]:
    """Compute overlapping and non-overlapping angular-frequency Allan variance."""
    n_traj, n_time = series.shape
    max_m_overlapping = (n_time - min_windows) // 2
    max_m_independent = (n_time - 1) // (2 * min_independent_windows)
    max_m = min(max_m_overlapping, max_m_independent)
    if max_m < 1:
        raise ValueError(
            "trajectory is too short for the requested Allan window constraints"
        )
    if taus is None:
        candidates = np.geomspace(1, max_m, num=points)
        averaging_samples = np.unique(np.rint(candidates).astype(int))
    else:
        averaging_samples = np.unique(
            np.asarray([_duration_samples(value, dt) for value in taus], dtype=int)
        )
        if averaging_samples[-1] > max_m:
            raise ValueError(
                "an Allan tau violates allan_min_windows or "
                "allan_min_independent_windows"
            )

    amplitude = np.abs(series)
    phase_increments = np.angle(series[:, 1:] * np.conj(series[:, :-1]))
    valid_increments = (amplitude[:, 1:] > amplitude_floor) & (
        amplitude[:, :-1] > amplitude_floor
    )
    phase_increments[~valid_increments] = 0.0
    phase = np.empty((n_traj, n_time), dtype=float)
    phase[:, 0] = 0.0
    np.cumsum(phase_increments, axis=1, out=phase[:, 1:])
    count_dtype = np.int32 if n_time <= np.iinfo(np.int32).max else np.int64
    valid_cumulative = np.empty((n_traj, n_time), dtype=count_dtype)
    valid_cumulative[:, 0] = 0
    np.cumsum(
        valid_increments,
        axis=1,
        dtype=count_dtype,
        out=valid_cumulative[:, 1:],
    )
    del amplitude, phase_increments, valid_increments

    n_tau = len(averaging_samples)
    per_trajectory = np.full((n_traj, n_tau), np.nan, dtype=float)
    valid_counts = np.zeros((n_traj, n_tau), dtype=np.int64)
    nonoverlap_per_trajectory = np.full((n_traj, n_tau), np.nan, dtype=float)
    nonoverlap_valid_counts = np.zeros((n_traj, n_tau), dtype=np.int64)
    nominal_independent = np.empty(n_tau, dtype=np.int64)

    for column, m in enumerate(averaging_samples):
        tau = m * dt
        delta = phase[:, 2 * m :] - 2.0 * phase[:, m:-m] + phase[:, : -2 * m]
        valid = valid_cumulative[:, 2 * m :] - valid_cumulative[:, : -2 * m]
        valid = valid == 2 * m
        valid_counts[:, column] = np.sum(valid, axis=1)
        per_trajectory[:, column] = _masked_square_mean(delta, valid, axis=1) / (
            2.0 * tau**2
        )

        starts = np.arange(0, n_time - 2 * m, 2 * m, dtype=np.int64)
        nominal_independent[column] = starts.size
        nonoverlap_delta = (
            phase[:, starts + 2 * m] - 2.0 * phase[:, starts + m] + phase[:, starts]
        )
        nonoverlap_valid = (
            valid_cumulative[:, starts + 2 * m] - valid_cumulative[:, starts]
        ) == 2 * m
        nonoverlap_valid_counts[:, column] = np.sum(nonoverlap_valid, axis=1)
        nonoverlap_per_trajectory[:, column] = _masked_square_mean(
            nonoverlap_delta, nonoverlap_valid, axis=1
        ) / (2.0 * tau**2)

    mean, sem, sample_count = summarize_trajectories(per_trajectory)
    nonoverlap_mean, nonoverlap_sem, nonoverlap_sample_count = summarize_trajectories(
        nonoverlap_per_trajectory
    )
    tau = averaging_samples.astype(float) * dt
    return {
        "quantity": "allan_variance",
        "variable": "angular_frequency",
        "tau_unit": "seconds",
        "tau": tau,
        "angular_frequency_variance": mean,
        "angular_frequency_variance_sem": sem,
        "per_trajectory": per_trajectory,
        "valid_second_differences": valid_counts,
        "trajectory_sample_count": sample_count,
        "nonoverlap_angular_frequency_variance": nonoverlap_mean,
        "nonoverlap_angular_frequency_variance_sem": nonoverlap_sem,
        "nonoverlap_per_trajectory": nonoverlap_per_trajectory,
        "nonoverlap_valid_second_differences": nonoverlap_valid_counts,
        "nonoverlap_trajectory_sample_count": nonoverlap_sample_count,
        "nominal_independent_windows_per_trajectory": nominal_independent,
        "total_independent_window_count": np.sum(
            nonoverlap_valid_counts, axis=0, dtype=np.int64
        ),
        "observation_duration": float((n_time - 1) * dt),
        "min_independent_windows": int(min_independent_windows),
        "definition": "overlapping_phase_second_difference",
        "independent_definition": "nonoverlapping_phase_second_difference",
    }
