"""Tests for Rayleigh-matched short-delay carrier estimation."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.coherence_carrier import (
    CoherenceCarrierAnalyzer,
    CoherenceCarrierConfig,
    estimate_coherence_carrier,
)
from qphase_sde.state import TrajectorySet


def _oscillator_trajectories(
    frequencies: list[float],
    populations: list[int],
    *,
    dt: float = 0.01,
    n_samples: int = 256,
) -> TrajectorySet:
    time = np.arange(n_samples, dtype=float) * dt
    trajectories = []
    for mode, (frequency, count) in enumerate(
        zip(frequencies, populations, strict=True)
    ):
        for _ in range(count):
            values = np.zeros((n_samples, len(frequencies)), dtype=complex)
            values[:, mode] = np.exp(-1j * frequency * time)
            trajectories.append(values)
    return TrajectorySet(
        data=np.asarray(trajectories),
        dt=dt,
        meta={"mode_indices": list(range(len(frequencies)))},
    )


def test_short_delay_carrier_recovers_phase_orientation():
    dt = 0.02
    omega = 0.37
    lag = np.arange(13) * dt
    correlation = np.tile(np.exp(-1j * omega * lag), (8, 1))

    physical, _ = estimate_coherence_carrier(
        correlation,
        dt,
        orientation="phase_decreasing",
    )
    legacy, _ = estimate_coherence_carrier(
        correlation,
        dt,
        orientation="phase_increasing",
    )

    assert physical.frequency == pytest.approx(omega, abs=1e-12)
    assert legacy.frequency == pytest.approx(-omega, abs=1e-12)
    assert physical.first_lag_coherence == pytest.approx(1.0)


def test_nested_windows_detect_resolved_multifrequency_curvature():
    dt = 0.08
    lag = np.arange(13) * dt
    correlation = 0.7 * np.exp(-0.15j * lag) + 0.3 * np.exp(-1.2j * lag)
    per_trajectory = np.tile(correlation, (6, 1))

    result, candidates = estimate_coherence_carrier(
        per_trajectory,
        dt,
        polynomial_order=1,
        minimum_lag_points=3,
        maximum_lag_points=12,
    )

    assert result.frequency == pytest.approx(0.465, rel=0.02)
    assert result.selected_lag_points < 12
    assert len(candidates) == 10


def test_trace_carrier_equals_cam_rayleigh_quotient():
    frequencies = [0.2, 0.8]
    populations = [3, 2]
    trajectories = _oscillator_trajectories(frequencies, populations)
    analyzer = CoherenceCarrierAnalyzer(
        CoherenceCarrierConfig(
            modes=[0, 1],
            include_trace=True,
            maximum_lag_points=10,
        )
    )

    result = analyzer.analyze(trajectories, NumpyBackend()).data_dict

    expected_trace = (3.0 * frequencies[0] + 2.0 * frequencies[1]) / 5.0
    assert result["measurement_names"] == ["mode_0", "mode_1", "trace"]
    assert result["frequency"][0] == pytest.approx(frequencies[0], abs=1e-12)
    assert result["frequency"][1] == pytest.approx(frequencies[1], abs=1e-12)
    assert result["frequency"][2] == pytest.approx(expected_trace, abs=2e-5)
    assert "Tr[W*H(R)*R]" in result["cam_correspondence"]


def test_coherent_channel_uses_same_weight_matrix_as_cam():
    trajectories = _oscillator_trajectories([0.15, 0.65], [4, 4])
    analyzer = CoherenceCarrierAnalyzer(
        CoherenceCarrierConfig(
            include_trace=False,
            channels={"balanced": [1.0 + 0.0j, 1.0j]},
            maximum_lag_points=8,
        )
    )

    payload = analyzer.analyze(trajectories, NumpyBackend()).data_dict
    weight = np.asarray(payload["measurement_matrices"])[0]

    assert payload["measurement_names"] == ["balanced"]
    assert weight == pytest.approx(
        np.asarray([[0.5, -0.5j], [0.5j, 0.5]], dtype=complex)
    )
    assert payload["frequency"][0] == pytest.approx(0.4, abs=2e-5)


def test_trajectory_batch_accumulator_recomputes_ratio_and_uncertainty():
    trajectories = _oscillator_trajectories([0.2, 0.8], [3, 2])
    analyzer = CoherenceCarrierAnalyzer(
        CoherenceCarrierConfig(maximum_lag_points=8)
    )
    backend = NumpyBackend()
    complete = analyzer.analyze(trajectories, backend).data_dict
    accumulator = analyzer.create_result_accumulator()
    for indices in (slice(0, 3), slice(3, 5)):
        batch = TrajectorySet(
            data=trajectories.data[indices],
            dt=trajectories.dt,
            meta=dict(trajectories.meta),
        )
        accumulator.update(analyzer.analyze(batch, backend).data_dict)

    merged = accumulator.finalize()

    assert merged["frequency"] == pytest.approx(complete["frequency"])
    assert merged["frequency_sem"] == pytest.approx(complete["frequency_sem"])
    assert merged["n_traj"] == 5
    assert merged["uncertainty"]["independent_unit"] == "trajectory"


def test_config_rejects_under_determined_local_polynomial():
    with pytest.raises(ValueError, match="minimum_lag_points"):
        CoherenceCarrierConfig(
            polynomial_order=3,
            minimum_lag_points=3,
        )
