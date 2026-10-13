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
from qphase_sde.contracts.analyser import AnalyserWorkspaceRequest
from qphase_sde.result import _trajectory_product
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


def _product_payload(product):
    payload = {}
    values = list(product.attributes.get("payload_meta", {}).items()) + [
        (variable.name, product.handle(variable.name).materialize())
        for variable in product.variables
    ]
    for name, value in values:
        if np.asarray(value).ndim == 0:
            value = np.asarray(value).item()
        target = payload
        parts = name.split(".")
        for part in parts[:-1]:
            key = int(part) if part.isdigit() else part
            target = target.setdefault(key, {})
        target[parts[-1]] = value
    return payload


def _execute(analyzer, trajectories: TrajectorySet, backend=None):
    product = analyzer.execute(
        {"trajectories": _typed_trajectories(trajectories)},
        backend=backend or NumpyBackend(),
        label="carrier",
    )["carrier"]
    return product, _product_payload(product)


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

    _, result = _execute(analyzer, trajectories)

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

    _, payload = _execute(analyzer, trajectories)
    weight = np.asarray(payload["measurement_matrices"])[0]

    assert payload["measurement_names"] == ["balanced"]
    assert weight == pytest.approx(
        np.asarray([[0.5, -0.5j], [0.5j, 0.5]], dtype=complex)
    )
    assert payload["frequency"][0] == pytest.approx(0.4, abs=2e-5)


def test_trajectory_batch_accumulator_recomputes_ratio_and_uncertainty():
    trajectories = _oscillator_trajectories([0.2, 0.8], [3, 2])
    analyzer = CoherenceCarrierAnalyzer(CoherenceCarrierConfig(maximum_lag_points=8))
    backend = NumpyBackend()
    _, complete = _execute(analyzer, trajectories, backend)
    accumulator = analyzer.create_result_accumulator()
    for indices in (slice(0, 3), slice(3, 5)):
        batch = TrajectorySet(
            data=trajectories.data[indices],
            dt=trajectories.dt,
            meta=dict(trajectories.meta),
        )
        partial, _ = _execute(analyzer, batch, backend)
        accumulator.consume_product(partial)

    products = accumulator.finalize_product("carrier")
    assert products is not None
    merged = _product_payload(products["carrier"])

    assert merged["frequency"] == pytest.approx(complete["frequency"])
    assert merged["frequency_sem"] == pytest.approx(complete["frequency_sem"])
    assert merged["n_traj"] == 5
    assert merged["uncertainty"]["independent_unit"] == "trajectory"


def test_coherence_carrier_invocation_workspace_tracks_physical_batch():
    analyzer = CoherenceCarrierAnalyzer(
        CoherenceCarrierConfig(
            modes=[0, 1],
            channels={"balanced": [1.0, 1.0, 0.0]},
            maximum_lag_points=10,
            time_chunk_samples=128,
        )
    )
    common = dict(
        trajectory_bytes=1,
        saved_samples=512,
        n_record_modes=3,
        real_itemsize=8,
        backend_name="cupy",
        total_n_traj=128,
    )

    small = analyzer.workspace(AnalyserWorkspaceRequest(n_traj=32, **common))
    large = analyzer.workspace(AnalyserWorkspaceRequest(n_traj=64, **common))

    measurement_matrices = 4 * 3 * 3 * 2 * 8
    assert (small.device_bytes - measurement_matrices) * 2 == (
        large.device_bytes - measurement_matrices
    )
    assert small.host_bytes == large.host_bytes == 0


def test_coherence_carrier_retained_workspace_covers_ensemble_and_batches():
    analyzer = CoherenceCarrierAnalyzer(
        CoherenceCarrierConfig(
            modes=[0, 1],
            channels={"balanced": [1.0, 1.0, 0.0]},
            maximum_lag_points=10,
        )
    )
    estimate = analyzer.workspace(
        AnalyserWorkspaceRequest(
            trajectory_bytes=1,
            n_traj=32,
            total_n_traj=100,
            saved_samples=512,
            n_record_modes=3,
            real_itemsize=8,
            backend_name="numpy",
        )
    )
    trajectory_payload = 100 * 4 * 11 * 2 * 8
    fixed_payload = 4 * 3 * 3 * 2 * 8 + 3 * 8 + 4 * 128

    assert estimate.retained_host_bytes == 2 * trajectory_payload + 4 * fixed_payload
    assert estimate.retained_device_bytes == 0


def test_config_rejects_under_determined_local_polynomial():
    with pytest.raises(ValueError, match="minimum_lag_points"):
        CoherenceCarrierConfig(
            polynomial_order=3,
            minimum_lag_points=3,
        )


class _LoopAbortError(Exception):
    """Raised by the fake cancellation token mid-loop."""


class _FakeCancellation:
    """Cancellation token that raises after a fixed number of polls."""

    def __init__(self, abort_after: int) -> None:
        self.calls = 0
        self._abort_after = abort_after

    def raise_if_cancelled(self) -> None:
        self.calls += 1
        if self.calls > self._abort_after:
            raise _LoopAbortError


class _FakeProgress:
    """Progress reporter recording structured updates."""

    def __init__(self) -> None:
        self.updates: list[dict] = []

    def update(
        self,
        completed=None,
        total=None,
        *,
        unit=None,
        stage=None,
        message="",
        metadata=None,
    ):
        self.updates.append(
            {
                "completed": completed,
                "total": total,
                "unit": unit,
                "stage": stage,
                "message": message,
            }
        )


def _typed_trajectories(trajectories: TrajectorySet):
    return _trajectory_product(trajectories, scan_size=None, n_traj_per_point=None)


def test_coherence_carrier_execute_aborts_mid_lag_loop_on_cancellation():
    from types import SimpleNamespace

    trajectories = _oscillator_trajectories([0.2, 0.8], [3, 2])
    analyzer = CoherenceCarrierAnalyzer(
        CoherenceCarrierConfig(modes=[0, 1], maximum_lag_points=24)
    )
    context = SimpleNamespace(
        cancellation=_FakeCancellation(abort_after=3), progress=None
    )

    with pytest.raises(_LoopAbortError):
        analyzer.execute(
            {"trajectories": _typed_trajectories(trajectories)},
            backend=NumpyBackend(),
            label="carrier",
            context=context,
        )
    # 1 map_trajectory_product poll + 25 lag polls + 2 measurement polls bound
    # a complete run; the abort must cut the loop well before that.
    assert context.cancellation.calls < 1 + 25 + 2 + 1


def test_coherence_carrier_execute_reports_sample_progress():
    from types import SimpleNamespace

    trajectories = _oscillator_trajectories([0.2, 0.8], [3, 2])
    analyzer = CoherenceCarrierAnalyzer(
        CoherenceCarrierConfig(modes=[0, 1], maximum_lag_points=8)
    )
    progress = _FakeProgress()
    context = SimpleNamespace(cancellation=None, progress=progress)

    products = analyzer.execute(
        {"trajectories": _typed_trajectories(trajectories)},
        backend=NumpyBackend(),
        label="carrier",
        context=context,
    )

    assert products
    assert progress.updates
    completed = [update["completed"] for update in progress.updates]
    assert completed == sorted(completed)
    assert all(update["unit"] == "sample" for update in progress.updates)
    assert all(
        str(update["stage"]).startswith("analyze") for update in progress.updates
    )
    assert all(update["total"] is None for update in progress.updates)
