"""Tests for the typed coherence-frequency parent analyser execute path."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.coherence_frequency import (
    CoherenceFrequencyAnalyzer,
    CoherenceFrequencyAnalyzerConfig,
)
from qphase_sde.analyser.psd import PsdAnalyzer
from qphase_sde.coherence_frequency import BandLimitedCoherenceFrequency


def _typed_psd_product(scan_values: list[float]):
    """Typed PSD scan product for execute()."""
    from qphase_sde.products import add_scan_parameter_coordinates
    from qphase_sde.result import _trajectory_product
    from qphase_sde.state import TrajectorySet

    time = np.arange(2048) * 0.1
    points = []
    for value in scan_values:
        signal = np.exp(1j * (0.7 + 0.1 * value) * time)
        points.append(np.repeat(signal[None, :, None], 4, axis=0))
    trajectory = TrajectorySet(np.concatenate(points), t0=0.0, dt=0.1)
    product = _trajectory_product(
        trajectory, scan_size=len(scan_values), n_traj_per_point=4
    )
    psd = PsdAnalyzer(modes=[0], kind="complex").execute(
        {"trajectories": product}, backend=NumpyBackend(), label="psd"
    )["psd"]
    return add_scan_parameter_coordinates(
        psd, {"epsilon": np.asarray(scan_values, dtype=float)}
    )


def _analyzer() -> CoherenceFrequencyAnalyzer:
    return CoherenceFrequencyAnalyzer(
        CoherenceFrequencyAnalyzerConfig(
            scan_param="epsilon",
            input_key="psd",
            estimator={
                "band_limited": {
                    "freq_min": -2.0,
                    "freq_max": 0.2,
                    "maximum_lag": 100.0,
                }
            },
        ),
        subplugins={
            "estimator": BandLimitedCoherenceFrequency(
                freq_min=-2.0, freq_max=0.2, maximum_lag=100.0
            )
        },
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


def test_coherence_frequency_execute_aborts_mid_loop_on_cancellation():
    from types import SimpleNamespace

    analyzer = _analyzer()
    context = SimpleNamespace(
        cancellation=_FakeCancellation(abort_after=2), progress=None
    )

    with pytest.raises(_LoopAbortError):
        analyzer.execute(
            {"psd": _typed_psd_product([0.0, 1.0, 2.0, 3.0, 4.0])},
            backend=NumpyBackend(),
            label="frequency",
            context=context,
        )
    assert context.cancellation.calls < 5


def test_coherence_frequency_execute_reports_spectrum_progress():
    from types import SimpleNamespace

    analyzer = _analyzer()
    progress = _FakeProgress()
    context = SimpleNamespace(cancellation=None, progress=progress)

    products = analyzer.execute(
        {"psd": _typed_psd_product([0.0, 1.0, 2.0])},
        backend=NumpyBackend(),
        label="frequency",
        context=context,
    )

    assert products
    assert progress.updates
    completed = [update["completed"] for update in progress.updates]
    assert completed == sorted(completed)
    assert all(update["unit"] == "spectrum" for update in progress.updates)
    assert all(
        str(update["stage"]).startswith("analyze") for update in progress.updates
    )
    assert all(update["total"] is None for update in progress.updates)
