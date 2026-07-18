"""Smoke tests for qphase_viz plotters (headless Agg rendering)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qphase_viz.plotters.evolution import TimeSeriesPlotter
from qphase_viz.plotters.parameter import ParameterEvolutionPlotter
from qphase_viz.plotters.phase import PhasePlanePlotter
from qphase_viz.plotters.spectrum import PowerSpectrumPlotter


class _FakeTrajectory:
    """Minimal TrajectorySet-like object (no qphase_sde dependency)."""

    def __init__(self, data: np.ndarray, dt: float = 0.1):
        self._data = data
        self.dt = dt

    def to_numpy(self) -> np.ndarray:
        return self._data

    @property
    def times(self) -> np.ndarray:
        return np.arange(self._data.shape[1]) * self.dt


class _FakeResult:
    """Minimal ResultProtocol-like object for aggregate plotting."""

    def __init__(self, data, metadata):
        self._data = data
        self._metadata = metadata

    @property
    def data(self):
        return self._data

    @property
    def metadata(self):
        return self._metadata


def _trajectory(n_traj: int = 4, n_steps: int = 64, n_modes: int = 2):
    """Synthetic complex trajectories with a decaying oscillation."""
    t = np.arange(n_steps) * 0.1
    rng = np.random.default_rng(42)
    data = np.empty((n_traj, n_steps, n_modes), dtype=np.complex128)
    for mode in range(n_modes):
        envelope = np.exp(-0.1 * t) * np.exp(1j * (mode + 1) * t)
        data[:, :, mode] = envelope[None, :] * (
            1.0 + 0.05 * rng.standard_normal((n_traj, n_steps))
        )
    return _FakeTrajectory(data)


def _assert_files_generated(files: list[Path], output_dir: Path) -> None:
    assert files, "plotter returned no files"
    for path in files:
        assert path.parent == output_dir
        assert path.exists()
        assert path.stat().st_size > 0


def test_time_series_plotter_renders(tmp_path):
    plotter = TimeSeriesPlotter(
        plots=[{"channels": [0, 1], "transform": "real", "trajectories": "mean"}]
    )
    files = plotter.plot(_trajectory(), tmp_path, "png")
    _assert_files_generated(files, tmp_path)


def test_phase_plane_plotter_renders_hist2d(tmp_path):
    plotter = PhasePlanePlotter(
        plots=[{"channel_x": 0, "mode": "hist2d", "bins": 8}]
    )
    files = plotter.plot(_trajectory(), tmp_path, "png")
    _assert_files_generated(files, tmp_path)


def test_power_spectrum_plotter_from_precomputed_psd(tmp_path):
    """Pre-computed PSD dicts render without touching qphase_sde."""
    axis = np.linspace(-1.0, 1.0, 64)
    psd = np.exp(-((axis - 0.3) ** 2) / 0.01)[:, None]  # (64, 1)
    data = {"axis": axis, "psd": psd, "modes": [0]}

    plotter = PowerSpectrumPlotter(plots=[{"channels": [0], "scale": "log"}])
    files = plotter.plot(data, tmp_path, "png")
    _assert_files_generated(files, tmp_path)


def test_parameter_evolution_plotter_from_precomputed_peaks(tmp_path):
    """Aggregate dicts with pre-computed PSD peaks render without qphase_sde."""
    data = {
        f"sim[omega_a={value}]": _FakeResult(
            data={"peaks": {0: {"values": [1.0], "frequencies": [value]}}},
            metadata={"params": {"omega_a": value}},
        )
        for value in (0.1, 0.2, 0.3)
    }

    plotter = ParameterEvolutionPlotter(
        plots=[{"parameter": "omega_a", "metric": "psd_peak_freq", "channel": 0}]
    )
    files = plotter.plot(data, tmp_path, "png")
    _assert_files_generated(files, tmp_path)


def test_parameter_evolution_plotter_ignores_non_aggregate(tmp_path):
    plotter = ParameterEvolutionPlotter(
        plots=[{"parameter": "omega_a", "metric": "psd_peak_freq", "channel": 0}]
    )
    assert plotter.plot(np.zeros((2, 4, 1)), tmp_path, "png") == []
