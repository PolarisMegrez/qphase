"""Smoke tests for typed qphase_viz plotters."""

from __future__ import annotations

from pathlib import Path

import pytest
from qphase_viz.plotters.evolution import TimeSeriesPlotter
from qphase_viz.plotters.parameter import ParameterEvolutionPlotter
from qphase_viz.plotters.phase import PhasePlanePlotter
from qphase_viz.plotters.spectrum import PowerSpectrumPlotter
from qphase_viz.typed_input import ProductNotFoundError

from tests.qphase_viz.typed_fixtures import (
    VizTestBundle,
    peak_scan_bundle,
    restored,
    spectrum_bundle,
    spectrum_scan_bundle,
    trajectory_bundle,
)


def _assert_files_generated(files: list[Path], output_dir: Path) -> None:
    assert files, "plotter returned no files"
    for path in files:
        assert path.parent == output_dir
        assert path.exists()
        assert path.stat().st_size > 0


def test_time_series_plotter_renders_typed_bundle(tmp_path):
    plotter = TimeSeriesPlotter(
        plots=[{"channels": [0, 1], "transform": "real", "trajectories": "mean"}]
    )
    files = plotter.plot(trajectory_bundle(), tmp_path, "png")
    _assert_files_generated(files, tmp_path)


def test_phase_plane_plotter_renders_typed_bundle(tmp_path):
    plotter = PhasePlanePlotter(
        plots=[{"channel_x": 0, "mode": "hist2d", "bins": 8}]
    )
    files = plotter.plot(trajectory_bundle(), tmp_path, "png")
    _assert_files_generated(files, tmp_path)


def test_power_spectrum_plots_typed_psd(tmp_path):
    source = spectrum_bundle()
    # The PSD product alone must be enough: no trajectories are required.
    bundle = VizTestBundle({"psd": source.products["psd"]})
    files = PowerSpectrumPlotter(
        plots=[{"channels": [0], "scale": "log"}]
    ).plot(bundle, tmp_path, "png")
    _assert_files_generated(files, tmp_path)


def test_power_spectrum_requires_typed_psd(tmp_path):
    assert "psd" not in trajectory_bundle().products
    plotter = PowerSpectrumPlotter(plots=[{"channels": [0], "scale": "log"}])
    with pytest.raises(
        ProductNotFoundError, match="needs a persisted PSD product.*psd analyser"
    ):
        plotter.plot(trajectory_bundle(), tmp_path, "png")


def test_power_spectrum_rejects_ambiguous_typed_psd(tmp_path):
    source = spectrum_bundle()
    products = source.products
    ambiguous = VizTestBundle({**products, "other_psd": products["psd"]})
    plotter = PowerSpectrumPlotter(plots=[{"channels": [0], "scale": "log"}])
    with pytest.raises(ValueError, match="exactly one match"):
        plotter.plot(ambiguous, tmp_path, "png")


def test_parameter_evolution_uses_typed_peak_product(tmp_path):
    files = ParameterEvolutionPlotter(
        plots=[{"parameter": "omega", "metric": "psd_peak_freq", "channel": 0}]
    ).plot(peak_scan_bundle(), tmp_path, "png")
    _assert_files_generated(files, tmp_path)


def test_time_phase_and_spectrum_render_restored_bundle(tmp_path):
    bundle = restored(spectrum_bundle(), tmp_path / "artifact")
    files = TimeSeriesPlotter(
        plots=[{"channels": [0], "transform": "abs"}]
    ).plot(bundle, tmp_path, "png")
    files += PhasePlanePlotter(
        plots=[{"channel_x": 0, "mode": "hist2d", "bins": 8}]
    ).plot(bundle, tmp_path, "png")
    files += PowerSpectrumPlotter(
        plots=[{"channels": [0], "scale": "log"}]
    ).plot(bundle, tmp_path, "png")
    _assert_files_generated(files, tmp_path)


def test_parameter_evolution_renders_restored_bundle(tmp_path):
    bundle = restored(peak_scan_bundle(), tmp_path / "artifact")
    files = ParameterEvolutionPlotter(
        plots=[{"parameter": "omega", "metric": "psd_peak_val", "channel": 0}]
    ).plot(bundle, tmp_path, "png")
    _assert_files_generated(files, tmp_path)


@pytest.mark.parametrize(
    ("plotter", "bundle_factory"),
    [
        (
            TimeSeriesPlotter(plots=[{"channels": [0]}]),
            lambda: trajectory_bundle(scan_values=(0.1, 0.2, 0.3)),
        ),
        (
            PhasePlanePlotter(plots=[{"channel_x": 0}]),
            lambda: trajectory_bundle(scan_values=(0.1, 0.2, 0.3)),
        ),
        (
            PowerSpectrumPlotter(plots=[{"channels": [0]}]),
            spectrum_scan_bundle,
        ),
    ],
)
@pytest.mark.parametrize("restore_first", [False, True])
def test_non_scan_plotters_reject_multi_point_scan(
    tmp_path, plotter, bundle_factory, restore_first
):
    bundle = bundle_factory()
    if restore_first:
        bundle = restored(bundle, tmp_path / "artifact")
    with pytest.raises(ValueError, match="input.mode: map.*select.*point_view"):
        plotter.plot(bundle, tmp_path, "png")


@pytest.mark.parametrize(
    ("plotter", "bundle_factory"),
    [
        (
            TimeSeriesPlotter(plots=[{"channels": [0]}]),
            lambda: trajectory_bundle(scan_values=(0.1,)),
        ),
        (
            PhasePlanePlotter(plots=[{"channel_x": 0}]),
            lambda: trajectory_bundle(scan_values=(0.1,)),
        ),
        (
            PowerSpectrumPlotter(plots=[{"channels": [0]}]),
            spectrum_bundle,
        ),
    ],
)
def test_non_scan_plotters_accept_singleton_scan(tmp_path, plotter, bundle_factory):
    files = plotter.plot(bundle_factory(), tmp_path, "png")
    _assert_files_generated(files, tmp_path)


@pytest.mark.parametrize(
    ("plotter", "bundle_factory"),
    [
        (
            TimeSeriesPlotter(plots=[{"channels": [0]}]),
            lambda: trajectory_bundle(scan_values=(0.1, 0.2, 0.3)),
        ),
        (
            PhasePlanePlotter(plots=[{"channel_x": 0}]),
            lambda: trajectory_bundle(scan_values=(0.1, 0.2, 0.3)),
        ),
        (
            PowerSpectrumPlotter(plots=[{"channels": [0]}]),
            spectrum_scan_bundle,
        ),
    ],
)
def test_non_scan_plotters_accept_point_view(tmp_path, plotter, bundle_factory):
    bundle = bundle_factory().point_view((1,))
    files = plotter.plot(bundle, tmp_path, "png")
    _assert_files_generated(files, tmp_path)
