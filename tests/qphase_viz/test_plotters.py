"""Smoke tests for typed qphase_viz plotters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import qphase_viz.plotters.evolution as evolution_module
import qphase_viz.plotters.parameter as parameter_module
import qphase_viz.plotters.phase as phase_module
import qphase_viz.plotters.spectrum as spectrum_module
from qphase_viz.plotters.evolution import TimeSeriesPlotter
from qphase_viz.plotters.parameter import ParameterEvolutionPlotter
from qphase_viz.plotters.phase import PhasePlanePlotter
from qphase_viz.plotters.spectrum import PowerSpectrumPlotter
from qphase_viz.typed_input import ProductNotFoundError

from tests.qphase_viz.typed_fixtures import (
    VizTestBundle,
    peak_scan_bundle,
    restored,
    role_bundle,
    spectrum_bundle,
    spectrum_dataset,
    spectrum_scan_bundle,
    swapped_quantity_spectrum_bundle,
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


#: One case per single-axes plotter, for the constrained-layout forwarding check.
_LAYOUT_CASES: list[tuple[Any, Any, dict[str, Any], Any]] = [
    (
        evolution_module,
        TimeSeriesPlotter,
        {"channels": [0], "transform": "abs"},
        trajectory_bundle,
    ),
    (
        phase_module,
        PhasePlanePlotter,
        {"channel_x": 0, "mode": "hist2d", "bins": 8},
        trajectory_bundle,
    ),
    (
        spectrum_module,
        PowerSpectrumPlotter,
        {"channels": [0], "scale": "log"},
        spectrum_bundle,
    ),
    (
        parameter_module,
        ParameterEvolutionPlotter,
        {"parameter": "omega", "metric": "psd_peak_freq", "channel": 0},
        peak_scan_bundle,
    ),
]
_LAYOUT_IDS = ["time_series", "phase_plane", "power_spectrum", "parameter_evolution"]


def _captured_constrained_layout(
    monkeypatch, tmp_path: Path, module, plotter, bundle
) -> bool:
    """Render through the public ``plot`` and report the figure's layout flag."""
    captured: dict[str, bool] = {}

    def fake_save(fig, output_dir, filename, format):
        captured["constrained_layout"] = bool(fig.get_constrained_layout())
        return tmp_path / "stub.png"

    monkeypatch.setattr(module, "save_figure", fake_save)
    plotter.plot(bundle, tmp_path, "png")
    return captured["constrained_layout"]


@pytest.mark.parametrize(
    ("module", "plotter_type", "spec", "bundle_factory"),
    _LAYOUT_CASES,
    ids=_LAYOUT_IDS,
)
def test_constrained_layout_reaches_the_figure(
    tmp_path, monkeypatch, module, plotter_type, spec, bundle_factory
):
    plotter = plotter_type(plots=[{**spec, "constrained_layout": True}])

    assert _captured_constrained_layout(
        monkeypatch, tmp_path, module, plotter, bundle_factory()
    )


@pytest.mark.parametrize(
    ("module", "plotter_type", "spec", "bundle_factory"),
    _LAYOUT_CASES,
    ids=_LAYOUT_IDS,
)
def test_constrained_layout_stays_off_by_default(
    tmp_path, monkeypatch, module, plotter_type, spec, bundle_factory
):
    """The flag is forwarded, not hard-coded on."""
    plotter = plotter_type(plots=[{**spec}])

    assert not _captured_constrained_layout(
        monkeypatch, tmp_path, module, plotter, bundle_factory()
    )


def test_power_spectrum_rejects_psd_quantity_on_a_sibling_variable(tmp_path):
    """``axis`` declaring the PSD quantity must not stand in for ``psd``."""
    plotter = PowerSpectrumPlotter(plots=[{"channels": [0], "scale": "log"}])

    with pytest.raises(ProductNotFoundError, match="needs a persisted PSD product"):
        plotter.plot(swapped_quantity_spectrum_bundle(), tmp_path, "png")


def test_power_spectrum_rejects_a_role_product_with_the_quantity_elsewhere(tmp_path):
    plotter = PowerSpectrumPlotter(plots=[{"channels": [0], "scale": "log"}])
    bundle = role_bundle(
        {"psd": swapped_quantity_spectrum_bundle().products["psd"]},
        {"primary_spectrum": "psd"},
    )

    with pytest.raises(TypeError, match="does not satisfy plot requirements"):
        plotter.plot(bundle, tmp_path, "png")


def test_power_spectrum_accepts_the_correct_product_through_a_role(tmp_path):
    bundle = role_bundle({"psd": spectrum_dataset()}, {"primary_spectrum": "psd"})

    files = PowerSpectrumPlotter(plots=[{"channels": [0], "scale": "log"}]).plot(
        bundle, tmp_path, "png"
    )

    _assert_files_generated(files, tmp_path)
