"""Tests for the PSD-across-scan plotter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qphase.data import SpectralDataset
from qphase_viz.plotters.spectral_scan import SpectralScanPlotter
from qphase_viz.styles import RenderOptions

from tests.qphase_viz.typed_fixtures import (
    SCAN_POINTS,
    VizTestBundle,
    restored,
    spectral_scan_bundle,
)


def _plotter(**spec) -> SpectralScanPlotter:
    spec.setdefault("scan_parameter", "omega")
    return SpectralScanPlotter(plots=[spec])


def _svg(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_renders_one_file_with_every_scan_point_by_default(tmp_path):
    files = _plotter().plot(spectral_scan_bundle(), tmp_path, "svg")

    assert len(files) == 1
    svg = _svg(files[0])
    for value in SCAN_POINTS:
        assert f"omega={value:.6g}" in svg


def test_scan_indices_select_points_in_order(tmp_path):
    files = _plotter(scan_indices=[4, 0]).plot(spectral_scan_bundle(), tmp_path, "svg")

    svg = _svg(files[0])
    assert "omega=0.5" in svg
    assert "omega=0.1" in svg
    assert "omega=0.3" not in svg


def test_scan_coordinates_use_nearest_point(tmp_path):
    files = _plotter(scan_coordinates=[0.19, 0.41], scan_tolerance=0.02).plot(
        spectral_scan_bundle(), tmp_path, "svg"
    )

    svg = _svg(files[0])
    assert "omega=0.2" in svg
    assert "omega=0.4" in svg


def test_scan_coordinate_outside_tolerance_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="away from the nearest point"):
        _plotter(scan_coordinates=[0.13], scan_tolerance=0.01).plot(
            spectral_scan_bundle(), tmp_path, "svg"
        )


def test_explicit_tolerance_widens_the_nearest_match(tmp_path):
    files = _plotter(scan_coordinates=[0.32], scan_tolerance=0.2).plot(
        spectral_scan_bundle(), tmp_path, "svg"
    )

    assert "omega=0.3" in _svg(files[0])


def test_index_and_coordinate_selection_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        _plotter(
            scan_indices=[0], scan_coordinates=[0.1], scan_tolerance=0.01
        ).plot(spectral_scan_bundle(), tmp_path, "svg")


def test_scan_index_out_of_range_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="out of range"):
        _plotter(scan_indices=[9]).plot(spectral_scan_bundle(), tmp_path, "svg")


def test_unknown_mode_is_rejected_with_the_available_modes(tmp_path):
    with pytest.raises(ValueError, match=r"not in the stored modes \[0, 1\]"):
        _plotter(channels=[0, 5]).plot(spectral_scan_bundle(), tmp_path, "svg")


def test_point_view_psd_is_rejected_with_a_pointer_to_the_single_point_plotter(
    tmp_path,
):
    bundle = spectral_scan_bundle().point_view((0,))

    with pytest.raises(ValueError, match="power_spectrum plotter"):
        _plotter().plot(bundle, tmp_path, "svg")


def _with_axis(transform) -> VizTestBundle:
    """Rebuild the PSD product with its frequency axis transformed."""
    product = spectral_scan_bundle().products["psd"]
    arrays = {
        variable.name: np.asarray(product.handle(variable.name).materialize())
        for variable in product.variables
    }
    arrays["axis"] = transform(arrays["axis"])
    return VizTestBundle(
        {"psd": SpectralDataset.from_arrays(product.schema, arrays, owner="test.viz")}
    )


def test_inconsistent_frequency_axes_across_scan_are_rejected(tmp_path):
    scale = np.linspace(1.0, 2.0, len(SCAN_POINTS))[:, None]
    bundle = _with_axis(lambda axis: axis * scale)

    with pytest.raises(ValueError, match="different frequency axes"):
        _plotter().plot(bundle, tmp_path, "svg")


def test_unsorted_frequency_axis_is_rejected(tmp_path):
    bundle = _with_axis(lambda axis: axis[:, ::-1].copy())

    with pytest.raises(ValueError, match="strictly increasing"):
        _plotter().plot(bundle, tmp_path, "svg")


def test_log_floor_halfway_to_the_smallest_positive_value():
    psd = np.asarray([[[0.0, 2.0], [4.0, -1.0]]])

    assert SpectralScanPlotter._log_floor(psd) == pytest.approx(1.0)


def test_all_non_positive_psd_cannot_use_a_log_scale(tmp_path):
    product = spectral_scan_bundle().products["psd"]
    arrays = {
        variable.name: np.asarray(product.handle(variable.name).materialize())
        for variable in product.variables
    }
    arrays["psd"] = np.zeros_like(arrays["psd"])
    bundle = VizTestBundle(
        {"psd": SpectralDataset.from_arrays(product.schema, arrays, owner="test.viz")}
    )

    with pytest.raises(ValueError, match="no positive finite values"):
        _plotter().plot(bundle, tmp_path, "svg")


def test_linear_scale_skips_the_floor(tmp_path):
    files = _plotter(log_scale=False, scan_indices=[0]).plot(
        spectral_scan_bundle(), tmp_path, "svg"
    )

    assert "omega=0.1" in _svg(files[0])


def test_title_label_templates_and_annotation_land_on_the_figure(tmp_path):
    plotter = SpectralScanPlotter(
        plots=[
            {
                "scan_parameter": "omega",
                "scan_indices": [0, 2],
                "channels": [0, 1],
                "title": "Scan title",
                "xlabel": "Frequency axis",
                "label_template": "pt {index} ({value:.2f})",
                "mode_label_template": "Channel {mode}",
                "annotations": [{"text": "panel-note", "x": 0.05, "y": 0.9}],
            }
        ]
    )

    files = plotter.plot(spectral_scan_bundle(), tmp_path, "svg")

    svg = _svg(files[0])
    assert "Scan title" in svg
    assert "Frequency axis" in svg
    assert "pt 0 (0.10)" in svg
    assert "Channel 0" in svg
    assert "Channel 1" in svg
    # The annotation rule is "first panel only", so it appears exactly once.
    assert svg.count("panel-note") == 1


def test_restored_artifact_scan_renders(tmp_path):
    bundle = restored(spectral_scan_bundle(), tmp_path / "artifact")

    files = _plotter(scan_indices=[0, 1]).plot(bundle, tmp_path / "plots", "svg")

    assert files[0].is_file()
    assert "omega=0.2" in _svg(files[0])


def test_style_options_reach_the_scan_figure(tmp_path):
    files = _plotter(scan_indices=[0], filename="styled").plot(
        spectral_scan_bundle(), tmp_path, "svg", RenderOptions(style="sm")
    )

    assert "font-family: 'STIXGeneral'" in _svg(files[0])
    assert "font-size: 13.5px" in _svg(files[0])


def test_unknown_colormap_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown colormap"):
        _plotter(cmap="not-a-map").plot(spectral_scan_bundle(), tmp_path, "svg")


def _line_colours(monkeypatch, tmp_path: Path, **spec) -> list[Any]:
    """Render one figure and return the colours of the plotted curves."""
    import qphase_viz.plotters.spectral_scan as spectral_scan_module

    captured: dict[str, list[Any]] = {}

    def fake_save(fig, output_dir, filename, format):
        captured["colours"] = [
            tuple(float(channel) for channel in np.asarray(line.get_color()))
            for line in fig.axes[0].lines
        ]
        return tmp_path / "stub.png"

    monkeypatch.setattr(spectral_scan_module, "save_figure", fake_save)
    _plotter(**spec).plot(spectral_scan_bundle(), tmp_path, "png")
    return captured["colours"]


def test_scan_point_colour_comes_from_its_position_in_the_whole_scan(
    tmp_path, monkeypatch
):
    """Selecting a subset must not repaint the remaining curves."""
    full = _line_colours(monkeypatch, tmp_path, scan_indices=list(range(5)))
    subset = _line_colours(monkeypatch, tmp_path, scan_indices=[2, 4])

    assert len(full) == len(SCAN_POINTS)
    assert subset == [full[2], full[4]]
    assert subset[0] != full[0]


def test_coordinate_selection_requires_an_explicit_tolerance(tmp_path):
    with pytest.raises(ValueError, match="requires an explicit scan_tolerance"):
        _plotter(scan_coordinates=[0.19]).plot(
            spectral_scan_bundle(), tmp_path, "svg"
        )


def test_tolerance_must_be_positive():
    with pytest.raises(ValueError):
        SpectralScanPlotter(
            plots=[
                {
                    "scan_parameter": "omega",
                    "scan_coordinates": [0.19],
                    "scan_tolerance": 0.0,
                }
            ]
        )
