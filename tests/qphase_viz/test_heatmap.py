"""Tests for the scalar heatmap plotter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import qphase_viz.plotters.heatmap as heatmap_module
from matplotlib.collections import QuadMesh
from matplotlib.colors import LogNorm, Normalize
from qphase.data import StatisticsDataset
from qphase_viz.plotters.heatmap import HeatmapPlotter

from tests.qphase_viz.typed_fixtures import (
    HEATMAP_X,
    HEATMAP_Y,
    VizTestBundle,
    heatmap_bundle,
    restored,
)

BASE_SPEC: dict[str, Any] = {
    "x_variable": "x",
    "y_variable": "y",
    "value_variable": "value",
}


def _plotter(**spec) -> HeatmapPlotter:
    merged = {**BASE_SPEC, **spec}
    return HeatmapPlotter(plots=[merged])


def _capture(monkeypatch, tmp_path: Path) -> dict[str, Any]:
    """Capture the open figure instead of closing it, then save a stub file."""
    captured: dict[str, Any] = {}

    def fake_save(fig, output_dir, filename, format):
        path = tmp_path / f"{filename}.{format}"
        captured["figure"] = fig
        captured["axes"] = fig.axes
        captured["mesh"] = fig.axes[0].collections[0]
        fig.savefig(path, format=format)
        return path

    monkeypatch.setattr(heatmap_module, "save_figure", fake_save)
    return captured


def _with_values(values: np.ndarray, **extra) -> VizTestBundle:
    """Rebuild the heatmap product with a replaced ``value`` array."""
    product = heatmap_bundle().products["grid"]
    arrays = {
        variable.name: np.asarray(product.handle(variable.name).materialize())
        for variable in product.variables
    }
    arrays.update(extra)
    arrays["value"] = values
    arrays["valid"] = np.asarray(arrays["valid"], dtype=bool)
    return VizTestBundle(
        {
            "grid": StatisticsDataset.from_arrays(
                product.schema, arrays, owner="test.viz"
            )
        }
    )


def test_renders_one_file(tmp_path):
    files = _plotter().plot(heatmap_bundle(), tmp_path, "png")

    assert len(files) == 1
    assert files[0].suffix == ".png"
    assert files[0].stat().st_size > 0


def test_value_grid_is_row_x_column_y(tmp_path, monkeypatch):
    """``value[row, column]`` maps onto ``(x[row], y[column])``, not the reverse."""
    captured = _capture(monkeypatch, tmp_path)
    grid = heatmap_bundle().products["grid"]

    _plotter().plot(heatmap_bundle(), tmp_path, "png")

    mesh = captured["mesh"]
    assert isinstance(mesh, QuadMesh)
    array = np.ma.asarray(mesh.get_array())
    # pcolormesh stores C in (row=y, column=x) order.
    assert array.reshape(len(HEATMAP_Y), len(HEATMAP_X)).shape == (
        len(HEATMAP_Y),
        len(HEATMAP_X),
    )
    expected = np.asarray(grid.handle("value").materialize()).T
    stacked = np.ma.asarray(array).reshape(len(HEATMAP_Y), len(HEATMAP_X))
    np.testing.assert_allclose(stacked.data, expected, rtol=0, atol=0)


def test_validity_mask_blanks_disallowed_cells(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, tmp_path)
    grid = heatmap_bundle().products["grid"]
    valid = np.asarray(grid.handle("valid").materialize())

    _plotter(valid_variable="valid").plot(heatmap_bundle(), tmp_path, "png")

    mask = np.ma.getmaskarray(np.ma.asarray(captured["mesh"].get_array()))
    np.testing.assert_array_equal(
        mask.reshape(valid.shape[::-1]).T, ~valid
    )


def test_nan_values_are_blanked_not_zero(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, tmp_path)
    values = np.asarray(
        heatmap_bundle().products["grid"].handle("value").materialize()
    ).astype(float)
    values[0, 0] = np.nan
    bundle = _with_values(values, valid=np.ones(values.shape, dtype=bool))

    _plotter(valid_variable="valid").plot(bundle, tmp_path, "png")

    mask = np.ma.getmaskarray(np.ma.asarray(captured["mesh"].get_array()))
    assert mask.reshape(values.shape)[0, 0]
    assert not mask.reshape(values.shape)[1, 1]


def test_all_nan_grid_is_rejected(tmp_path):
    values = np.full(
        np.asarray(
            heatmap_bundle().products["grid"].handle("value").materialize()
        ).shape,
        np.nan,
    )
    bundle = _with_values(values, valid=np.ones(values.shape, dtype=bool))

    with pytest.raises(ValueError, match="no finite valid values"):
        _plotter(valid_variable="valid").plot(bundle, tmp_path, "png")


def test_linear_norm_and_explicit_limits(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, tmp_path)

    _plotter(vmin=0.0, vmax=100.0).plot(heatmap_bundle(), tmp_path, "png")

    norm = captured["mesh"].norm
    assert isinstance(norm, Normalize)
    assert (norm.vmin, norm.vmax) == (0.0, 100.0)


def test_log_norm_uses_the_positive_range(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, tmp_path)

    _plotter(norm="log").plot(heatmap_bundle(), tmp_path, "png")

    norm = captured["mesh"].norm
    assert isinstance(norm, LogNorm)
    assert norm.vmin > 0.0


def test_log_norm_rejects_non_positive_data(tmp_path):
    values = np.asarray(
        heatmap_bundle().products["grid"].handle("value").materialize()
    ).astype(float)
    values[1, 1] = 0.0
    bundle = _with_values(values, valid=np.ones(values.shape, dtype=bool))

    with pytest.raises(ValueError, match="non-positive values"):
        _plotter(norm="log", valid_variable="valid").plot(bundle, tmp_path, "png")


def test_unknown_colormap_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown colormap"):
        _plotter(cmap="definitely-not-a-map").plot(heatmap_bundle(), tmp_path, "png")


def test_a_grid_that_does_not_match_its_axes_cannot_be_constructed():
    """The typed schema already refuses a grid whose shape contradicts the axes.

    The plotter repeats the check as a defensive stance for products assembled
    outside the schema, so this contract belongs to ``qphase.data``.
    """
    values = np.zeros((len(HEATMAP_X) + 1, len(HEATMAP_Y)))

    with pytest.raises(ValueError, match="does not match closed axis size"):
        _with_values(values)


def test_non_monotonic_coordinate_is_rejected(tmp_path):
    product = heatmap_bundle().products["grid"]
    arrays = {
        variable.name: np.asarray(product.handle(variable.name).materialize())
        for variable in product.variables
    }
    arrays["x"] = np.asarray([0.0, 0.2, 0.1, 0.6, 0.8, 1.0])
    rebuilt = StatisticsDataset.from_arrays(
        product.schema, arrays, owner="test.viz"
    )
    bundle = VizTestBundle({"grid": rebuilt})

    with pytest.raises(ValueError, match="not monotonic"):
        _plotter().plot(bundle, tmp_path, "png")


def test_ambiguous_product_selection_is_rejected(tmp_path):
    products = heatmap_bundle().products
    bundle = VizTestBundle({**products, "copy": products["grid"]})

    with pytest.raises(ValueError, match="ambiguous"):
        _plotter().plot(bundle, tmp_path, "png")


def test_named_product_disambiguates(tmp_path):
    products = heatmap_bundle().products
    bundle = VizTestBundle({**products, "copy": products["grid"]})

    files = _plotter(product="copy").plot(bundle, tmp_path, "png")

    assert files[0].is_file()


def test_mask_contour_and_explicit_point_series(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, tmp_path)

    _plotter(
        valid_variable="valid",
        contours=[{"levels": [0.5]}],
        series=[
            {
                "x_variable": "boundary_x",
                "y_variable": "boundary_y",
                "linestyle": "--",
                "label": "boundary",
            },
            {
                "x": [0.0],
                "y": [10.0],
                "marker": "*",
                "linestyle": None,
                "markersize": 12.0,
            },
        ],
    ).plot(heatmap_bundle(), tmp_path, "png")

    ax = captured["axes"][0]
    # One mesh, one contour set.
    assert len(ax.collections) == 2
    assert len(ax.lines) == 2
    marker_line = ax.lines[1]
    assert marker_line.get_marker() == "*"
    assert marker_line.get_linestyle() in {"None", "none", ""}
    assert marker_line.get_markersize() == pytest.approx(12.0)


def test_variable_contour_uses_the_named_field(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, tmp_path)

    _plotter(contours=[{"variable": "height", "levels": [12.0]}]).plot(
        heatmap_bundle(), tmp_path, "png"
    )

    assert len(captured["axes"][0].collections) == 2


def test_contouring_a_missing_mask_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="needs 'valid_variable'"):
        _plotter(contours=[{"levels": [0.5]}]).plot(heatmap_bundle(), tmp_path, "png")


def test_series_needs_exactly_one_source_per_axis():
    from qphase_viz.config import HeatmapSeries

    with pytest.raises(ValueError, match="exactly one x source"):
        HeatmapSeries(x=[1.0], y=[1.0], x_variable="x")
    with pytest.raises(ValueError, match="equal x and y lengths"):
        HeatmapSeries(x=[1.0], y=[1.0, 2.0])


def test_restored_artifact_heatmap_renders(tmp_path):
    bundle = restored(heatmap_bundle(), tmp_path / "artifact")

    files = _plotter().plot(bundle, tmp_path / "plots", "png")

    assert files[0].is_file()
    assert files[0].stat().st_size > 0


def test_title_labels_and_annotation_are_applied(tmp_path):
    files = _plotter(
        title="Grid title",
        xlabel="X axis",
        ylabel="Y axis",
        colorbar_label="Scalar",
        annotations=[{"text": "corner", "x": 0.05, "y": 0.95}],
    ).plot(heatmap_bundle(), tmp_path, "svg")

    svg = files[0].read_text(encoding="utf-8")
    assert "Grid title" in svg
    assert "X axis" in svg
    assert "Y axis" in svg
    assert "Scalar" in svg
    assert "corner" in svg
