"""Tests for the sample-distribution plotter and its FWHM helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import qphase_viz.plotters.distribution as distribution_module
from qphase.data import StatisticsDataset
from qphase_viz.config import DistributionSpec
from qphase_viz.plotters.distribution import DistributionPlotter, fwhm_bounds
from qphase_viz.styles import PALETTE

from tests.qphase_viz.typed_fixtures import (
    SCAN_POINTS,
    VizTestBundle,
    distribution_bundle,
    distribution_dataset,
    restored,
)

BASE_SPEC: dict[str, Any] = {
    "value_variable": "estimate",
    "scan_parameter": "omega",
    "bin_width": 0.002,
}

# --- fwhm_bounds -----------------------------------------------------------


def test_fwhm_bounds_on_a_triangular_density():
    x = np.arange(5, dtype=float)  # 0,1,2,3,4
    density = np.asarray([0.0, 1.0, 2.0, 1.0, 0.0])

    left, right = fwhm_bounds(x, density)

    # Half maximum is 1.0, reached exactly at the neighbouring bins.
    assert left == pytest.approx(1.0)
    assert right == pytest.approx(3.0)


def test_fwhm_bounds_interpolates_between_bins():
    x = np.arange(4, dtype=float)
    density = np.asarray([0.0, 1.0, 3.0, 0.0])

    left, right = fwhm_bounds(x, density)

    # Half maximum 1.5 lies between bins 1 and 2 and between 2 and 3.
    assert left == pytest.approx(1.25)
    assert right == pytest.approx(2.5)


def test_fwhm_bounds_on_a_flat_plateau():
    x = np.arange(5, dtype=float)
    density = np.asarray([0.0, 2.0, 2.0, 2.0, 0.0])

    left, right = fwhm_bounds(x, density)

    assert left == pytest.approx(0.5)
    assert right == pytest.approx(3.5)


def test_fwhm_bounds_rejects_a_band_touching_the_start():
    x = np.arange(4, dtype=float)
    density = np.asarray([3.0, 2.0, 1.0, 0.0])

    with pytest.raises(ValueError, match="reaches the start of the range"):
        fwhm_bounds(x, density)


def test_fwhm_bounds_rejects_a_band_touching_the_end():
    x = np.arange(4, dtype=float)
    density = np.asarray([0.0, 1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="reaches the end of the range"):
        fwhm_bounds(x, density)


def test_fwhm_bounds_needs_a_fully_contained_peak():
    # A peak inside the range but with a shoulder above half maximum at the
    # boundary has no crossing either.
    x = np.arange(4, dtype=float)
    density = np.asarray([1.0, 3.0, 2.0, 2.0])

    with pytest.raises(ValueError, match="reaches the end of the range"):
        fwhm_bounds(x, density)


def test_fwhm_bounds_rejects_degenerate_input():
    with pytest.raises(ValueError, match="matching lengths"):
        fwhm_bounds(np.arange(3.0), np.ones(4))
    with pytest.raises(ValueError, match="at least two bins"):
        fwhm_bounds(np.arange(1.0), np.ones(1))
    with pytest.raises(ValueError, match="positive maximum"):
        fwhm_bounds(np.arange(3.0), np.zeros(3))
    with pytest.raises(ValueError, match="one-dimensional"):
        fwhm_bounds(np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="finite bin positions"):
        fwhm_bounds(np.asarray([0.0, np.nan, 2.0]), np.asarray([0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="finite densities"):
        fwhm_bounds(np.arange(3.0), np.asarray([0.0, np.inf, 0.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        fwhm_bounds(np.asarray([0.0, 1.0, 1.0]), np.asarray([0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="non-negative densities"):
        fwhm_bounds(np.arange(3.0), np.asarray([-1.0, 1.0, 0.0]))


# --- configuration ---------------------------------------------------------


def test_bin_width_and_bins_are_mutually_exclusive():
    with pytest.raises(ValueError, match="exactly one of"):
        DistributionSpec(value_variable="v")
    with pytest.raises(ValueError, match="exactly one of"):
        DistributionSpec(value_variable="v", bin_width=0.1, bins=10)


def test_zero_scale_is_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        DistributionSpec(value_variable="v", bins=5, scale=0.0)


def test_annotation_labels_require_their_switch():
    with pytest.raises(ValueError, match="show_means"):
        DistributionSpec(value_variable="v", bins=5, shift_label="shift")
    with pytest.raises(ValueError, match="show_fwhm"):
        DistributionSpec(value_variable="v", bins=5, width_label="width")


def test_tick_labels_need_ticks():
    with pytest.raises(ValueError, match="needs xticks"):
        DistributionSpec(value_variable="v", bins=5, xticklabels=["a", "b"])
    with pytest.raises(ValueError, match="same length"):
        DistributionSpec(
            value_variable="v", bins=5, xticks=[0.0, 1.0], xticklabels=["a"]
        )


# --- rendering helpers -----------------------------------------------------


def _plotter(**spec) -> DistributionPlotter:
    return DistributionPlotter(plots=[{**BASE_SPEC, **spec}])


def _capture(monkeypatch, tmp_path: Path) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_save(fig, output_dir, filename, format):
        ax = fig.axes[0]
        captured["axes"] = ax
        captured["lines"] = ax.lines
        captured["collections"] = ax.collections
        captured["texts"] = [text.get_text() for text in ax.texts]
        captured["text_sizes"] = {
            text.get_text(): float(text.get_fontsize()) for text in ax.texts
        }
        captured["figure"] = fig
        return tmp_path / "stub.png"

    monkeypatch.setattr(distribution_module, "save_figure", fake_save)
    return captured


def _vertical_markers(ax) -> list[float]:
    """X positions of every vertical marker line drawn on *ax*."""
    positions: list[float] = []
    for collection in ax.collections:
        segments = getattr(collection, "get_segments", None)
        if segments is None:
            continue
        for segment in segments():
            (x0, _), (x1, _) = segment
            if abs(float(x1) - float(x0)) < 1e-12:
                positions.append(float(x0))
    return positions


def _line_y(ax, index: int) -> np.ndarray:
    return np.asarray(ax.lines[index].get_ydata(), dtype=float)


def _line_x(ax, index: int) -> np.ndarray:
    return np.asarray(ax.lines[index].get_xdata(), dtype=float)


# --- data semantics --------------------------------------------------------


def test_identity_transform_bins_the_raw_samples(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, tmp_path)
    dataset = distribution_dataset()
    raw = np.asarray(dataset.handle("estimate").materialize(), dtype=float)

    _plotter(scan_indices=[0], bins=20, bin_width=None).plot(
        distribution_bundle(), tmp_path, "png"
    )

    # Density integrates to one, so the mean of the drawn curve matches the
    # mean of the raw finite samples inside the same range.
    ax = captured["axes"]
    centres, density = _line_x(ax, 0), _line_y(ax, 0)
    finite = raw[0][np.isfinite(raw[0])]
    keep = (centres >= finite.min()) & (centres <= finite.max())
    integral = np.sum(density[keep]) * float(np.diff(centres).mean())
    assert integral == pytest.approx(1.0, rel=0.1)


def test_reference_mean_centring_with_negative_scale_reverses_the_series(
    tmp_path, monkeypatch
):
    captured = _capture(monkeypatch, tmp_path)
    dataset = distribution_dataset()
    raw = np.asarray(dataset.handle("estimate").materialize(), dtype=float)
    reference = float(np.nanmean(raw[0]))

    _plotter(
        scan_indices=[0, 2],
        bins=30,
        bin_width=None,
        center="reference_mean",
        scale=-1.0,
        show_means=True,
    ).plot(distribution_bundle(), tmp_path, "png")

    ax = captured["axes"]
    means = _vertical_markers(ax)
    observed = float(np.mean(raw[2][np.isfinite(raw[2])]))
    # samples = scale * value + reference_mean = reference - value here.
    assert means[0] == pytest.approx(0.0, abs=2e-3)
    assert means[1] == pytest.approx(reference - observed, abs=2e-3)
    # The perturbed series estimated a larger frequency, so the reversed
    # series sits below the reference.
    assert means[1] < means[0]


def test_reference_first_series_orientation_is_a_choice_of_order(
    tmp_path, monkeypatch
):
    """Listing the higher-frequency point first turns the same shift positive."""
    captured = _capture(monkeypatch, tmp_path)
    raw = np.asarray(
        distribution_dataset().handle("estimate").materialize(), dtype=float
    )

    _plotter(
        scan_indices=[2, 0],
        bins=30,
        bin_width=None,
        center="reference_mean",
        scale=-1.0,
        show_means=True,
    ).plot(distribution_bundle(), tmp_path, "png")

    means = _vertical_markers(captured["axes"])
    reference = float(np.mean(raw[2][np.isfinite(raw[2])]))
    observed = float(np.mean(raw[0][np.isfinite(raw[0])]))
    assert means[0] == pytest.approx(0.0, abs=2e-3)
    assert means[1] == pytest.approx(reference - observed, abs=2e-3)
    assert means[1] > means[0]


def test_positive_scale_keeps_the_series_direction(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, tmp_path)
    raw = np.asarray(
        distribution_dataset().handle("estimate").materialize(), dtype=float
    )

    _plotter(
        scan_indices=[0, 2],
        bins=30,
        bin_width=None,
        center="reference_mean",
        scale=1.0,
        show_means=True,
    ).plot(distribution_bundle(), tmp_path, "png")

    means = _vertical_markers(captured["axes"])
    reference = float(np.mean(raw[0][np.isfinite(raw[0])]))
    observed = float(np.mean(raw[2][np.isfinite(raw[2])]))
    # samples = value - reference here: the reference still lands on zero and
    # the ordering of the raw series survives.
    assert means[0] == pytest.approx(0.0, abs=2e-3)
    assert means[1] == pytest.approx(observed - reference, abs=2e-3)
    assert means[1] > means[0]


def test_scan_points_are_not_pooled(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, tmp_path)

    _plotter(scan_indices=[1, 3], bins=20, bin_width=None).plot(
        distribution_bundle(), tmp_path, "png"
    )

    ax = captured["axes"]
    # Two series, each normalised on its own sample count.
    assert len([line for line in ax.lines if line.get_label() != "_child2"]) >= 2


def test_scan_coordinates_need_an_explicit_tolerance(tmp_path):
    with pytest.raises(ValueError, match="explicit scan_tolerance"):
        _plotter(
            scan_coordinates=[0.1], bins=10, bin_width=None
        ).plot(distribution_bundle(), tmp_path, "png")


def test_scan_coordinates_select_the_nearest_point(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, tmp_path)
    raw = np.asarray(
        distribution_dataset().handle("estimate").materialize(), dtype=float
    )

    _plotter(
        scan_coordinates=[0.11, 0.51],
        scan_tolerance=0.02,
        bins=10,
        bin_width=None,
        show_means=True,
        labels=["at 0.1", "at 0.5"],
    ).plot(distribution_bundle(), tmp_path, "png")

    means = _vertical_markers(captured["axes"])
    assert len(means) == 2
    # 0.11 -> point 0 (0.1) and 0.51 -> point 4 (0.5).
    assert means[0] == pytest.approx(float(np.nanmean(raw[0])), abs=1e-6)
    assert means[1] == pytest.approx(float(np.nanmean(raw[4])), abs=1e-6)


def test_out_of_tolerance_coordinate_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="away from the nearest point"):
        _plotter(
            scan_coordinates=[0.14],
            scan_tolerance=0.01,
            bins=10,
            bin_width=None,
        ).plot(distribution_bundle(), tmp_path, "png")


def test_missing_scan_parameter_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="set 'scan_parameter'"):
        _plotter(scan_parameter=None, bins=10, bin_width=None).plot(
            distribution_bundle(), tmp_path, "png"
        )


def test_histogram_density_formula(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, tmp_path)

    _plotter(scan_indices=[1], bin_width=0.005, xlim=(-0.05, 0.05)).plot(
        distribution_bundle(), tmp_path, "png"
    )

    ax = captured["axes"]
    centres, density = _line_x(ax, 0), _line_y(ax, 0)
    width = float(np.diff(centres).mean())
    assert width == pytest.approx(0.005)
    # A density integrates to 1 over the bins that contain samples.
    assert np.sum(density) * width == pytest.approx(1.0, rel=1e-9)


def test_bins_count_is_used_when_bin_width_is_not(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, tmp_path)

    _plotter(scan_indices=[1], bins=8, bin_width=None, xlim=(0.0, 0.08)).plot(
        distribution_bundle(), tmp_path, "png"
    )

    ax = captured["axes"]
    assert len(_line_x(ax, 0)) == 8


def test_labels_and_colours_must_match_the_series_count(tmp_path):
    with pytest.raises(ValueError, match="'labels' has 1 entries but 2 series"):
        _plotter(scan_indices=[0, 1], bins=10, bin_width=None, labels=["only"]).plot(
            distribution_bundle(), tmp_path, "png"
        )
    with pytest.raises(ValueError, match="'colors' has 3 entries but 2 series"):
        _plotter(
            scan_indices=[0, 1],
            bins=10,
            bin_width=None,
            colors=["#000", "#111", "#222"],
        ).plot(distribution_bundle(), tmp_path, "png")


def test_default_colours_come_from_the_palette(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, tmp_path)

    _plotter(scan_indices=[0, 1], bins=10, bin_width=None).plot(
        distribution_bundle(), tmp_path, "png"
    )

    colours = [line.get_color() for line in captured["lines"]]
    assert PALETTE["cam"] in colours
    assert PALETTE["sde"] in colours


def test_reference_series_always_lands_on_zero(tmp_path, monkeypatch):
    for scale in (-1.0, 1.0, 2.5):
        captured = _capture(monkeypatch, tmp_path)
        _plotter(
            scan_indices=[0, 3],
            bins=30,
            bin_width=None,
            center="reference_mean",
            scale=scale,
            show_means=True,
        ).plot(distribution_bundle(), tmp_path, "png")
        means = _vertical_markers(captured["axes"])
        assert means[0] == pytest.approx(0.0, abs=1e-9), scale


# --- annotations and styling ----------------------------------------------


def test_fwhm_and_mean_markers_are_drawn(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, tmp_path)

    _plotter(
        scan_indices=[0],
        bins=20,
        bin_width=None,
        show_fwhm=True,
        show_means=True,
    ).plot(distribution_bundle(), tmp_path, "png")

    ax = captured["axes"]
    # Two half-maximum bounds plus one mean.
    assert len(_vertical_markers(ax)) == 3


def test_single_bin_histogram_draws_without_fwhm(tmp_path):
    """A one-bin histogram is plottable; only the width annotation needs more."""
    files = _plotter(
        scan_indices=[0],
        bin_width=0.1,
        xlim=(-0.12, -0.03),
    ).plot(distribution_bundle(), tmp_path, "png")

    assert files[0].is_file()
    assert files[0].stat().st_size > 0


def test_single_bin_histogram_rejects_fwhm(tmp_path):
    with pytest.raises(ValueError, match="at least two bins"):
        _plotter(
            scan_indices=[0],
            bin_width=0.1,
            xlim=(-0.12, -0.03),
            show_fwhm=True,
        ).plot(distribution_bundle(), tmp_path, "png")


def test_boundary_peak_without_fwhm_still_draws(tmp_path):
    """A peak at the edge of the range is fine unless half-max is requested."""
    files = _plotter(
        scan_indices=[0],
        bins=10,
        bin_width=None,
        xlim=(-0.1, -0.09),
    ).plot(distribution_bundle(), tmp_path, "png")

    assert files[0].is_file()


def test_boundary_peak_with_fwhm_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="reaches the start of the range"):
        _plotter(
            scan_indices=[0],
            bins=10,
            bin_width=None,
            xlim=(-0.1, -0.09),
            show_fwhm=True,
        ).plot(distribution_bundle(), tmp_path, "png")


def test_annotation_fontsize_inherits_or_overrides(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, tmp_path)
    _plotter(
        scan_indices=[0, 2],
        bins=20,
        bin_width=None,
        show_means=True,
        show_fwhm=True,
        shift_label="shift",
        width_label="width",
    ).plot(distribution_bundle(), tmp_path, "png")
    # No explicit size: the annotation inherits font.size from the style.
    assert captured["text_sizes"]["shift"] == pytest.approx(12.0)
    assert captured["text_sizes"]["width"] == pytest.approx(12.0)

    captured = _capture(monkeypatch, tmp_path)
    _plotter(
        scan_indices=[0, 2],
        bins=20,
        bin_width=None,
        show_means=True,
        show_fwhm=True,
        shift_label="shift",
        width_label="width",
        annotation_fontsize=9.0,
    ).plot(distribution_bundle(), tmp_path, "png")
    assert captured["text_sizes"]["shift"] == pytest.approx(9.0)
    assert captured["text_sizes"]["width"] == pytest.approx(9.0)


def test_window_normalisation_uses_the_full_sample_count(tmp_path, monkeypatch):
    """A clipped window keeps the density of the complete distribution."""
    captured = _capture(monkeypatch, tmp_path)
    dataset = distribution_dataset()
    raw = np.asarray(dataset.handle("estimate").materialize(), dtype=float)
    finite = raw[1][np.isfinite(raw[1])]
    lower, upper, bins = -0.02, 0.0, 10

    _plotter(
        scan_indices=[1],
        bins=bins,
        bin_width=None,
        xlim=(lower, upper),
    ).plot(distribution_bundle(), tmp_path, "png")

    centres, density = _line_x(captured["axes"], 0), _line_y(captured["axes"], 0)
    width = (upper - lower) / bins
    edges = np.linspace(lower, upper, bins + 1)
    counts, _ = np.histogram(finite, bins=edges)
    # Denominator is every finite sample, not just the ones inside the window.
    expected = counts / (finite.size * width)
    np.testing.assert_allclose(density, expected, rtol=1e-12)
    # A clipped distribution therefore integrates to less than one.
    assert float(np.sum(density) * width) < 1.0
    assert len(centres) == bins


def test_shift_and_width_labels_are_drawn(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, tmp_path)

    _plotter(
        scan_indices=[0, 2],
        bins=20,
        bin_width=None,
        show_means=True,
        show_fwhm=True,
        shift_label="signal shift",
        width_label="noise width",
    ).plot(distribution_bundle(), tmp_path, "png")

    assert "signal shift" in captured["texts"]
    assert "noise width" in captured["texts"]


def test_shift_label_needs_two_series(tmp_path):
    with pytest.raises(ValueError, match="at least two scan points"):
        _plotter(
            scan_indices=[0],
            bins=10,
            bin_width=None,
            show_means=True,
            shift_label="shift",
        ).plot(distribution_bundle(), tmp_path, "png")


def test_xticks_and_hidden_yticks_are_applied(tmp_path):
    files = _plotter(
        scan_indices=[0, 2],
        bins=20,
        bin_width=None,
        xticks=[-0.005, 0.0, 0.005],
        xticklabels=["a", "b", "c"],
        hide_yticks=True,
    ).plot(distribution_bundle(), tmp_path, "svg")

    svg = files[0].read_text(encoding="utf-8")
    assert ">a<" in svg and ">b<" in svg and ">c<" in svg


def test_title_labels_and_annotation_reach_the_svg(tmp_path):
    files = _plotter(
        scan_indices=[0],
        bins=20,
        bin_width=None,
        title="Distribution title",
        xlabel="Shift",
        ylabel="Density",
        annotations=[{"text": "note", "x": 0.05, "y": 0.9}],
    ).plot(distribution_bundle(), tmp_path, "svg")

    svg = files[0].read_text(encoding="utf-8")
    assert "Distribution title" in svg
    assert "Shift" in svg
    assert "Density" in svg
    assert "note" in svg


def test_restored_artifact_input_renders(tmp_path):
    bundle = restored(distribution_bundle(), tmp_path / "artifact")

    files = _plotter(scan_indices=[0, 1], bins=20, bin_width=None).plot(
        bundle, tmp_path / "plots", "png"
    )

    assert files[0].is_file()
    assert files[0].stat().st_size > 0


# --- error paths -----------------------------------------------------------


def test_non_typed_input_is_rejected(tmp_path):
    class NotTyped:
        products = {"x": {"estimate": [1.0]}}

    with pytest.raises(TypeError, match="typed DatasetResultProtocol"):
        _plotter(bins=10, bin_width=None).plot(NotTyped(), tmp_path, "png")


def test_unknown_variable_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no typed product carries"):
        _plotter(value_variable="nope", bins=10, bin_width=None).plot(
            distribution_bundle(), tmp_path, "png"
        )


def test_named_product_missing_variable_is_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"has no variable 'nope'"):
        _plotter(
            product="estimates", value_variable="nope", bins=10, bin_width=None
        ).plot(distribution_bundle(), tmp_path, "png")


def test_ambiguous_product_is_rejected(tmp_path):
    products = distribution_bundle().products
    bundle = VizTestBundle({**products, "copy": products["estimates"]})

    with pytest.raises(ValueError, match="ambiguous"):
        _plotter(bins=10, bin_width=None).plot(bundle, tmp_path, "png")


def test_all_non_finite_samples_are_rejected(tmp_path):
    dataset = distribution_dataset()
    arrays = {
        variable.name: np.asarray(dataset.handle(variable.name).materialize())
        for variable in dataset.variables
    }
    arrays["estimate"] = np.full_like(arrays["estimate"], np.nan)
    broken = StatisticsDataset.from_arrays(dataset.schema, arrays, owner="test.viz")
    bundle = VizTestBundle(
        {"estimates": broken}, axes={"omega": np.asarray(SCAN_POINTS)}
    )

    with pytest.raises(ValueError, match="no finite sample"):
        _plotter(scan_indices=[0], bins=10, bin_width=None).plot(
            bundle, tmp_path, "png"
        )


def test_scan_selection_on_a_single_point_variable_is_rejected(tmp_path):
    dataset = distribution_dataset()
    arrays = {
        variable.name: np.asarray(dataset.handle(variable.name).materialize())
        for variable in dataset.variables
    }
    point = StatisticsDataset.from_arrays(
        dataset.point_view(scan=0).schema,
        {
            name: value
            for name, value in (
                (variable.name, arrays[variable.name][0])
                for variable in dataset.point_view(scan=0).variables
            )
        },
        owner="test.viz",
    )
    bundle = VizTestBundle({"estimates": point})

    with pytest.raises(ValueError, match="no scan axis"):
        _plotter(scan_indices=[0], bins=10, bin_width=None).plot(
            bundle, tmp_path, "png"
        )
