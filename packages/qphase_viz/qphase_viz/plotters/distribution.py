"""qphase_viz: Sample Distribution Plotter
---------------------------------------------------------
Renders the distribution of already-computed per-realization samples: one
density curve per selected scan point, drawn from ``value[scan, realization]``.

The plotter never computes an estimate and never reads a file: the samples come
from a typed product, the transform and the histogram are explicit, and the
per-series label, colour and annotation text are configuration.

Public API
----------
``DistributionPlotter``
``fwhm_bounds`` : Half-maximum width of one binned density, by linear crossing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from matplotlib.axes import Axes
from qphase.core.dataset import DatasetResultProtocol
from qphase.data import Dataset

from ..config import DistributionConfig, DistributionSpec
from ..render import apply_axes_style, draw_annotations, save_figure, styled_subplots
from ..styles import PALETTE, RenderOptions
from ..typed_input import bundle_products, scan_values
from .base import PlotterProtocol

#: Default per-series colours, in the order the series are drawn.
_DEFAULT_COLOURS = ("cam", "sde", "green", "purple", "accent", "magenta")

#: Colour of the shift annotation, kept distinct from every series colour.
_ANNOTATION_COLOUR = PALETTE["ink"]

__all__ = ["DistributionPlotter", "fwhm_bounds"]


def fwhm_bounds(x: np.ndarray, density: np.ndarray) -> tuple[float, float]:
    """Return the half-maximum width bounds of one binned density.

    The peak is located by the maximum density, then the walk moves outwards
    while the density stays at or above half the maximum; the two crossing
    positions are linearly interpolated between the neighbouring bin centres.
    A density with no positive maximum, fewer than two bins or mismatched
    shapes raises ``ValueError`` instead of returning a degenerate width.
    """
    x_values = np.asarray(x, dtype=float)
    values = np.asarray(density, dtype=float)
    if x_values.ndim != 1 or values.ndim != 1:
        raise ValueError("fwhm_bounds needs one-dimensional arrays")
    if x_values.size != values.size:
        raise ValueError(
            f"fwhm_bounds needs matching lengths, got {x_values.size} and "
            f"{values.size}"
        )
    if values.size < 2:
        raise ValueError("fwhm_bounds needs at least two bins")
    if not np.any(values > 0.0):
        raise ValueError("fwhm_bounds needs a positive maximum density")

    peak_index = int(np.argmax(values))
    half_maximum = 0.5 * float(values[peak_index])
    left = peak_index
    right = peak_index
    while left > 0 and values[left] >= half_maximum:
        left -= 1
    while right < values.size - 1 and values[right] >= half_maximum:
        right += 1

    left_crossing = _interpolate_crossing(
        float(x_values[left]),
        float(values[left]),
        float(x_values[left + 1]),
        float(values[left + 1]),
        half_maximum,
    )
    right_crossing = _interpolate_crossing(
        float(x_values[right - 1]),
        float(values[right - 1]),
        float(x_values[right]),
        float(values[right]),
        half_maximum,
    )
    return left_crossing, right_crossing


def _interpolate_crossing(
    x0: float, y0: float, x1: float, y1: float, target: float
) -> float:
    if y1 == y0:
        return 0.5 * (x0 + x1)
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


class DistributionPlotter(PlotterProtocol):
    """Plots per-realization sample distributions as density curves."""

    name: ClassVar[str] = "distribution"
    description: ClassVar[str] = "Sample distributions of a typed product"
    config_schema: ClassVar[type[DistributionConfig]] = DistributionConfig

    def __init__(
        self, config: DistributionConfig | None = None, **kwargs: Any
    ) -> None:
        if config is None:
            config = DistributionConfig(**kwargs)
        self.config = config

    def plot(
        self,
        data: DatasetResultProtocol,
        output_dir: Path,
        format: str,
        options: RenderOptions | None = None,
    ) -> list[Path]:
        base = options or RenderOptions()
        return [
            self._plot_single(data, spec, output_dir, format, base)
            for spec in self.config.plots
        ]

    def _plot_single(
        self,
        data: DatasetResultProtocol,
        spec: DistributionSpec,
        output_dir: Path,
        format: str,
        base: RenderOptions,
    ) -> Path:
        product = self._select_product(data, spec)
        samples = self._samples(product, spec, data)
        labels: list[str | None] = (
            list(self._per_series(spec.labels, len(samples), "labels"))
            if spec.labels is not None
            else [None for _ in samples]
        )
        colours = self._per_series(
            spec.colors or self._default_colours(len(samples)),
            len(samples),
            "colors",
        )

        reference = float(np.mean(samples[0])) if spec.center != "none" else 0.0
        offset = reference if spec.center == "reference_mean" else 0.0
        transformed = [spec.scale * values + offset for values in samples]
        means = [float(np.mean(values)) for values in transformed]

        centres, densities = self._histograms(transformed, spec)
        widths = [fwhm_bounds(centres, density) for density in densities]
        maximum = max(float(np.max(density)) for density in densities)

        options = base.merged(
            style=spec.style,
            editable=spec.editable,
            overrides=spec.style_overrides,
        )
        with styled_subplots(
            options,
            figsize=spec.figsize,
            dpi=spec.dpi,
            constrained_layout=spec.constrained_layout,
        ) as (fig, ax):
            axes = ax[0, 0]
            for values, bounds, colour, label in zip(
                densities, widths, colours, labels, strict=True
            ):
                self._draw_density(axes, centres, values, bounds, colour, label, spec)
            if spec.show_means:
                for mean, colour in zip(means, colours, strict=True):
                    axes.vlines(
                        mean,
                        0.0,
                        1.12 * maximum,
                        color=colour,
                        linewidth=1.0,
                        linestyle="--",
                        alpha=0.8,
                    )
            if spec.shift_label is not None:
                if len(means) < 2:
                    raise ValueError(
                        "shift_label compares the first two series; select at "
                        f"least two scan points (got {len(means)})"
                    )
                self._annotate_shift(axes, means, colours, maximum, spec.shift_label)
            if spec.width_label is not None:
                self._annotate_width(
                    axes, widths[0], maximum, colours[0], spec.width_label
                )

            apply_axes_style(
                axes,
                title=spec.title,
                xlabel=spec.xlabel or spec.value_variable,
                ylabel=spec.ylabel,
                xlim=spec.xlim or (float(centres[0]), float(centres[-1])),
                ylim=spec.ylim or (0.0, 1.16 * maximum),
                grid=spec.grid,
            )
            if spec.labels is not None:
                axes.legend(loc="upper left", handlelength=1.2, borderaxespad=0.2)
            if spec.xticks is not None:
                axes.set_xticks(spec.xticks)
                if spec.xticklabels is not None:
                    axes.set_xticklabels(spec.xticklabels)
            if spec.hide_yticks:
                axes.set_yticks([])
            draw_annotations(axes, spec.annotations)
            return save_figure(
                fig,
                output_dir,
                spec.filename or f"distribution_{spec.value_variable}",
                format,
            )

    def _draw_density(
        self,
        ax: Axes,
        centres: np.ndarray,
        density: np.ndarray,
        bounds: tuple[float, float],
        colour: str,
        label: str | None,
        spec: DistributionSpec,
    ) -> None:
        ax.fill_between(centres, 0.0, density, color=colour, alpha=0.13)
        if spec.show_fwhm:
            inside = ((centres >= bounds[0]) & (centres <= bounds[1])).tolist()
            ax.fill_between(
                centres, 0.0, density, where=inside, color=colour, alpha=0.34
            )
            for boundary in bounds:
                ax.vlines(
                    boundary,
                    0.0,
                    0.5 * float(np.max(density)),
                    color=colour,
                    linewidth=1.5,
                    linestyle=":",
                    alpha=0.95,
                )
        ax.plot(centres, density, color=colour, linewidth=1.8, label=label)

    def _annotate_shift(
        self,
        ax: Axes,
        means: list[float],
        colours: list[str],
        maximum: float,
        text: str,
    ) -> None:
        height = 1.04 * maximum
        ax.annotate(
            "",
            xy=(means[1], height),
            xytext=(means[0], height),
            arrowprops={
                "arrowstyle": "<->",
                "color": _ANNOTATION_COLOUR,
                "linewidth": 1.2,
            },
        )
        ax.text(
            0.5 * (means[0] + means[1]),
            height - 0.12 * maximum,
            text,
            color=_ANNOTATION_COLOUR,
            fontsize=13.5,
            ha="center",
            va="bottom",
        )

    def _annotate_width(
        self,
        ax: Axes,
        bounds: tuple[float, float],
        maximum: float,
        colour: str,
        text: str,
    ) -> None:
        height = 0.29 * maximum
        ax.annotate(
            "",
            xy=(bounds[1], height),
            xytext=(bounds[0], height),
            arrowprops={"arrowstyle": "<->", "color": colour, "linewidth": 1.25},
        )
        ax.text(
            0.5 * (bounds[0] + bounds[1]),
            height - 0.045 * maximum,
            text,
            color=colour,
            fontsize=13.5,
            ha="center",
            va="top",
        )

    def _select_product(
        self, data: DatasetResultProtocol, spec: DistributionSpec
    ) -> Dataset:
        _, products = bundle_products(data)
        if spec.product is not None:
            product = products.get(spec.product)
            if product is None:
                raise ValueError(
                    f"bundle has no product {spec.product!r}; "
                    f"available: {sorted(products)}"
                )
            if spec.value_variable not in {
                variable.name for variable in product.variables
            }:
                raise ValueError(
                    f"product {spec.product!r} has no variable "
                    f"{spec.value_variable!r}; available: "
                    f"{sorted(variable.name for variable in product.variables)}"
                )
            return product

        matches = [
            name
            for name, product in products.items()
            if spec.value_variable in {v.name for v in product.variables}
        ]
        if not matches:
            raise ValueError(
                f"no typed product carries the sample variable "
                f"{spec.value_variable!r}"
            )
        if len(matches) > 1:
            raise ValueError(
                "distribution product selection is ambiguous; set 'product' to "
                f"one of {sorted(matches)}"
            )
        return products[matches[0]]

    def _samples(
        self,
        product: Dataset,
        spec: DistributionSpec,
        data: DatasetResultProtocol,
    ) -> list[np.ndarray]:
        """Return one finite sample array per selected scan point."""
        values = np.asarray(product.handle(spec.value_variable).materialize())
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            if spec.scan_indices is not None or spec.scan_coordinates is not None:
                raise ValueError(
                    f"variable {spec.value_variable!r} has no scan axis; scan "
                    "selection needs value[scan, realization]"
                )
            selected = [0]
            pool = [values]
            coordinates: np.ndarray | None = None
        elif values.ndim == 2:
            if spec.scan_parameter is None:
                raise ValueError(
                    f"variable {spec.value_variable!r} has a scan axis; set "
                    "'scan_parameter' to name the stored scan coordinate"
                )
            coordinates = scan_values(data, product, spec.scan_parameter)
            if coordinates.size != values.shape[0]:
                raise ValueError(
                    f"scan coordinate {spec.scan_parameter!r} has "
                    f"{coordinates.size} points but the samples have "
                    f"{values.shape[0]}"
                )
            selected = self._select_scan_points(coordinates, spec)
            pool = [values[index] for index in selected]
        else:
            raise ValueError(
                f"variable {spec.value_variable!r} must be [scan, realization] "
                f"or [realization], got rank {values.ndim}"
            )

        samples = []
        for index, raw in zip(selected, pool, strict=True):
            finite = raw[np.isfinite(raw)]
            if finite.size == 0:
                where = (
                    f"scan point {int(coordinates[index])}"
                    if coordinates is not None
                    else "the only sample set"
                )
                raise ValueError(
                    f"variable {spec.value_variable!r} has no finite sample at "
                    f"{where}"
                )
            samples.append(finite)
        return samples

    @staticmethod
    def _select_scan_points(
        coordinates: np.ndarray, spec: DistributionSpec
    ) -> list[int]:
        if spec.scan_indices is not None and spec.scan_coordinates is not None:
            raise ValueError(
                "choose scan points by either scan_indices or scan_coordinates, "
                "not both"
            )
        if spec.scan_indices is not None:
            selected: list[int] = []
            for index in spec.scan_indices:
                if not 0 <= index < coordinates.size:
                    raise ValueError(
                        f"scan index {index} is out of range for "
                        f"{coordinates.size} scan points"
                    )
                if index not in selected:
                    selected.append(int(index))
        elif spec.scan_coordinates is not None:
            if spec.scan_tolerance is None:
                raise ValueError(
                    "scan_coordinates requires an explicit scan_tolerance"
                )
            selected = []
            for value in spec.scan_coordinates:
                distance = np.abs(coordinates - float(value))
                index = int(np.argmin(distance))
                if float(distance[index]) > float(spec.scan_tolerance):
                    raise ValueError(
                        f"scan coordinate {value!r} is "
                        f"{float(distance[index]):.6g} away from the nearest "
                        f"point {float(coordinates[index]):.6g} "
                        f"(tolerance {spec.scan_tolerance:.6g})"
                    )
                if index not in selected:
                    selected.append(index)
        else:
            selected = list(range(coordinates.size))
        if not selected:
            raise ValueError("the distribution figure selects no scan points")
        return selected

    @staticmethod
    def _per_series(configured: list[str], count: int, name: str) -> list[str]:
        if len(configured) != count:
            raise ValueError(
                f"'{name}' has {len(configured)} entries but {count} series are "
                "selected"
            )
        return list(configured)

    @staticmethod
    def _default_colours(count: int) -> list[str]:
        return [
            PALETTE[_DEFAULT_COLOURS[index % len(_DEFAULT_COLOURS)]]
            for index in range(count)
        ]

    @staticmethod
    def _histograms(
        samples: list[np.ndarray], spec: DistributionSpec
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Bin every series on the same edges and normalise to a density."""
        if spec.xlim is not None:
            lower, upper = float(spec.xlim[0]), float(spec.xlim[1])
        else:
            lower = min(float(np.min(values)) for values in samples)
            upper = max(float(np.max(values)) for values in samples)
        if not upper > lower:
            raise ValueError(
                f"the sample range is degenerate ({lower:g}..{upper:g}); set xlim"
            )

        if spec.bin_width is not None:
            edges = np.arange(
                lower, upper + 0.5 * spec.bin_width, spec.bin_width
            )
            width = float(spec.bin_width)
        else:
            assert spec.bins is not None
            edges = np.linspace(lower, upper, spec.bins + 1)
            width = float((upper - lower) / spec.bins)
        if edges.size < 2:
            raise ValueError(
                "the histogram needs at least one bin; widen xlim or lower "
                "bin_width"
            )

        centres = 0.5 * (edges[:-1] + edges[1:])
        densities = []
        for index, values in enumerate(samples):
            counts, _ = np.histogram(values, bins=edges)
            if not np.any(counts):
                raise ValueError(
                    f"series {index} has no sample inside the histogram range "
                    f"[{lower:g}, {upper:g}]; widen xlim or check the selection"
                )
            densities.append(counts.astype(float) / (values.size * width))
        return centres, densities
