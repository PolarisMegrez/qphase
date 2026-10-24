"""qphase_viz: PSD-across-scan Plotter
---------------------------------------------------------
Renders one panel per mode for a multi-point PSD scan: every selected scan
point contributes one curve, coloured by its scan coordinate.

Public API
----------
``SpectralScanPlotter``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from qphase.core.dataset import DatasetResultProtocol
from qphase.data import DataKind, Dataset, SpectralQuantity

from ..config import SpectralScanConfig, SpectralScanSpec
from ..render import apply_axes_style, draw_annotations, save_figure, styled_subplots
from ..styles import RenderOptions, resolve_colormap
from ..typed_input import scan_values, select_product
from .base import PlotterProtocol

#: Fraction of the scan colormap the curves are sampled from, so the lightest
#: and darkest ends stay distinguishable on white paper.
_CMAP_SPAN = (0.12, 0.92)


class SpectralScanPlotter(PlotterProtocol):
    """Plots a multi-point PSD scan as one panel per mode."""

    name: ClassVar[str] = "spectral_scan"
    description: ClassVar[str] = "PSD across a scan, one panel per mode"
    config_schema: ClassVar[type[SpectralScanConfig]] = SpectralScanConfig

    def __init__(
        self, config: SpectralScanConfig | None = None, **kwargs: Any
    ) -> None:
        if config is None:
            config = SpectralScanConfig(**kwargs)
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
        spec: SpectralScanSpec,
        output_dir: Path,
        format: str,
        base: RenderOptions,
    ) -> Path:
        product = select_product(
            data,
            role="primary_spectrum",
            kind=DataKind.SPECTRAL,
            quantity=SpectralQuantity.POWER_SPECTRAL_DENSITY.value,
            fields=("axis", "psd"),
        )
        frequency, psd, modes = self._materialize(product)
        coordinates = scan_values(data, product, spec.scan_parameter)
        if coordinates.size != psd.shape[0]:
            raise ValueError(
                f"scan coordinate {spec.scan_parameter!r} has {coordinates.size} "
                f"points but the PSD has {psd.shape[0]} scan points"
            )
        selected = self._select_scan_points(coordinates, spec)
        channels = self._select_channels(modes, spec.channels)
        floor = self._log_floor(psd) if spec.log_scale else None
        # The colour of a scan point comes from its position in the whole
        # scan, not from its position among the selected points: a subset
        # must not repaint the remaining curves.
        colours = self._curve_colours(spec.cmap, coordinates.size)

        options = base.merged(
            style=spec.style,
            editable=spec.editable,
            overrides=spec.style_overrides,
        )
        rows = len(channels)
        with styled_subplots(
            options,
            rows=rows,
            cols=1,
            figsize=spec.figsize,
            dpi=spec.dpi,
            sharex=True,
            sharey=True,
            constrained_layout=spec.constrained_layout,
        ) as (fig, axes):
            for row, mode in enumerate(channels):
                ax = axes[row, 0]
                column = modes.index(mode)
                for scan_index in selected:
                    colour = colours[scan_index]
                    values = psd[scan_index, :, column]
                    if floor is not None:
                        values = np.maximum(values, floor)
                    label = spec.label_template.format(
                        name=spec.scan_parameter,
                        value=float(coordinates[scan_index]),
                        index=int(scan_index),
                    )
                    ax.plot(
                        frequency,
                        values,
                        color=colour,
                        linewidth=1.0,
                        label=label,
                    )
                if spec.log_scale:
                    ax.set_yscale("log")
                apply_axes_style(
                    ax,
                    xlabel=(
                        spec.xlabel or "Frequency"
                        if row == rows - 1
                        else None
                    ),
                    ylabel=spec.ylabel
                    or spec.mode_label_template.format(mode=mode),
                    xlim=spec.xlim,
                    ylim=spec.ylim,
                    grid=spec.grid,
                    legend=row == 0,
                )
            if spec.title:
                fig.suptitle(spec.title)
            draw_annotations(axes[0, 0], spec.annotations)
            return save_figure(
                fig,
                output_dir,
                spec.filename or f"spectral_scan_{spec.scan_parameter}",
                format,
            )

    def _materialize(
        self, product: Dataset
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Return ``(frequency, psd[scan, frequency, mode], modes)``."""
        psd = np.asarray(product.handle("psd").materialize(), dtype=float)
        if psd.ndim != 3:
            raise ValueError(
                "a PSD scan needs one scan axis: expected psd[scan, frequency, "
                f"mode], got rank {psd.ndim}; single-point PSDs belong to the "
                "power_spectrum plotter"
            )
        frequency = np.asarray(product.handle("axis").materialize(), dtype=float)
        if frequency.ndim == 2:
            if not np.allclose(frequency, frequency[0]):
                raise ValueError("PSD scan points use different frequency axes")
            frequency = frequency[0]
        if frequency.ndim != 1:
            raise ValueError("PSD frequency axis must be one-dimensional")
        if not np.all(np.isfinite(frequency)):
            raise ValueError("PSD frequency axis is not finite")
        if frequency.size > 1 and not np.all(np.diff(frequency) > 0):
            raise ValueError("PSD frequency axis is not strictly increasing")
        if psd.shape[1] != frequency.size:
            raise ValueError(
                f"PSD frequency axis has {frequency.size} points but the values "
                f"have {psd.shape[1]}"
            )

        variables = {variable.name for variable in product.variables}
        if "modes" in variables:
            modes = [
                int(mode) for mode in product.handle("modes").materialize()
            ]
        else:
            modes = list(range(psd.shape[2]))
        if len(modes) != psd.shape[2]:
            raise ValueError(
                f"PSD declares {len(modes)} modes but stores {psd.shape[2]} columns"
            )
        return frequency, psd, modes

    def _select_scan_points(
        self, coordinates: np.ndarray, spec: SpectralScanSpec
    ) -> list[int]:
        """Resolve the scan points to draw, by index or by coordinate value."""
        if spec.scan_indices is not None and spec.scan_coordinates is not None:
            raise ValueError(
                "choose scan points by either scan_indices or scan_coordinates, "
                "not both"
            )
        if spec.scan_indices is not None:
            selected = self._validate_indices(spec.scan_indices, coordinates.size)
        elif spec.scan_coordinates is not None:
            selected = self._nearest_indices(
                coordinates, spec.scan_coordinates, spec.scan_tolerance
            )
        else:
            selected = list(range(coordinates.size))
        if not selected:
            raise ValueError("the PSD scan figure selects no scan points")
        return selected

    @staticmethod
    def _validate_indices(indices: list[int], size: int) -> list[int]:
        selected: list[int] = []
        for index in indices:
            if not 0 <= index < size:
                raise ValueError(
                    f"scan index {index} is out of range for {size} scan points"
                )
            if index not in selected:
                selected.append(int(index))
        return selected

    @staticmethod
    def _nearest_indices(
        coordinates: np.ndarray,
        requested: list[float],
        tolerance: float | None,
    ) -> list[int]:
        """Map requested coordinate values onto the nearest scan points.

        The tolerance is mandatory: without an explicit bound a value that only
        roughly matches would silently draw a different scan point.
        """
        if tolerance is None:
            raise ValueError(
                "scan_coordinates requires an explicit scan_tolerance"
            )
        limit = float(tolerance)
        selected: list[int] = []
        for value in requested:
            distance = np.abs(coordinates - float(value))
            index = int(np.argmin(distance))
            if float(distance[index]) > limit:
                raise ValueError(
                    f"scan coordinate {value!r} is {float(distance[index]):.6g} "
                    f"away from the nearest point {float(coordinates[index]):.6g} "
                    f"(tolerance {limit:.6g})"
                )
            if index not in selected:
                selected.append(index)
        return selected

    @staticmethod
    def _select_channels(modes: list[int], channels: list[int]) -> list[int]:
        if not channels:
            raise ValueError("the PSD scan figure selects no modes")
        unknown = [channel for channel in channels if channel not in modes]
        if unknown:
            raise ValueError(
                f"PSD modes {unknown} are not in the stored modes {modes}"
            )
        return list(dict.fromkeys(channels))

    @staticmethod
    def _log_floor(psd: np.ndarray) -> float:
        """Floor that keeps empty PSD bins visible on a log axis."""
        positive = psd[np.isfinite(psd) & (psd > 0.0)]
        if positive.size == 0:
            raise ValueError(
                "the PSD has no positive finite values; a log scale needs at "
                "least one"
            )
        return max(0.5 * float(np.min(positive)), float(np.finfo(float).tiny))

    @staticmethod
    def _curve_colours(cmap: str, count: int) -> list[Any]:
        """One colour per scan point, sampled across the whole scan."""
        colours = resolve_colormap(cmap)(np.linspace(*_CMAP_SPAN, max(count, 1)))
        return [tuple(float(channel) for channel in colour) for colour in colours]
