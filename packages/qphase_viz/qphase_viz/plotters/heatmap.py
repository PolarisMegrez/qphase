"""qphase_viz: Scalar Heatmap Plotter
---------------------------------------------------------
Renders a two-dimensional scalar grid stored in a typed product: one
``pcolormesh`` plus a colorbar, optional validity mask, contour overlays and
explicit point or line series.

Public API
----------
``HeatmapPlotter``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from qphase.core.dataset import DatasetResultProtocol
from qphase.data import Dataset

from ..config import HeatmapConfig, HeatmapSpec
from ..render import apply_axes_style, draw_annotations, save_figure, styled_subplots
from ..styles import RenderOptions, resolve_colormap
from ..typed_input import bundle_products
from .base import PlotterProtocol


class HeatmapPlotter(PlotterProtocol):
    """Plots one scalar mesh with an optional mask and overlays."""

    name: ClassVar[str] = "heatmap"
    description: ClassVar[str] = "Scalar heatmap of a typed product grid"
    config_schema: ClassVar[type[HeatmapConfig]] = HeatmapConfig

    def __init__(self, config: HeatmapConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = HeatmapConfig(**kwargs)
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
        spec: HeatmapSpec,
        output_dir: Path,
        format: str,
        base: RenderOptions,
    ) -> Path:
        product = self._select_product(data, spec)
        x = self._coordinate(product, spec.x_variable)
        y = self._coordinate(product, spec.y_variable)
        values = self._grid(product, spec, x.size, y.size)
        mask = self._mask(product, spec, values.shape)
        # Non-finite cells are blanked, never drawn as zero; the mask adds
        # the product-declared invalid region on top of that.
        mesh_values = np.ma.masked_array(values, mask | ~np.isfinite(values))
        finite = mesh_values.compressed()
        if finite.size == 0:
            raise ValueError(
                f"heatmap variable {spec.value_variable!r} has no finite valid "
                "values in the selected grid"
            )
        norm = self._norm(spec, finite)
        cmap = resolve_colormap(spec.cmap)

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
        ) as (fig, axes):
            ax = axes[0, 0]
            mesh = ax.pcolormesh(
                x,
                y,
                mesh_values.T,
                shading="auto",
                cmap=cmap,
                norm=norm,
                rasterized=spec.rasterized,
            )
            colorbar = fig.colorbar(mesh, ax=ax, pad=0.03)
            colorbar.set_label(spec.colorbar_label or spec.value_variable)
            self._draw_contours(ax, product, spec, x, y, mask)
            legend = self._draw_series(ax, product, spec)
            apply_axes_style(
                ax,
                title=spec.title,
                xlabel=spec.xlabel or spec.x_variable,
                ylabel=spec.ylabel or spec.y_variable,
                xlim=spec.xlim,
                ylim=spec.ylim,
                grid=spec.grid,
                legend=legend,
            )
            draw_annotations(ax, spec.annotations)
            return save_figure(
                fig,
                output_dir,
                spec.filename or f"heatmap_{spec.value_variable}",
                format,
            )

    def _select_product(
        self, data: DatasetResultProtocol, spec: HeatmapSpec
    ) -> Dataset:
        fields = (spec.x_variable, spec.y_variable, spec.value_variable)
        # Rejects a non-typed input and any non-Dataset product value, the same
        # gate every other plotter passes through.
        _, products = bundle_products(data)
        if spec.product is not None:
            product = products.get(spec.product)
            if product is None:
                raise ValueError(
                    f"bundle has no product {spec.product!r}; "
                    f"available: {sorted(products)}"
                )
            missing = {
                name
                for name in fields
                if name not in {variable.name for variable in product.variables}
            }
            if missing:
                raise ValueError(
                    f"product {spec.product!r} is missing variables {sorted(missing)}"
                )
            if spec.valid_variable is not None:
                _require_variable(product, spec.valid_variable, "mask")
            return product

        matches = [
            name
            for name, product in products.items()
            if set(fields) <= {variable.name for variable in product.variables}
        ]
        if not matches:
            raise ValueError(
                f"no typed product carries the heatmap variables {list(fields)}"
            )
        if len(matches) > 1:
            raise ValueError(
                "heatmap product selection is ambiguous; set 'product' to one "
                f"of {sorted(matches)}"
            )
        return products[matches[0]]

    @staticmethod
    def _coordinate(product: Dataset, name: str) -> np.ndarray:
        values = np.asarray(
            _require_variable(product, name, "coordinate").materialize(),
            dtype=float,
        )
        if values.ndim != 1:
            raise ValueError(
                f"heatmap coordinate {name!r} must be one-dimensional, "
                f"got rank {values.ndim}"
            )
        if values.size == 0:
            raise ValueError(f"heatmap coordinate {name!r} is empty")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"heatmap coordinate {name!r} is not finite")
        if values.size > 1:
            steps = np.diff(values)
            if not (np.all(steps > 0) or np.all(steps < 0)):
                raise ValueError(
                    f"heatmap coordinate {name!r} is not monotonic; a rectangular "
                    "mesh needs sorted axis coordinates"
                )
        return values

    @staticmethod
    def _grid(
        product: Dataset, spec: HeatmapSpec, x_size: int, y_size: int
    ) -> np.ndarray:
        values = np.asarray(
            _require_variable(product, spec.value_variable, "value").materialize(),
            dtype=float,
        )
        if values.ndim != 2:
            raise ValueError(
                f"heatmap variable {spec.value_variable!r} must be two-dimensional "
                f"(row, column), got rank {values.ndim}"
            )
        if values.shape != (x_size, y_size):
            raise ValueError(
                f"heatmap variable {spec.value_variable!r} has shape "
                f"{values.shape} but the axes imply {(x_size, y_size)} "
                "(row = x, column = y)"
            )
        return values

    @staticmethod
    def _mask(
        product: Dataset, spec: HeatmapSpec, shape: tuple[int, int]
    ) -> np.ndarray:
        """Return the boolean mask where ``True`` means "leave this cell blank"."""
        if spec.valid_variable is None:
            return np.zeros(shape, dtype=bool)
        mask = np.asarray(
            _require_variable(product, spec.valid_variable, "mask").materialize()
        )
        if mask.dtype != bool:
            if not np.all(np.isin(mask, (0, 1))):
                raise ValueError(
                    f"heatmap mask {spec.valid_variable!r} must be boolean, "
                    f"got dtype {mask.dtype}"
                )
            mask = mask.astype(bool)
        if mask.shape != shape:
            raise ValueError(
                f"heatmap mask {spec.valid_variable!r} has shape {mask.shape} "
                f"but the grid is {shape}"
            )
        return ~mask

    @staticmethod
    def _norm(spec: HeatmapSpec, finite: np.ndarray) -> Normalize:
        if spec.norm == "log":
            if np.any(finite <= 0.0):
                raise ValueError(
                    f"heatmap variable {spec.value_variable!r} has non-positive "
                    "values; a log norm needs strictly positive data"
                )
            vmin = float(np.min(finite)) if spec.vmin is None else spec.vmin
            vmax = float(np.max(finite)) if spec.vmax is None else spec.vmax
            if vmin <= 0.0 or vmax <= vmin:
                raise ValueError(
                    "a log norm needs 0 < vmin < vmax; "
                    f"got vmin={vmin:g}, vmax={vmax:g}"
                )
            return LogNorm(vmin=vmin, vmax=vmax)
        vmin = float(np.min(finite)) if spec.vmin is None else spec.vmin
        vmax = float(np.max(finite)) if spec.vmax is None else spec.vmax
        if vmax == vmin:
            raise ValueError(
                f"heatmap colour range is degenerate ({vmin:g}); set vmin/vmax"
            )
        return Normalize(vmin=vmin, vmax=vmax)

    def _draw_contours(
        self,
        ax: Axes,
        product: Dataset,
        spec: HeatmapSpec,
        x: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        for contour in spec.contours:
            if not contour.levels:
                raise ValueError("heatmap contour needs at least one level")
            if contour.variable is None:
                if spec.valid_variable is None:
                    raise ValueError(
                        "contouring the mask needs 'valid_variable' on the "
                        "heatmap spec, or name a contour 'variable'"
                    )
                field = (~mask).astype(float)
            else:
                field = np.asarray(
                    _require_variable(
                        product, contour.variable, "contour"
                    ).materialize(),
                    dtype=float,
                )
                if field.shape != x.shape + y.shape:
                    raise ValueError(
                        f"contour variable {contour.variable!r} has shape "
                        f"{field.shape} but the grid is {(x.size, y.size)}"
                    )
            ax.contour(
                x,
                y,
                field.T,
                levels=list(contour.levels),
                colors=[contour.color],
                linestyles=[contour.linestyle],
                linewidths=[contour.linewidth],
            )

    @staticmethod
    def _draw_series(ax: Axes, product: Dataset, spec: HeatmapSpec) -> bool:
        """Draw the configured series and report whether any carries a label."""
        labelled = False
        for series in spec.series:
            x = _series_values(product, series.x_variable, series.x)
            y = _series_values(product, series.y_variable, series.y)
            if x.size != y.size:
                raise ValueError(
                    f"heatmap series x and y lengths differ ({x.size} vs {y.size})"
                )
            ax.plot(
                x,
                y,
                color=series.color,
                linestyle=series.linestyle if series.linestyle is not None else "none",
                linewidth=series.linewidth,
                marker=series.marker,
                markersize=series.markersize,
                markerfacecolor=(
                    series.marker_face_color
                    if series.marker_face_color is not None
                    else series.color
                ),
                label=series.label,
            )
            labelled = labelled or series.label is not None
        return labelled


def _require_variable(product: Dataset, name: str | None, role: str) -> Any:
    """Return one product variable handle, or explain which ones exist."""
    if name is None:
        raise ValueError(f"heatmap {role} needs a variable name")
    if name not in {variable.name for variable in product.variables}:
        raise ValueError(
            f"heatmap {role} variable {name!r} is not in the product; "
            f"available: {sorted(variable.name for variable in product.variables)}"
        )
    return product.handle(name)


def _series_values(
    product: Dataset, variable: str | None, constants: list[float]
) -> np.ndarray:
    if variable is None:
        return np.asarray(constants, dtype=float)
    values = np.asarray(
        _require_variable(product, variable, "series").materialize(), dtype=float
    )
    if values.ndim != 1:
        raise ValueError(
            f"heatmap series variable {variable!r} must be one-dimensional, "
            f"got rank {values.ndim}"
        )
    return values
