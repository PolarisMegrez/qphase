"""qphase_viz: Evolution Plotters
---------------------------------------------------------
Plotters for time-evolution data.

Public API
----------
``TimeSeriesPlotter``
"""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from qphase.core.dataset import DatasetResultProtocol
from qphase.data import DataKind

from ..config import TimeSeriesConfig, TimeSeriesSpec
from ..render import apply_axes_style, draw_annotations, save_figure, styled_figure
from ..styles import RenderOptions
from ..typed_input import (
    require_single_scan_point,
    select_product,
    time_values,
    trajectory_array,
)
from .base import PlotterProtocol


class TimeSeriesPlotter(PlotterProtocol):
    """Plots time series data (y vs t)."""

    name: ClassVar[str] = "time_series"
    description: ClassVar[str] = "Time Series Plotter"
    config_schema: ClassVar[type[TimeSeriesConfig]] = TimeSeriesConfig

    def __init__(self, config: TimeSeriesConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = TimeSeriesConfig(**kwargs)
        self.config = config

    def plot(
        self,
        data: DatasetResultProtocol,
        output_dir: Path,
        format: str,
        options: RenderOptions | None = None,
    ) -> list[Path]:
        base = options or RenderOptions()
        generated_files = []
        for spec in self.config.plots:
            generated_files.append(
                self._plot_single(data, spec, output_dir, format, base)
            )
        return generated_files

    def _plot_single(
        self,
        data: DatasetResultProtocol,
        spec: TimeSeriesSpec,
        output_dir: Path,
        format: str,
        base: RenderOptions,
    ) -> Path:
        product = select_product(
            data,
            role="trajectories",
            kind=DataKind.TIME_SERIES,
            quantity="field_amplitude",
            fields=("alpha",),
        )
        require_single_scan_point(data, product)
        y = trajectory_array(product)
        y = y.reshape(-1, y.shape[-2], y.shape[-1])
        t = time_values(product)

        options = base.merged(
            style=spec.style,
            editable=spec.editable,
            overrides=spec.style_overrides,
        )

        with styled_figure(options, figsize=spec.figsize, dpi=spec.dpi) as (fig, ax):
            # Data transformation
            for ch in spec.channels:
                y_ch = y[:, :, ch]  # (N, T)

                if spec.transform == "real":
                    val = np.real(y_ch)
                    label_suffix = "Re"
                elif spec.transform == "imag":
                    val = np.imag(y_ch)
                    label_suffix = "Im"
                elif spec.transform == "abs":
                    val = np.abs(y_ch)
                    label_suffix = "|.|"
                elif spec.transform == "angle":
                    val = np.angle(y_ch)
                    label_suffix = "Arg"
                elif spec.transform == "number":
                    val = np.abs(y_ch) ** 2
                    label_suffix = "n"
                else:
                    val = np.real(y_ch)
                    label_suffix = ""

                # Trajectory selection
                if spec.trajectories == "mean":
                    mean_val = np.mean(val, axis=0)
                    std_val = np.std(val, axis=0)
                    ax.plot(t, mean_val, label=f"Ch{ch} {label_suffix}")
                    ax.fill_between(
                        t, mean_val - std_val, mean_val + std_val, alpha=0.2
                    )
                elif spec.trajectories == "all":
                    for i in range(val.shape[0]):
                        ax.plot(t, val[i], alpha=0.3, linewidth=0.5)
                elif isinstance(spec.trajectories, int):
                    ax.plot(
                        t, val[spec.trajectories], label=f"Ch{ch} {label_suffix}"
                    )

            apply_axes_style(
                ax,
                title=spec.title,
                xlabel=spec.xlabel or "Time",
                ylabel=spec.ylabel,
                xlim=spec.xlim,
                ylim=spec.ylim,
                grid=spec.grid,
                legend=spec.legend,
            )
            draw_annotations(ax, spec.annotations)
            return save_figure(
                fig,
                output_dir,
                spec.filename or f"time_series_{spec.transform}",
                format,
            )
