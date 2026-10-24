"""qphase_viz: Parameter Evolution Plotter
---------------------------------------------------------
Plots metrics (e.g. PSD peaks) against scanning parameters.
"""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from matplotlib.axes import Axes
from qphase.core.dataset import DatasetResultProtocol
from qphase.data import DataKind

from ..config import ParameterEvolutionConfig, ParameterEvolutionSpec
from ..render import apply_axes_style, draw_annotations, save_figure, styled_figure
from ..styles import RenderOptions
from ..typed_input import scan_values, select_product, trajectory_array
from .base import PlotterProtocol


class ParameterEvolutionPlotter(PlotterProtocol):
    """Plots metrics vs parameters from aggregated results."""

    name: ClassVar[str] = "parameter_evolution"
    description: ClassVar[str] = "Parameter Evolution Plotter"
    config_schema: ClassVar[type[ParameterEvolutionConfig]] = ParameterEvolutionConfig

    def __init__(
        self, config: ParameterEvolutionConfig | None = None, **kwargs: Any
    ) -> None:
        if config is None:
            config = ParameterEvolutionConfig(**kwargs)
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

    def _metric_points(
        self, data: DatasetResultProtocol, spec: ParameterEvolutionSpec
    ) -> tuple[np.ndarray, np.ndarray]:
        if spec.metric.startswith("psd_peak_"):
            product = select_product(
                data,
                kind=DataKind.STATISTICS,
                quantity="peak_location",
                fields=(
                    "group_scan_index",
                    "group_channel_index",
                    "candidate_offsets",
                    "location",
                    "intensity",
                ),
            )
            x = scan_values(data, product, spec.parameter)
            group_scan = np.asarray(product.handle("group_scan_index").materialize())
            group_channel = np.asarray(
                product.handle("group_channel_index").materialize()
            )
            offsets = np.asarray(product.handle("candidate_offsets").materialize())
            locations = np.asarray(product.handle("location").materialize())
            intensities = np.asarray(product.handle("intensity").materialize())
            y = np.full(x.shape, np.nan, dtype=float)
            for group, (scan, channel) in enumerate(
                zip(group_scan, group_channel, strict=True)
            ):
                if int(channel) != spec.channel:
                    continue
                start, stop = int(offsets[group]), int(offsets[group + 1])
                if stop <= start:
                    continue
                selected = start + int(np.argmax(intensities[start:stop]))
                y[int(scan)] = (
                    locations[selected]
                    if spec.metric == "psd_peak_freq"
                    else intensities[selected]
                )
            return x, y

        product = select_product(
            data,
            role="trajectories",
            kind=DataKind.TIME_SERIES,
            quantity="field_amplitude",
            fields=("alpha",),
        )
        values = np.abs(trajectory_array(product)[..., spec.channel])
        x = scan_values(data, product, spec.parameter)
        reducer = np.mean if spec.metric == "mean" else np.var
        return x, reducer(values, axis=(1, 2))

    def _plot_single(
        self,
        data: DatasetResultProtocol,
        spec: ParameterEvolutionSpec,
        output_dir: Path,
        format: str,
        base: RenderOptions,
    ) -> Path:
        x_values, y_values = self._metric_points(data, spec)
        order = np.argsort(x_values)
        x = x_values[order].tolist()
        y = y_values[order].tolist()

        options = base.merged(
            style=spec.style,
            editable=spec.editable,
            overrides=spec.style_overrides,
        )
        with styled_figure(options, figsize=spec.figsize, dpi=spec.dpi) as (fig, ax):
            self._draw_metric(ax, x, y, spec)
            apply_axes_style(
                ax,
                title=spec.title or f"{spec.metric} vs {spec.parameter}",
                xlabel=spec.xlabel or spec.parameter,
                ylabel=spec.ylabel or spec.metric,
                xlim=spec.xlim,
                ylim=spec.ylim,
                grid=spec.grid,
                legend=True,
            )
            draw_annotations(ax, spec.annotations)
            return save_figure(
                fig,
                output_dir,
                spec.filename or f"evol_{spec.parameter}_{spec.metric}",
                format,
            )

    def _draw_metric(
        self,
        ax: Axes,
        x: list[float],
        y: list[float],
        spec: ParameterEvolutionSpec,
    ) -> None:
        """Draw the metric series and its optional fit on *ax*."""
        ax.plot(x, y, "o-", label=f"Ch{spec.channel}")

        # Curve Fitting
        if spec.fit_type:
            x_arr = np.array(x)
            y_arr = np.array(y)

            # Filter NaNs and Infs
            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            x_fit = x_arr[mask]
            y_fit = y_arr[mask]

            if len(x_fit) > 1:
                if spec.fit_type == "linear":
                    # y = ax + b
                    coeffs = np.polyfit(x_fit, y_fit, 1)
                    a, b = coeffs
                    # Plot fit across range
                    x_line = np.linspace(min(x_fit), max(x_fit), 100)
                    y_line = a * x_line + b
                    ax.plot(x_line, y_line, "r--", label=f"Fit: $y={a:.2g}x + {b:.2g}$")

                elif spec.fit_type == "power":
                    # y = a * |x|^b => log(y) = log(a) + b*log(|x|)
                    # Use absolute values as requested
                    x_abs = np.abs(x_fit)
                    y_abs = np.abs(y_fit)

                    # Filter out zeros so the log fit stays well-defined.
                    # Keep only strictly positive values.
                    pos_mask = (x_abs > 1e-20) & (y_abs > 1e-20)

                    if np.sum(pos_mask) > 1:
                        x_use = x_abs[pos_mask]
                        y_use = y_abs[pos_mask]

                        log_x = np.log(x_use)
                        log_y = np.log(y_use)

                        coeffs = np.polyfit(log_x, log_y, 1)
                        b, log_a = coeffs
                        a = np.exp(log_a)

                        # Determine the sign of the fit from residuals.
                        # Compare positive, negative, odd-aligned, odd-anti-aligned.
                        mag_pred = a * np.abs(x_fit) ** b

                        def positive(x: np.ndarray) -> np.ndarray:
                            return np.ones_like(x)

                        def negative(x: np.ndarray) -> np.ndarray:
                            return -1 * np.ones_like(x)

                        def odd(x: np.ndarray) -> np.ndarray:
                            return np.sign(x)

                        def odd_negative(x: np.ndarray) -> np.ndarray:
                            return -1 * np.sign(x)

                        candidates = [
                            (positive, ""),
                            (negative, "-"),
                            (odd, r"\text{sgn}(x)"),
                            (odd_negative, r"-\text{sgn}(x)"),
                        ]

                        best_mse = float("inf")
                        best_func = positive
                        best_prefix = ""

                        for func, prefix in candidates:
                            pred = func(x_fit) * mag_pred
                            mse = np.mean((y_fit - pred) ** 2)
                            if mse < best_mse:
                                best_mse = mse
                                best_func = func
                                best_prefix = prefix

                        # Generate smooth line
                        x_line = np.linspace(min(x_fit), max(x_fit), 100)
                        y_line = best_func(x_line) * a * np.abs(x_line) ** b
                        ax.plot(
                            x_line,
                            y_line,
                            "r--",
                            label=f"Fit: $y={best_prefix}{a:.2f}|x|^{{{b:.2f}}}$",
                        )
                    else:
                        print(
                            f"Warning: Power fit requires positive values (after abs). "
                            f"Found {np.sum(pos_mask)} valid points."
                        )
