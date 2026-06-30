"""qphase_viz: Parameter Evolution Plotter
---------------------------------------------------------
Plots metrics (e.g. PSD peaks) against scanning parameters.
"""

import re
from pathlib import Path
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.protocols import ResultProtocol
from qphase_sde.analyser import PsdAnalyzer, PsdAnalyzerConfig

from ..config import ParameterEvolutionConfig, ParameterEvolutionSpec
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

    def plot(self, data: Any, output_dir: Path, format: str) -> list[Path]:
        # data should be a dict of results (AggregateResult.data)
        if not isinstance(data, dict):
            # If not aggregated, we can't plot evolution
            return []

        generated_files = []
        for spec in self.config.plots:
            generated_files.append(self._plot_single(data, spec, output_dir, format))
        return generated_files

    def _extract_parameter(
        self, job_name: str, param_name: str, result: ResultProtocol | None = None
    ) -> float | None:
        """Extract parameter value from result metadata or job name string."""
        # Try metadata first
        if result:
            meta = getattr(result, "metadata", {})
            # Check 'params' dict
            params = meta.get("params", {})
            if param_name in params:
                try:
                    return float(params[param_name])
                except (ValueError, TypeError):
                    pass

            # Also check implicit filters used during aggregation.
            # Usually the iterator parameter is the grouped job value.
            pass

        # Fallback to Regex on job name
        # Pattern: name[p1=v1, p2=v2]
        # Regex to find "param_name=value"
        # Value can be number, scientific notation
        pattern = f"{param_name}=([^,\\]_]+)"
        match = re.search(pattern, job_name)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _compute_metric(
        self, result: ResultProtocol, spec: ParameterEvolutionSpec
    ) -> float:
        """Compute the requested metric for a single result."""
        # Get data
        data = result.data

        # Case 1: Pre-computed Analysis (Dictionary)
        if isinstance(data, dict):
            if spec.metric.startswith("psd_"):
                # Expecting data['peaks']
                peaks_info = data.get("peaks", {}).get(spec.channel, {})
                if not peaks_info:
                    # Maybe nested in 'psd'
                    peaks_info = (
                        data.get("psd", {}).get("peaks", {}).get(spec.channel, {})
                    )

                p_vals = peaks_info.get("values", [])
                p_freqs = peaks_info.get("frequencies", [])

                if len(p_vals) == 0:
                    return np.nan

                # Find dominant peak
                idx = np.argmax(p_vals)

                if spec.metric == "psd_peak_freq":
                    return p_freqs[idx]
                elif spec.metric == "psd_peak_val":
                    return p_vals[idx]

            # If not PSD metric, maybe mean/var?
            # Analysis dict might not have mean/var unless calculated.
            return np.nan

        # Case 2: Trajectory object
        traj = data
        if spec.metric.startswith("psd_"):
            # We need PSD
            # Configure analyzer
            analyzer_config = PsdAnalyzerConfig(
                kind="complex",
                modes=[spec.channel],
                convention="symmetric",
                dt=getattr(traj, "dt", 1.0),
                window=spec.psd_window,
                find_peaks=True,  # Always find peaks for metric extraction
            )
            analyzer = PsdAnalyzer(analyzer_config)
            res = analyzer.analyze(traj, backend=NumpyBackend())

            peaks_info = res.data.get("peaks", {}).get(spec.channel, {})
            p_vals = peaks_info.get("values", [])
            p_freqs = peaks_info.get("frequencies", [])

            if len(p_vals) == 0:
                return np.nan

            # Find dominant peak
            idx = np.argmax(p_vals)

            if spec.metric == "psd_peak_freq":
                return p_freqs[idx]
            elif spec.metric == "psd_peak_val":
                return p_vals[idx]

        elif spec.metric == "mean":
            # Mean of trajectory (abs)
            # traj.data shape (n_traj, n_time, n_modes)
            data_arr = traj.data
            if hasattr(data_arr, "ndim") and data_arr.ndim == 3:
                return float(np.mean(np.abs(data_arr[:, :, spec.channel])))

        elif spec.metric == "variance":
            data_arr = traj.data
            if hasattr(data_arr, "ndim") and data_arr.ndim == 3:
                return float(np.var(np.abs(data_arr[:, :, spec.channel])))

        return np.nan

    def _plot_single(
        self,
        data: dict[str, ResultProtocol],
        spec: ParameterEvolutionSpec,
        output_dir: Path,
        format: str,
    ) -> Path:
        # Extract (param, metric) pairs
        points = []
        for job_name, result in data.items():
            param_val = self._extract_parameter(job_name, spec.parameter, result)
            if param_val is not None:
                metric_val = self._compute_metric(result, spec)
                points.append((param_val, metric_val))

        # Sort by parameter
        points.sort(key=lambda x: x[0])

        x = [p[0] for p in points]
        y = [p[1] for p in points]

        # Plot
        config = spec.model_dump()
        fig, ax = plt.subplots(figsize=config["figsize"], dpi=config["dpi"])

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

        ax.legend()
        ax.set_xlabel(spec.xlabel or spec.parameter)
        ax.set_ylabel(spec.ylabel or spec.metric)
        if spec.title:
            ax.set_title(spec.title)
        else:
            ax.set_title(f"{spec.metric} vs {spec.parameter}")

        if spec.xlim:
            ax.set_xlim(spec.xlim)
        if spec.ylim:
            ax.set_ylim(spec.ylim)
        if spec.grid:
            ax.grid(True, alpha=0.3)

        # Save
        fname = spec.filename or f"evol_{spec.parameter}_{spec.metric}"
        out_path = output_dir / f"{fname}.{format}"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        return out_path
