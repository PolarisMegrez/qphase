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

    def _extract_parameter(self, job_name: str, param_name: str) -> float | None:
        """Extract parameter value from job name string."""
        # Pattern: name[p1=v1, p2=v2]
        # Regex to find "param_name=value"
        # Value can be number, scientific notation
        pattern = f"{param_name}=([^,\\]]+)"
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
        # Get trajectory data
        traj = result.data

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
                min_height=None,
                prominence=None,
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
            param_val = self._extract_parameter(job_name, spec.parameter)
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
