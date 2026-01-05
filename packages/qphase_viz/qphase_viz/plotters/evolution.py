"""qphase_viz: Evolution Plotters
---------------------------------------------------------
Plotters for time-evolution data.

Public API
----------
``TimeSeriesPlotter``
"""

from pathlib import Path
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
from qphase.backend.base import ArrayBase
from qphase.backend.xputil import convert_to_numpy

from ..config import TimeSeriesConfig, TimeSeriesSpec
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

    def plot(self, data: ArrayBase, output_dir: Path, format: str) -> list[Path]:
        generated_files = []
        for spec in self.config.plots:
            generated_files.append(self._plot_single(data, spec, output_dir, format))
        return generated_files

    def _plot_single(
        self, data: ArrayBase, spec: TimeSeriesSpec, output_dir: Path, format: str
    ) -> Path:
        config = spec.model_dump()
        # Extract data
        # Expecting TrajectorySet: (n_traj, n_steps, n_modes)
        # or State: (n_traj, n_modes) - but State has no time axis usually?
        # Assume TrajectorySet-like structure

        y = data.to_numpy()  # (N, T, M)
        if hasattr(data, "times"):
            t = convert_to_numpy(data.times)
        else:
            t = np.arange(y.shape[1])

        channels = config["channels"]
        transform = config["transform"]
        traj_sel = config["trajectories"]

        fig, ax = plt.subplots(figsize=config["figsize"], dpi=config["dpi"])

        # Data transformation
        for ch in channels:
            y_ch = y[:, :, ch]  # (N, T)

            if transform == "real":
                val = np.real(y_ch)
                label_suffix = "Re"
            elif transform == "imag":
                val = np.imag(y_ch)
                label_suffix = "Im"
            elif transform == "abs":
                val = np.abs(y_ch)
                label_suffix = "|.|"
            elif transform == "angle":
                val = np.angle(y_ch)
                label_suffix = "Arg"
            elif transform == "number":
                val = np.abs(y_ch) ** 2
                label_suffix = "n"
            else:
                val = np.real(y_ch)
                label_suffix = ""

            # Trajectory selection
            if traj_sel == "mean":
                mean_val = np.mean(val, axis=0)
                std_val = np.std(val, axis=0)
                ax.plot(t, mean_val, label=f"Ch{ch} {label_suffix}")
                ax.fill_between(t, mean_val - std_val, mean_val + std_val, alpha=0.2)
            elif traj_sel == "all":
                for i in range(val.shape[0]):
                    ax.plot(t, val[i], alpha=0.3, linewidth=0.5)
            elif isinstance(traj_sel, int):
                ax.plot(t, val[traj_sel], label=f"Ch{ch} {label_suffix}")

        # Styling
        if config["title"]:
            ax.set_title(config["title"])
        if config["xlabel"]:
            ax.set_xlabel(config["xlabel"])
        else:
            ax.set_xlabel("Time")
        if config["ylabel"]:
            ax.set_ylabel(config["ylabel"])
        if config["xlim"]:
            ax.set_xlim(config["xlim"])
        if config["ylim"]:
            ax.set_ylim(config["ylim"])
        if config["grid"]:
            ax.grid(True, alpha=0.3)
        if config["legend"]:
            ax.legend()

        # Save
        filename = config["filename"] or f"time_series_{transform}"
        out_path = output_dir / f"{filename}.{format}"
        fig.savefig(out_path, format=format, bbox_inches="tight")
        plt.close(fig)

        return out_path
