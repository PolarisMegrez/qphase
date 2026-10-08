"""qphase_viz: Spectrum Plotters
---------------------------------------------------------
Plotters for spectral analysis (PSD).

Public API
----------
`PowerSpectrumPlotter` : Plots Power Spectral Density (PSD).
"""

from pathlib import Path
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
from qphase.backend.base import ArrayBase
from qphase.backend.numpy_backend import NumpyBackend

from ..config import PowerSpectrumConfig, PowerSpectrumSpec
from .base import PlotterProtocol


def _get_psd_analyzer_classes():
    """Lazy import of PsdAnalyzer to keep qphase_sde optional."""
    try:
        from qphase_sde.analyser import PsdAnalyzer, PsdAnalyzerConfig
        from qphase_sde.peak_finder import ScipyPeakFinder
        from qphase_sde.result import _trajectory_product
        from qphase_sde.spectral_estimator import create_builtin_estimator
    except ImportError as e:
        raise ImportError(
            "Power spectrum plotting from raw trajectories requires "
            "qphase-viz[sde] (qphase-sde)."
        ) from e
    return (
        PsdAnalyzer,
        PsdAnalyzerConfig,
        ScipyPeakFinder,
        _trajectory_product,
        create_builtin_estimator,
    )


class PowerSpectrumPlotter(PlotterProtocol):
    """Plots Power Spectral Density (PSD)."""

    name: ClassVar[str] = "power_spectrum"
    description: ClassVar[str] = "Power Spectrum Plotter"
    config_schema: ClassVar[type[PowerSpectrumConfig]] = PowerSpectrumConfig

    def __init__(
        self, config: PowerSpectrumConfig | None = None, **kwargs: Any
    ) -> None:
        if config is None:
            config = PowerSpectrumConfig(**kwargs)
        self.config = config

    def plot(self, data: ArrayBase, output_dir: Path, format: str) -> list[Path]:
        generated_files = []
        for spec in self.config.plots:
            generated_files.append(self._plot_single(data, spec, output_dir, format))
        return generated_files

    def _plot_single(
        self, data: ArrayBase, spec: PowerSpectrumSpec, output_dir: Path, format: str
    ) -> Path:
        config = spec.model_dump()
        fig, ax = plt.subplots(figsize=config["figsize"], dpi=config["dpi"])
        channels = config["channels"]
        scale = config["scale"]

        # Check if data is pre-computed PSD (dict)
        # Handle nesting in analyzer key (e.g. data['psd']['psd'])
        psd_data = None
        if isinstance(data, dict):
            if "psd" in data and "axis" in data:
                psd_data = data
            elif (
                "psd" in data and isinstance(data["psd"], dict) and "psd" in data["psd"]
            ):
                psd_data = data["psd"]

        if psd_data:
            # Pre-computed PSD
            f = psd_data["axis"]
            Pxx_all = psd_data["psd"]
            available_modes = psd_data.get("modes", list(range(Pxx_all.shape[1])))
            # Extract peaks if available
            peaks_info = psd_data.get("peaks", {})
        else:
            # Compute using PsdAnalyzer
            (
                PsdAnalyzer,
                PsdAnalyzerConfig,
                ScipyPeakFinder,
                trajectory_product,
                create_estimator,
            ) = _get_psd_analyzer_classes()
            if hasattr(data, "dt"):
                dt = data.dt
            elif hasattr(data, "times"):
                t = data.times
                if len(t) > 1:
                    dt = t[1] - t[0]
                else:
                    dt = 1.0
            else:
                dt = 1.0

            # Configure analyzer
            # We assume complex signal by default for generality
            # Note: peak finding parameters rely on PsdAnalyzer defaults
            # now to ensure consistency
            analyzer_config = PsdAnalyzerConfig(
                kind="complex",
                modes=channels,
                convention="symmetric",
                dt=dt,
            )
            analyzer = PsdAnalyzer(
                analyzer_config,
                subplugins={
                    "estimator": create_estimator(
                        "periodogram", {"window": spec.window}
                    )
                },
            )

            trajectory = trajectory_product(
                data, scan_size=None, n_traj_per_point=None
            ).point_view(scan=0)
            product = analyzer.execute(
                {"trajectories": trajectory},
                backend=NumpyBackend(),
                label="psd",
            )["psd"]
            f = product.handle("axis").materialize()
            Pxx_all = product.handle("psd").materialize()
            available_modes = [
                int(mode) for mode in product.handle("modes").materialize()
            ]
            finder = ScipyPeakFinder()
            peaks_info = (
                {
                    mode: finder.find_peaks(f, Pxx_all[:, index]).model_dump()
                    for index, mode in enumerate(available_modes)
                }
                if spec.annotate_peaks
                else {}
            )

        # Plotting
        for ch in channels:
            if ch in available_modes:
                idx = available_modes.index(ch)
                val = Pxx_all[:, idx]

                if scale == "dB":
                    val = 10 * np.log10(val + 1e-20)
                    ylabel = "PSD [dB/Hz]"
                elif scale == "log":
                    ax.set_yscale("log")
                    ylabel = "PSD [V**2/Hz]"
                else:
                    ylabel = "PSD [V**2/Hz]"

                (line,) = ax.plot(f, val, label=f"Ch{ch}")

                # Annotate peaks
                if spec.annotate_peaks and ch in peaks_info:
                    p_freqs = peaks_info[ch]["frequencies"]
                    p_vals = peaks_info[ch]["values"]

                    # If scale is dB, we need to transform peak values too for plotting
                    if scale == "dB":
                        p_vals_plot = 10 * np.log10(p_vals + 1e-20)
                    else:
                        p_vals_plot = p_vals

                    ax.plot(p_freqs, p_vals_plot, "x", color=line.get_color())

                    # Add text labels for top 3 peaks
                    # Sort by value
                    sorted_idx = np.argsort(p_vals)[::-1]
                    for i in sorted_idx[:3]:
                        ax.annotate(
                            f"{p_freqs[i]:.2f}",
                            xy=(p_freqs[i], p_vals_plot[i]),
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha="center",
                            fontsize=8,
                        )
            else:
                print(f"Warning: Channel {ch} not found in PSD data.")

        # Styling
        if config["title"]:
            ax.set_title(config["title"])
        if config["xlabel"]:
            ax.set_xlabel(config["xlabel"])
        else:
            ax.set_xlabel("Frequency [Hz]")
        if config["ylabel"]:
            ax.set_ylabel(config["ylabel"])
        else:
            ax.set_ylabel(ylabel)
        if config["xlim"]:
            ax.set_xlim(config["xlim"])
        if config["ylim"]:
            ax.set_ylim(config["ylim"])
        if config["grid"]:
            ax.grid(True, alpha=0.3, which="both")
        if config.get("legend", True):
            ax.legend()

        # Save
        filename = config["filename"] or "power_spectrum"
        out_path = output_dir / f"{filename}.{format}"
        fig.savefig(out_path, format=format, bbox_inches="tight")
        plt.close(fig)

        return out_path
