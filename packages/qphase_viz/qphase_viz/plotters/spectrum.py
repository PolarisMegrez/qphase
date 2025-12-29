"""qphase_viz: Spectrum Plotters
--------------------------

Plotters for spectral analysis (PSD).
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .base import PlotterProtocol


class PowerSpectrumPlotter(PlotterProtocol):
    """Plots Power Spectral Density (PSD)."""

    def plot(
        self, data: Any, config: dict[str, Any], output_dir: Path, format: str
    ) -> Path:
        fig, ax = plt.subplots(figsize=config["figsize"], dpi=config["dpi"])
        channels = config["channels"]
        scale = config["scale"]

        # Check if data is pre-computed PSD (dict)
        if isinstance(data, dict) and "psd" in data and "axis" in data:
            # Pre-computed PSD
            # data['psd'] shape: (n_freq, n_modes)
            # data['axis'] shape: (n_freq,)
            f = data["axis"]
            Pxx_all = data["psd"]

            # Map channels to columns in Pxx_all
            # Assuming channels correspond to indices in Pxx_all
            # But wait, Pxx_all has n_modes columns.
            # If the analysis was run for specific modes, we need to know which
            # column corresponds to which mode. The analysis result should
            # probably include 'modes' list.

            available_modes = data.get("modes", list(range(Pxx_all.shape[1])))

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

                    ax.plot(f, val, label=f"Ch{ch}")
                else:
                    print(f"Warning: Channel {ch} not found in pre-computed PSD data.")

        else:
            # Expecting TrajectorySet: (n_traj, n_steps, n_modes)
            if hasattr(data, "to_numpy"):
                y = data.to_numpy()  # (N, T, M)
            else:
                y = np.array(data)

            # Determine sampling frequency
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

            fs = 1.0 / dt

            window = config["window"] or "hann"
            nperseg = config["nperseg"]
            detrend = "constant" if config["detrend"] else False

            for ch in channels:
                # Compute PSD for each trajectory and average
                # y[:, :, ch] is (N, T)

                f, Pxx = signal.welch(
                    y[:, :, ch],
                    fs=fs,
                    window=window,
                    nperseg=nperseg,
                    detrend=detrend,
                    axis=1,  # Time axis is 1
                    scaling="density",
                )

                # Pxx is (N, F)
                # Average over trajectories
                Pxx_mean = np.mean(Pxx, axis=0)

                if scale == "dB":
                    val = 10 * np.log10(Pxx_mean + 1e-20)
                    ylabel = "PSD [dB/Hz]"
                elif scale == "log":
                    val = Pxx_mean
                    ax.set_yscale("log")
                    ylabel = "PSD [V**2/Hz]"
                else:
                    val = Pxx_mean
                    ylabel = "PSD [V**2/Hz]"

                ax.plot(f, val, label=f"Ch{ch}")

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
        if config["legend"]:
            ax.legend()

        # Save
        filename = config["filename"] or "power_spectrum"
        out_path = output_dir / f"{filename}.{format}"
        fig.savefig(out_path, format=format, bbox_inches="tight")
        plt.close(fig)

        return out_path
