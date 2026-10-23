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
from qphase.core.dataset import DatasetResultProtocol
from qphase.data import DataKind, SpectralQuantity

from ..config import PowerSpectrumConfig, PowerSpectrumSpec
from ..typed_input import (
    ProductNotFoundError,
    require_single_scan_point,
    select_product,
)
from .base import PlotterProtocol


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

    def plot(
        self, data: DatasetResultProtocol, output_dir: Path, format: str
    ) -> list[Path]:
        generated_files = []
        for spec in self.config.plots:
            generated_files.append(self._plot_single(data, spec, output_dir, format))
        return generated_files

    def _plot_single(
        self,
        data: DatasetResultProtocol,
        spec: PowerSpectrumSpec,
        output_dir: Path,
        format: str,
    ) -> Path:
        config = spec.model_dump()
        fig, ax = plt.subplots(figsize=config["figsize"], dpi=config["dpi"])
        channels = config["channels"]
        scale = config["scale"]

        try:
            product = select_product(
                data,
                role="primary_spectrum",
                kind=DataKind.SPECTRAL,
                quantity=SpectralQuantity.POWER_SPECTRAL_DENSITY.value,
                fields=("axis", "psd"),
            )
        except ProductNotFoundError as exc:
            raise ProductNotFoundError(
                "power spectrum plotting needs a persisted PSD product "
                "(role 'primary_spectrum', kind 'spectral', quantity "
                "'power_spectral_density', variables 'axis' and 'psd'); run a PSD "
                "analysis first — e.g. the workflow's psd analyser — and pass its "
                "typed output to the viz engine"
            ) from exc

        require_single_scan_point(data, product)

        f = np.asarray(product.handle("axis").materialize())
        spectra = np.asarray(product.handle("psd").materialize())
        if f.ndim == 2:
            if not np.allclose(f, f[0]):
                raise ValueError("PSD scan points use different frequency axes")
            f = f[0]
        if spectra.ndim == 3:
            spectra = np.mean(spectra, axis=0)
        if spectra.ndim == 1:
            spectra = spectra[:, None]
        Pxx_all = spectra
        variables = {variable.name for variable in product.variables}
        available_modes = (
            [int(mode) for mode in product.handle("modes").materialize()]
            if "modes" in variables
            else list(range(Pxx_all.shape[1]))
        )
        peak_product = None
        if spec.annotate_peaks:
            peak_product = select_product(
                data,
                kind=DataKind.STATISTICS,
                quantity="peak_location",
                fields=(
                    "group_channel_index",
                    "candidate_offsets",
                    "location",
                    "intensity",
                ),
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
                if peak_product is not None:
                    group_channels = np.asarray(
                        peak_product.handle("group_channel_index").materialize()
                    )
                    offsets = np.asarray(
                        peak_product.handle("candidate_offsets").materialize()
                    )
                    locations = np.asarray(
                        peak_product.handle("location").materialize()
                    )
                    intensities = np.asarray(
                        peak_product.handle("intensity").materialize()
                    )
                    indices: list[int] = []
                    for group, channel in enumerate(group_channels):
                        if int(channel) == idx:
                            indices.extend(
                                range(int(offsets[group]), int(offsets[group + 1]))
                            )
                    if indices:
                        selected = np.array(indices, dtype=np.int64)
                        selected = selected[
                            np.argsort(intensities[selected])[-3:]
                        ]
                        peak_values = intensities[selected]
                        if scale == "dB":
                            peak_values = 10 * np.log10(peak_values + 1e-20)
                        ax.plot(
                            locations[selected],
                            peak_values,
                            "x",
                            color=line.get_color(),
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
