"""qphase_viz: Phase Plane Plotters
---------------------------------------------------------
Plotters for phase space correlations.

Public API
----------
``PhasePlanePlotter`` : Plots phase plane data (Im vs Re or Ch_j vs Ch_i).
"""

from pathlib import Path
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
from qphase.backend.base import ArrayBase

from ..config import PhasePlaneConfig, PhasePlaneSpec
from .base import PlotterProtocol


class PhasePlanePlotter(PlotterProtocol):
    """Plots phase plane data (Im vs Re or Ch_j vs Ch_i).

    Attributes
    ----------
    name : ClassVar[str]
        The name of the plotter ("phase_plane").
    description : ClassVar[str]
        A description of the plotter.
    config_schema : ClassVar[type[PhasePlaneConfig]]
        The configuration schema for this plotter.

    """

    name: ClassVar[str] = "phase_plane"
    description: ClassVar[str] = "Phase Plane Plotter"
    config_schema: ClassVar[type[PhasePlaneConfig]] = PhasePlaneConfig

    def __init__(self, config: PhasePlaneConfig | None = None, **kwargs: Any) -> None:
        """Initialize the PhasePlanePlotter.

        Parameters
        ----------
        config : PhasePlaneConfig | None, optional
            The configuration object.
        **kwargs : Any
            Keyword arguments to create a configuration if none is provided.

        """
        if config is None:
            config = PhasePlaneConfig(**kwargs)
        self.config = config

    def plot(self, data: ArrayBase, output_dir: Path, format: str) -> list[Path]:
        """Generate phase plane plots from the provided data.

        Parameters
        ----------
        data : ArrayBase
            The simulation data.
        output_dir : Path
            The directory to save the plots to.
        format : str
            The file format for the plots (e.g., "png", "pdf").

        Returns
        -------
        list[Path]
            A list of paths to the generated plot files.

        """
        generated_files = []
        for spec in self.config.plots:
            generated_files.append(self._plot_single(data, spec, output_dir, format))
        return generated_files

    def _plot_single(
        self, data: ArrayBase, spec: PhasePlaneSpec, output_dir: Path, format: str
    ) -> Path:
        """Generate a single phase plane plot.

        Parameters
        ----------
        data : ArrayBase or dict
            The simulation data or analysis result.
        spec : PhasePlaneSpec
            The specification for the single plot.
        output_dir : Path
            The directory to save the plot to.
        format : str
            The file format for the plot.

        Returns
        -------
        Path
            The path to the generated plot file.

        """
        config = spec.model_dump()

        # Check for pre-computed distribution (Analysis Result)
        if isinstance(data, dict):
            # Unwrap if wrapped in analyzer name (e.g. "dist")
            tgt = None
            if "distributions" in data:
                tgt = data
            elif (
                "dist" in data
                and isinstance(data["dist"], dict)
                and "distributions" in data["dist"]
            ):
                tgt = data["dist"]

            if tgt:
                distributions = tgt["distributions"]
                ch_x = config["channel_x"]
                ch_y = config["channel_y"]

            # Map channel to the analyzer's mode key
            # Analyzer keyed by mode index (int)
            dist_data = distributions.get(ch_x)
            if dist_data is None:
                # Try string key if JSON serialization happened
                dist_data = distributions.get(str(ch_x))

            if dist_data:
                fig, ax = plt.subplots(figsize=config["figsize"], dpi=config["dpi"])

                if dist_data["type"] == "2d_complex":
                    H = dist_data["hist"]
                    xedges = dist_data["xedges"]
                    yedges = dist_data["yedges"]

                    # Plot using imshow (better for pre-binned data than pcolormesh)
                    # Note: numpy histogram2d returns H of shape (nx, ny).
                    # x corresponds to first dim, y to second.
                    # imshow expects (rows, cols) -> (y, x). So transpose H.

                    im = ax.imshow(
                        H.T,
                        origin="lower",
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                        aspect="auto",
                        cmap=config["cmap"],
                    )
                    fig.colorbar(im, ax=ax, label="Density")

                    # Styling
                    if config["title"]:
                        ax.set_title(config["title"])
                    else:
                        ax.set_title(f"Phase Space Density (Mode {ch_x})")

                    if config["xlabel"]:
                        ax.set_xlabel(config["xlabel"])
                    else:
                        ax.set_xlabel("Re")
                    if config["ylabel"]:
                        ax.set_ylabel(config["ylabel"])
                    else:
                        ax.set_ylabel("Im")

                elif dist_data["type"] == "1d_real":
                    # Handle 1D case if necessary (PhasePlane is usually 2D)
                    # Maybe bar plot?
                    pass

                # Save
                filename = config["filename"] or f"phase_plane_dist_{ch_x}"
                out_path = output_dir / f"{filename}.{format}"
                fig.savefig(out_path, format=format, bbox_inches="tight")
                plt.close(fig)
                return out_path

        # Existing Trajectory Logic
        y = data.to_numpy()

        ch_x = config["channel_x"]
        ch_y = config["channel_y"]

        # Flatten trajectories for phase plane statistics
        # (N*T, M)
        y_flat = y.reshape(-1, y.shape[-1])

        if ch_y is None:
            # Re vs Im of single channel
            x_data = np.real(y_flat[:, ch_x])
            y_data = np.imag(y_flat[:, ch_x])
            xlabel = f"Re(Ch{ch_x})"
            ylabel = f"Im(Ch{ch_x})"
        else:
            # Ch_y vs Ch_x (Real parts usually, or Abs? Let's assume Real for
            # now or just raw values if real)
            # If complex, phase plane usually implies Re/Im.
            # If comparing two modes, maybe |a|^2 vs |b|^2 or Re(a) vs Re(b)?
            # Let's assume we plot Real parts if complex, or values if real.
            # Or maybe we should stick to the standard "Phase Space" definition.
            # Let's use Real parts for general correlation.
            x_data = np.real(y_flat[:, ch_x])
            y_data = np.real(y_flat[:, ch_y])
            xlabel = f"Re(Ch{ch_x})"
            ylabel = f"Re(Ch{ch_y})"

        fig, ax = plt.subplots(figsize=config["figsize"], dpi=config["dpi"])

        mode = config["mode"]

        if mode == "scatter":
            # Downsample if too many points for scatter
            max_points = 10000
            if len(x_data) > max_points:
                idx = np.random.choice(len(x_data), max_points, replace=False)
                x_plot = x_data[idx]
                y_plot = y_data[idx]
            else:
                x_plot = x_data
                y_plot = y_data

            ax.scatter(x_plot, y_plot, alpha=0.1, s=1, c="k")

        elif mode == "hist2d":
            h = ax.hist2d(
                x_data, y_data, bins=config["bins"], cmap=config["cmap"], density=True
            )
            fig.colorbar(h[3], ax=ax, label="Probability Density")

        elif mode == "kde":
            # Simple Gaussian KDE approximation or contour
            # For now, fallback to hist2d as KDE is expensive on large datasets
            # without scipy optimization or implement a simple contour over hist2d
            counts, xedges, yedges = np.histogram2d(
                x_data, y_data, bins=config["bins"], density=True
            )
            x_centers = (xedges[:-1] + xedges[1:]) / 2
            y_centers = (yedges[:-1] + yedges[1:]) / 2
            X, Y = np.meshgrid(x_centers, y_centers)

            # Smooth slightly?

            c = ax.contourf(X, Y, counts.T, cmap=config["cmap"], levels=20)
            fig.colorbar(c, ax=ax, label="Density")

        # Styling
        if config["title"]:
            ax.set_title(config["title"])
        if config["xlabel"]:
            ax.set_xlabel(config["xlabel"])
        else:
            ax.set_xlabel(xlabel)
        if config["ylabel"]:
            ax.set_ylabel(config["ylabel"])
        else:
            ax.set_ylabel(ylabel)
        if config["xlim"]:
            ax.set_xlim(config["xlim"])
        if config["ylim"]:
            ax.set_ylim(config["ylim"])
        if config["grid"]:
            ax.grid(True, alpha=0.3)

        # Save
        filename = config["filename"] or f"phase_plane_{mode}"
        out_path = output_dir / f"{filename}.{format}"
        fig.savefig(out_path, format=format, bbox_inches="tight")
        plt.close(fig)

        return out_path
