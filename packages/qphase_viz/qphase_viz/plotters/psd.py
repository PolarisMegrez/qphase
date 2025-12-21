"""qphase_sde: PSD Renderer
-----------------------
Render power spectral density (PSD) for one or more modes from multi-trajectory
time series using analysis utilities and plot onto Matplotlib axes.

Behavior
--------
- Plot pre-calculated PSD data; choose plot scales (linear/log)
    and sidedness based on spec and style. Details are documented by functions.
"""

__all__ = [
    "PsdPlotter",
    "PsdConfig",
    "PsdSpec",
]

from typing import Any, ClassVar, Literal

import numpy as np
from matplotlib.axes import Axes
from pydantic import BaseModel, Field, model_validator


class PsdConfig(BaseModel):
    """Power Spectral Density (PSD) figure specification schema.

    Parameters
    ----------
    kind : {'complex', 'modular'}
        FFT of complex signal or FFT of |signal|.
    modes : list of int
        Mode indices for analysis.
    convention : {'symmetric', 'unitary', 'pragmatic'}, optional
        PSD convention (default: 'symmetric').
    t_range : tuple of float, optional
        Optional [t_start, t_end] for analysis window.
    plot_type : {'linear', 'loglog', 'semilogx', 'semilogy'}, optional
        Plot scaling type (default: None, inferred from kind).
    dt : float, optional
        Sampling interval (default: 1.0).

    Attributes
    ----------
    kind : str
    modes : list of int
    convention : str
    t_range : tuple of float, optional
    plot_type : str, optional
    dt : float

    """

    kind: Literal["complex", "modular"] = Field(
        ..., description="FFT of complex signal or FFT of |signal|"
    )
    modes: list[int] = Field(..., description="Mode indices for analysis")
    convention: Literal["symmetric", "unitary", "pragmatic"] = Field(
        "symmetric", description="PSD convention"
    )
    t_range: tuple[float, float] | None = Field(
        None, description="Optional [t_start, t_end] for analysis window"
    )
    plot_type: Literal["linear", "loglog", "semilogx", "semilogy"] | None = Field(
        None, description="Plot scaling type"
    )
    dt: float = Field(1.0, description="Sampling interval")

    @model_validator(mode="after")
    def validate_modes(self) -> "PsdConfig":
        if not self.modes:
            raise ValueError("modes must be non-empty")
        return self


# Alias for backward compatibility
PsdSpec = PsdConfig


class PsdPlotter:
    """Render power spectral density (PSD) for one or more modes on a matplotlib Axes.

    This visualizer plots the PSD for specified modes, using the provided
    pre-calculated data and configuration spec.

    Attributes
    ----------
    config_schema : type
        Configuration schema for this visualizer.

    """

    name: ClassVar[str] = "psd"
    description: ClassVar[str] = (
        "Power spectral density visualization (complex or modular)"
    )
    config_schema: ClassVar[type[PsdConfig]] = PsdConfig

    def __init__(self, config: PsdConfig | None = None, **kwargs):
        if config is None:
            self.config = self.config_schema(**kwargs)
        else:
            self.config = config

    def render(
        self,
        ax: Axes,
        data: dict[str, Any] | np.ndarray,
        plot_style: dict[str, Any] | None = None,
    ) -> str:
        """Render the visualization onto an existing matplotlib Axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot on.
        data : dict
            Pre-calculated PSD data containing 'axis' and 'psd'.
        plot_style : dict, optional
            Matplotlib Line2D keyword arguments for styling.

        Returns
        -------
        str
            Category tag for filenames (e.g., 'psd_complex').

        """
        if not isinstance(data, dict):
            raise ValueError("PsdPlotter expects dictionary data from PsdAnalyzer")

        vspec = self.config
        kind = vspec.kind
        convention = vspec.convention
        modes = vspec.modes

        # effective plot type: style override wins, then spec, then default by kind
        # Derive plot type from style x_scale/y_scale or defaults by kind
        style = plot_style or {}
        x_scale = style.get("x_scale")
        y_scale = style.get("y_scale")

        if x_scale or y_scale:
            xs = (x_scale or "linear").lower()
            ys = (y_scale or "linear").lower()
            if xs == "log" and ys == "log":
                eff_plot_type = "loglog"
            elif xs == "log" and ys != "log":
                eff_plot_type = "semilogx"
            elif xs != "log" and ys == "log":
                eff_plot_type = "semilogy"
            else:
                eff_plot_type = "linear"
        else:
            eff_plot_type = vspec.plot_type or (
                "linear" if kind == "complex" else "loglog"
            )

        # one- vs two-sided policy by kind
        sided = "two" if kind == "complex" else "one"

        # Draw
        axis = data["axis"]
        Pmat = data["psd"]  # (n_freq, n_modes)

        # Plot each mode line
        for j, m in enumerate(modes):
            P = Pmat[:, j]
            label = f"mode {m}"

            # two-sided handling: sort by frequency for continuity. For log-x
            # plots only the positive side is possible.
            if sided == "two":
                if eff_plot_type in ("loglog", "semilogx"):
                    mask = axis > 0
                    axis_p, P_p = axis[mask], P[mask]
                    # plot positive side only (log x cannot show negative)
                    if eff_plot_type == "loglog":
                        ax.loglog(axis_p, P_p, label=label)
                    elif eff_plot_type == "semilogx":
                        ax.semilogx(axis_p, P_p, label=label)
                else:
                    # linear x: show both sides; sort for continuity
                    idx = np.argsort(axis)
                    axis_s, P_s = axis[idx], P[idx]
                    if eff_plot_type == "semilogy":
                        ax.semilogy(axis_s, P_s, label=label)
                    else:  # linear
                        ax.plot(axis_s, P_s, label=label)
            else:
                # one-sided: keep non-negative for linear/semilogy, positive for log-x
                if eff_plot_type in ("loglog", "semilogx"):
                    mask = axis > 0
                else:
                    mask = axis >= 0
                axis_1, P_1 = axis[mask], P[mask]
                if eff_plot_type == "loglog":
                    ax.loglog(axis_1, P_1, label=label)
                elif eff_plot_type == "semilogx":
                    ax.semilogx(axis_1, P_1, label=label)
                elif eff_plot_type == "semilogy":
                    ax.semilogy(axis_1, P_1, label=label)
                else:
                    ax.plot(axis_1, P_1, label=label)

        # Style
        if style:
            if "xlim" in style:
                ax.set_xlim(*style["xlim"])
            if "ylim" in style:
                ax.set_ylim(*style["ylim"])
            if "legend" in style and bool(style["legend"]):
                ax.legend()

        # Show legend by default if any modes>1
        if not style or not style.get("legend", False):
            if len(modes) > 1:
                ax.legend()

        # Labels
        if convention in ("symmetric", "unitary"):
            ax.set_xlabel("ω [rad/s]")
        else:
            ax.set_xlabel("f [Hz]")
        ax.set_ylabel("PSD (a.u.)")

        category = f"psd_{kind}"
        return category
