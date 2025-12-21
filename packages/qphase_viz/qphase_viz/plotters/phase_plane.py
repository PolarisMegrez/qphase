"""qphase_sde: Phase Portrait Renderer
----------------------------------
Render phase portraits from multi-trajectory time series for one or two modes.

Behavior
--------
- Support 're_im' (Re vs Im for one mode) and 'abs_abs' (|m1| vs |m2|) kinds.
    Spec validation and rendering semantics are documented by the functions.

Notes
-----
- Avoid importing heavy plotting libraries outside of rendering contexts.

"""

__all__ = [
    "PhasePortraitPlotter",
    "PhasePortraitConfig",
    "PhasePortraitSpec",
]


from typing import Any, ClassVar

import numpy as np
from matplotlib.axes import Axes
from pydantic import BaseModel, Field, model_validator


class PhasePortraitConfig(BaseModel):
    """Phase portrait figure specification schema.

    Parameters
    ----------
    kind : str
        Must be 're_im' or 'abs_abs'.
    modes : list of int
        Mode indices for plotting.
    t_range : tuple of float, optional
        Optional [t_start, t_end] for time slicing.
    plot_every : int, optional
        Optional decimation factor for plotting only (>=1).

    Attributes
    ----------
    kind : str
    modes : list of int
    t_range : tuple of float, optional
    plot_every : int, optional

    """

    kind: str = Field(..., description="Must be 're_im' or 'abs_abs'")
    modes: list[int] = Field(..., description="Mode indices for plotting")
    t_range: tuple[float, float] | None = Field(
        None, description="Optional [t_start, t_end] for time slicing"
    )
    plot_every: int = Field(
        1, ge=1, description="Optional decimation factor for plotting only (>=1)"
    )

    @model_validator(mode="after")
    def validate_kind(self) -> "PhasePortraitConfig":
        if self.kind not in ("re_im", "abs_abs"):
            raise ValueError(f"Invalid kind: {self.kind}. Must be 're_im' or 'abs_abs'")
        return self


# Alias for backward compatibility
PhasePortraitSpec = PhasePortraitConfig


class PhasePortraitPlotter:
    """Render a phase portrait onto an existing matplotlib Axes.

    Supports two figure kinds:
        - 're_im': Re(alpha_m) vs Im(alpha_m) for a single mode m
        - 'abs_abs': |alpha_m1| vs |alpha_m2| for a pair of modes

    Attributes
    ----------
    config_schema : type
        Configuration schema for this visualizer.

    """

    name: ClassVar[str] = "phase_portrait"
    description: ClassVar[str] = "Phase portrait visualization (Re/Im or |m1|/|m2|)"
    config_schema: ClassVar[type[PhasePortraitConfig]] = PhasePortraitConfig

    def __init__(self, config: PhasePortraitConfig | None = None, **kwargs):
        if config is None:
            self.config = self.config_schema(**kwargs)
        else:
            self.config = config

    def render(
        self,
        ax: Axes,
        data: np.ndarray | dict[str, Any],
        plot_style: dict[str, Any] | None = None,
    ) -> str:
        """Render the visualization onto an existing matplotlib Axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot on.
        data : np.ndarray
            Complex array of shape (n_traj, n_keep, n_modes), time series data.
        plot_style : dict, optional
            Matplotlib Line2D keyword arguments for styling.

        Returns
        -------
        str
            Category tag for filenames ('pp_reim' or 'pp_abs').

        """
        if isinstance(data, dict):
            # Handle case where data might be wrapped
            raise ValueError("PhasePortraitPlotter expects numpy array data")

        vspec = self.config

        style = dict(plot_style or {})
        n_traj, n_keep, n_modes = data.shape
        kind = vspec.kind

        if kind == "re_im":
            modes = vspec.modes
            m = modes[0]
            x = data[:, :, m].real
            y = data[:, :, m].imag
            for i in range(n_traj):
                ax.plot(x[i], y[i], **style)
            ax.set_xlabel(f"Re(Mode {m})")
            ax.set_ylabel(f"Im(Mode {m})")
            return "pp_reim"
        elif kind == "abs_abs":
            modes = vspec.modes
            m1, m2 = modes
            x = np.abs(data[:, :, m1])
            y = np.abs(data[:, :, m2])
            for i in range(n_traj):
                ax.plot(x[i], y[i], **style)
            ax.set_xlabel(f"|Mode {m1}|")
            ax.set_ylabel(f"|Mode {m2}|")
            return "pp_abs"

        return "unknown"
