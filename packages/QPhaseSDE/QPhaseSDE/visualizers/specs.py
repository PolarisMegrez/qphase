"""
QPhaseSDE: Visualization specs
------------------------------
Pydantic models that describe the input schema for visualization requests used
by the visualization service and renderers.

Behavior
- Provide strict validation and light normalization (e.g., defaults) before a
    renderer is selected.

Notes
- Detailed field semantics, validation rules, and examples are documented in
    the individual class docstrings below.
"""

__all__ = [
    "BaseSpec",
    "PhasePortraitSpec",
    "PsdSpec",
]

from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, model_validator


class BaseSpec(BaseModel):
    """
    Base class for visualization specification schemas.

    Parameters
    ----------
    kind : str
        Figure kind identifier.

    Attributes
    ----------
    kind : str
        Figure kind identifier.
    """
    kind: str = Field(..., description="Figure kind identifier")


class PhasePortraitSpec(BaseSpec):
    """
    Phase portrait figure specification schema.

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

    Methods
    -------
    _check

    Examples
    --------
    >>> PhasePortraitSpec(kind="re_im", modes=[0], t_range=(0.0, 10.0), plot_every=2)
    """
    kind: str = Field(..., pattern=r"^(re_im|abs_abs)$")
    modes: List[int] = Field(..., description="Mode indices")
    t_range: Optional[Tuple[float, float]] = Field(default=None, description="Optional [t_start, t_end]")
    plot_every: Optional[int] = Field(default=None, description="Optional decimation factor for plotting only (>=1)")

    @model_validator(mode="after")
    def _check(self) -> "PhasePortraitSpec":
        """
        Validate the phase portrait spec fields.
        """
        if self.kind == "re_im" and len(self.modes) != 1:
            raise ValueError("re_im requires exactly one mode index")
        if self.kind == "abs_abs" and len(self.modes) != 2:
            raise ValueError("abs_abs requires exactly two mode indices")
        if self.t_range is not None:
            t0, t1 = float(self.t_range[0]), float(self.t_range[1])
            if not (t1 > t0):
                raise ValueError("t_range must satisfy t_end > t_start")
        if self.plot_every is not None:
            pe = int(self.plot_every)
            if pe < 1:
                raise ValueError("plot_every must be >= 1")
        return self


class PsdSpec(BaseSpec):
    """
    Power Spectral Density (PSD) figure specification schema.

    Parameters
    ----------
    kind : {'complex', 'modular'}
        FFT of complex signal or FFT of |signal|.
    modes : list of int
        Mode indices to plot.
    convention : {'symmetric', 'unitary', 'pragmatic'}
        Fourier convention; 'unitary' treated as 'symmetric'.
    plot_type : {'linear', 'loglog', 'semilogy', 'semilogx'}, optional
        Plot type for frequency axis.
    xlim : tuple of float, optional
        [xmin, xmax] for frequency axis limits.
    t_range : tuple of float, optional
        [t_start, t_end] for time slicing.
    welch : dict, optional
        Placeholder for future Welch method parameters.
    zeropad : int, optional
        Placeholder for future zero-padding.

    Attributes
    ----------
    kind : str
    modes : list of int
    convention : str
    plot_type : str, optional
    xlim : tuple of float, optional
    t_range : tuple of float, optional
    welch : dict, optional
    zeropad : int, optional

    Methods
    -------
    _check

    Examples
    --------
    >>> PsdSpec(kind="complex", modes=[0], convention="symmetric", t_range=(0.0, 10.0))
    """

    kind: Literal["complex", "modular"]
    modes: List[int] = Field(..., description="Mode indices to include")
    convention: Literal["symmetric", "unitary", "pragmatic"] = Field(
        "symmetric", description="Fourier convention; 'unitary' treated as 'symmetric'"
    )
    plot_type: Optional[Literal["linear", "loglog", "semilogy", "semilogx"]] = None
    xlim: Optional[Tuple[float, float]] = Field(default=None, description="Optional [xmin, xmax] for frequency axis")
    t_range: Optional[Tuple[float, float]] = Field(default=None, description="Optional [t_start, t_end]")
    welch: Optional[dict] = Field(default=None, description="Placeholder for future Welch method parameters")
    zeropad: Optional[int] = Field(default=None, description="Placeholder for future zero-padding")

    @model_validator(mode="after")
    def _check(self) -> "PsdSpec":
        """
        Validate the PSD spec fields and set defaults.
        """
        if not self.modes:
            raise ValueError("psd.modes must contain at least one index")
        if self.t_range is not None:
            t0, t1 = float(self.t_range[0]), float(self.t_range[1])
            if not (t1 > t0):
                raise ValueError("t_range must satisfy t_end > t_start")
        # default plot_type based on kind
        if self.plot_type is None:
            self.plot_type = "linear" if self.kind == "complex" else "loglog"
        return self
