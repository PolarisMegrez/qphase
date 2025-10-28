from __future__ import annotations

"""Visualization Spec models (validated with Pydantic).

These schemas define input data formats and required fields for each figure
type. Validation happens before renderer selection to fail fast on mistakes.
"""

from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, model_validator


class BaseSpec(BaseModel):
    kind: str = Field(..., description="Figure kind identifier")


class PhasePortraitSpec(BaseSpec):
    kind: str = Field(..., pattern=r"^(re_im|abs_abs)$")
    modes: List[int] = Field(..., description="Mode indices")
    t_range: Optional[Tuple[float, float]] = Field(default=None, description="Optional [t_start, t_end]")

    @model_validator(mode="after")
    def _check(self) -> "PhasePortraitSpec":
        if self.kind == "re_im" and len(self.modes) != 1:
            raise ValueError("re_im requires exactly one mode index")
        if self.kind == "abs_abs" and len(self.modes) != 2:
            raise ValueError("abs_abs requires exactly two mode indices")
        if self.t_range is not None:
            t0, t1 = float(self.t_range[0]), float(self.t_range[1])
            if not (t1 > t0):
                raise ValueError("t_range must satisfy t_end > t_start")
        return self


class PsdSpec(BaseSpec):
    """Power Spectral Density figure specification.

    Fields:
    - kind: 'complex' (FFT of complex signal) or 'modular' (FFT of |signal|)
    - modes: one or more mode indices to plot on the same figure
    - convention: 'symmetric' (alias 'unitary') uses unitary FFT normalization and ω axis,
                  'pragmatic' uses standard FFT (no forward norm) and f axis.
    - plot_type: 'linear' | 'loglog' | 'semilogy' | 'semilogx' (optional; defaults: complex->linear, modular->loglog)
    - xlim: optional (xmin, xmax) for frequency axis limits
    - t_range: optional [t_start, t_end]
    - welch: optional dict placeholder for future Welch parameters (ignored for now)
    - zeropad: optional int placeholder for future zero-padding (ignored for now)
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
