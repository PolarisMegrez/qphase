"""qphase_sde: Power Spectral Density
---------------------------------
Compute power spectral density (PSD) from multi-trajectory time series for one
or more modes using FFT-based periodograms.

Behavior
--------
- Support two input interpretations: complex-valued directly (``kind='complex'``)
    or magnitude-based (``kind='modular'``).
- Provide common PSD conventions: unitary/symmetric (angular frequency ω) and
    pragmatic (frequency f). Exact scaling, return shapes, and error semantics are
    specified by the function docstrings.

Notes
-----
- These utilities are backend-agnostic with NumPy implementations and are used
    by visualizer as well as analysis pipelines.

"""

from typing import Any, ClassVar, Literal, cast

import numpy as _np
from pydantic import BaseModel, Field, model_validator
from qphase.backend.xputil import convert_to_numpy

from .base import Analyzer

__all__ = [
    "PsdAnalyzer",
    "PsdAnalyzerConfig",
]


class PsdAnalyzerConfig(BaseModel):
    """Configuration for PSD Analyzer."""

    kind: Literal["complex", "modular"] = Field(
        ..., description="FFT of complex signal or FFT of |signal|"
    )
    modes: list[int] = Field(..., description="Mode indices for analysis")
    convention: Literal["symmetric", "unitary", "pragmatic"] = Field(
        "symmetric", description="PSD convention"
    )
    dt: float = Field(1.0, description="Sampling interval")

    @model_validator(mode="after")
    def validate_modes(self) -> "PsdAnalyzerConfig":
        if not self.modes:
            raise ValueError("modes must be non-empty")
        return self


class PsdAnalyzer(Analyzer):
    """Analyzer for Power Spectral Density."""

    name: ClassVar[str] = "psd"
    description: ClassVar[str] = "Power Spectral Density analyzer"
    config_schema: ClassVar[type[PsdAnalyzerConfig]] = PsdAnalyzerConfig  # type: ignore[assignment]

    def __init__(self, config: PsdAnalyzerConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)  # type: ignore[arg-type]

    def analyze(self, data: Any, **kwargs) -> dict:
        """Compute PSD for multiple modes.

        Parameters
        ----------
        data : numpy.ndarray
            Complex-like time series array of shape ``(n_traj, n_time, n_modes)``.
        **kwargs : Any
            Additional arguments (ignored).

        Returns
        -------
        dict
            Dictionary with keys 'axis', 'psd', 'modes', 'kind', 'convention'.

        """
        config = cast(PsdAnalyzerConfig, self.config)
        dt = config.dt
        modes = config.modes
        kind = config.kind
        convention = config.convention

        # Optional backend override from kwargs
        backend = kwargs.get("backend")

        # Compute first to get axis
        axis0, P0 = self._compute_single(
            data[:, :, modes[0]], dt, kind=kind, convention=convention, backend=backend
        )
        P_list = [P0]
        for m in modes[1:]:
            _, Pm = self._compute_single(
                data[:, :, m], dt, kind=kind, convention=convention, backend=backend
            )
            P_list.append(Pm)
        P_mat = _np.vstack(P_list).T  # shape (n_freq, n_modes)
        return {
            "axis": axis0,
            "psd": P_mat,
            "modes": modes,
            "kind": kind,
            "convention": convention,
        }

    def _compute_single(
        self,
        x: Any,
        dt: float,
        *,
        kind: str = "complex",
        convention: str = "symmetric",
        backend: Any | None = None,
    ) -> tuple[Any, Any]:
        """Compute two-sided power spectral density (PSD) for a single mode."""
        if backend is None:
            from qphase.backend.numpy_backend import NumpyBackend

            backend = NumpyBackend()

        # Convert input to backend array
        x_arr = backend.asarray(x)

        if kind == "modular":
            x_proc = backend.abs(x_arr)
        else:
            x_proc = x_arr

        # Ensure 2D: (n_traj, n_time)
        ndim = getattr(x_proc, "ndim", None)
        if ndim == 1:
            # Try to add dimension using slicing
            try:
                x_proc = x_proc[None, :]
            except Exception:
                # If slicing fails, we might be dealing with a backend
                # that doesn't support it
                # But standard backends (numpy, torch, cupy) do.
                pass
        elif ndim is None or ndim < 1:
            raise ValueError("[524] input `x` must be a 1-D or 2-D array")

        n_time = int(x_proc.shape[-1])

        if convention in ("symmetric", "unitary"):
            norm = "ortho"
        else:
            norm = None

        # FFT
        X = backend.fft(x_proc, axis=-1, norm=norm)  # type: ignore[arg-type]

        # Power: |X|^2
        absX = backend.abs(X)
        absX2 = absX**2

        # Mean over trajectories (axis 0)
        P_backend = backend.mean(absX2, axis=0)

        # Frequencies
        axis_backend = backend.fftfreq(n_time, d=dt)

        # Scaling
        if convention in ("symmetric", "unitary"):
            scale_p = dt / (2.0 * backend.pi)
            P_backend = P_backend * scale_p

            scale_axis = 2.0 * backend.pi
            axis_backend = axis_backend * scale_axis
        else:
            scale_p = dt / float(n_time)
            P_backend = P_backend * scale_p

        # Convert to numpy for return
        axis = convert_to_numpy(axis_backend)
        P = convert_to_numpy(P_backend)

        return axis, P
