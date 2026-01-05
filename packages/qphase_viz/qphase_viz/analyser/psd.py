"""qphase_viz: Power Spectral Density
---------------------------------------------------------
Compute power spectral density (PSD) from multi-trajectory time series for one
or more modes using FFT-based periodograms.

Behavior
--------
- Support two input interpretations: complex-valued directly (``kind='complex'``)
    or magnitude-based (``kind='modular'``).
- Provide common PSD conventions: unitary/symmetric (angular frequency 蠅) and
    pragmatic (frequency f). Exact scaling, return shapes, and error semantics are
    specified by the function docstrings.

Public API
----------
``PsdAnalyzer`` : Power spectral density analyzer.
``PsdAnalyzerConfig`` : Configuration for PSD analyzer.

Notes
-----
- These utilities are backend-agnostic with NumPy implementations and are used
    by visualizer as well as analysis pipelines.

"""

from typing import Any, ClassVar, Literal, cast

import numpy as _np
from pydantic import BaseModel, Field, model_validator
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy

from .base import Analyzer
from .result import AnalysisResult

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
    window: str | None = Field(
        None, description="Window function name (e.g. 'hanning')"
    )
    # Peak finding configuration
    find_peaks: bool = Field(False, description="Whether to find peaks")
    min_height: float | None = Field(None, description="Minimum peak height")
    prominence: float | None = Field(None, description="Peak prominence")
    distance: int | None = Field(None, description="Minimum horizontal distance")
    smooth_window: int | None = Field(None, description="Window length for smoothing")
    noise_threshold: float | None = Field(None, description="Threshold vs noise floor")

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

    def analyze(self, data: Any, backend: BackendBase) -> AnalysisResult:
        """Compute PSD for multiple modes.

        Parameters
        ----------
        data : Any
            Complex-like time series array of shape ``(n_traj, n_time, n_modes)``
            or TrajectorySet.
        backend : BackendBase
            Backend to use for computation.

        Returns
        -------
        AnalysisResult
            Result containing PSD data.

        """
        config = cast(PsdAnalyzerConfig, self.config)
        dt = config.dt
        modes = config.modes
        kind = config.kind
        convention = config.convention

        # Extract data array
        if hasattr(data, "data"):
            data_arr = data.data
        else:
            data_arr = data

        # Compute first to get axis
        axis0, P0 = self._compute_single(
            data_arr[:, :, modes[0]],
            dt,
            kind=kind,
            convention=convention,
            window=config.window,
            backend=backend,
        )
        P_list = [P0]
        for m in modes[1:]:
            _, Pm = self._compute_single(
                data_arr[:, :, m],
                dt,
                kind=kind,
                convention=convention,
                window=config.window,
                backend=backend,
            )
            P_list.append(Pm)
        P_mat = _np.vstack(P_list).T  # shape (n_freq, n_modes)

        # Peak finding
        peaks_info = {}
        if config.find_peaks:
            from scipy.signal import find_peaks, savgol_filter

            for i, m in enumerate(modes):
                # Find peaks for this mode
                p_data = P_mat[:, i]

                # Smoothing (on a copy to avoid affecting returned data)
                p_smooth = p_data.copy()
                if config.smooth_window:
                    try:
                        window_len = config.smooth_window
                        if window_len % 2 == 0:
                            window_len += 1
                        if window_len < len(p_smooth):
                            p_smooth = savgol_filter(p_smooth, window_len, 3)
                    except Exception:
                        pass

                # Prepare kwargs
                fp_kwargs = {}

                # Auto-calculate height/prominence if requested via noise_threshold
                if config.noise_threshold is not None:
                    # Estimate noise floor using median of smoothed data
                    noise_floor = _np.median(p_smooth)
                    min_h = noise_floor * config.noise_threshold
                    fp_kwargs["height"] = min_h
                    if config.prominence is None:
                        fp_kwargs["prominence"] = min_h * 0.5  # Heuristic

                if config.min_height is not None:
                    fp_kwargs["height"] = config.min_height
                if config.prominence is not None:
                    fp_kwargs["prominence"] = config.prominence
                if config.distance is not None:
                    fp_kwargs["distance"] = config.distance

                # Find peaks on smoothed data
                peaks, props = find_peaks(p_smooth, **fp_kwargs)

                # Store results (values from original data or smoothed?)
                # Usually we want values from the original data at those indices,
                # or the smoothed values. Let's store original values for accuracy
                # but use indices found on smoothed data.
                peaks_info[m] = {
                    "indices": peaks,
                    "frequencies": axis0[peaks],
                    "values": p_data[peaks],
                    "properties": props,
                }

        result_dict = {
            "axis": axis0,
            "psd": P_mat,
            "modes": modes,
            "kind": kind,
            "convention": convention,
            "peaks": peaks_info,
        }

        return AnalysisResult(data_dict=result_dict, meta=result_dict)

    def _compute_single(
        self,
        x: Any,
        dt: float,
        *,
        kind: str = "complex",
        convention: str = "symmetric",
        window: str | None = None,
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

        # Apply window if requested
        if window:
            try:
                # Use numpy to generate window
                win_func = getattr(_np, window)
                w = win_func(n_time)
                w_backend = backend.asarray(w)
                x_proc = x_proc * w_backend
            except AttributeError:
                # Fallback or warning? For now silent fallback
                pass

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

        # Shift zero frequency to center
        axis = _np.fft.fftshift(axis)
        P = _np.fft.fftshift(P, axes=0)

        return axis, P
