"""qphase_sde: Power Spectral Density
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
from pydantic import Field, model_validator
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy
from qphase.core.protocols import PluginConfigBase

from .base import Analyzer
from .result import AnalysisResult

__all__ = [
    "PsdAnalyzer",
    "PsdAnalyzerConfig",
]


class PsdAnalyzerConfig(PluginConfigBase):
    """Configuration for PSD Analyzer."""

    kind: Literal["complex", "modular"] = Field(
        ..., description="FFT of complex signal or FFT of |signal|"
    )
    modes: list[int] = Field(..., description="Mode indices for analysis")
    convention: Literal["symmetric", "unitary", "pragmatic"] = Field(
        "symmetric", description="PSD convention"
    )
    dt: float | None = Field(None, description="Sampling interval (override)")
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
    max_peaks: int | None = Field(None, description="Maximum number of peaks to return")

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

        # Determine dt: priority to config override, then data attribute, then default
        dt = 1.0
        if config.dt is not None:
            dt = config.dt
        elif hasattr(data, "dt"):
            dt = float(data.dt)

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

                # Robust Smoothing Strategy:
                # 1. Work in Log domain (dB) to handle dynamic range and stabilize
                # 2. Apply Savitzky-Golay filter
                # 3. Convert back to linear for peak finding (to respect linear)

                p_log = _np.log10(p_data + 1e-20)

                # Determine window length
                if config.smooth_window:
                    w_len = config.smooth_window
                else:
                    # Default: ~2% of spectrum, min 5, must be odd
                    w_len = int(len(p_log) * 0.02)
                    if w_len < 5:
                        w_len = 5

                if w_len % 2 == 0:
                    w_len += 1

                try:
                    # Polyorder 2 preserves peak shapes better than higher orders
                    p_log_smooth = savgol_filter(p_log, w_len, 2)
                    p_smooth = 10**p_log_smooth
                except Exception:
                    # Fallback if filter fails (e.g. array too short)
                    p_smooth = p_data.copy()

                # Prepare kwargs
                fp_kwargs = {}

                # Determine height threshold
                # Use the stricter (maximum) of noise-based threshold and min_height
                if config.noise_threshold is not None:
                    # Estimate noise floor using median of smoothed data
                    noise_floor = _np.median(p_smooth)
                    calc_min_h = noise_floor * config.noise_threshold

                final_min_h = calc_min_h
                if config.min_height is not None:
                    if final_min_h is not None:
                        final_min_h = max(final_min_h, config.min_height)
                    else:
                        final_min_h = config.min_height

                if final_min_h is not None:
                    fp_kwargs["height"] = final_min_h

                # Determine prominence
                if config.prominence is not None:
                    fp_kwargs["prominence"] = config.prominence
                elif calc_min_h is not None:
                    # Heuristic: prominence is half the noise-based height threshold
                    fp_kwargs["prominence"] = calc_min_h * 0.5

                if config.distance is not None:
                    fp_kwargs["distance"] = config.distance

                # Find peaks on smoothed data
                peaks, props = find_peaks(p_smooth, **fp_kwargs)

                # Store results
                # We store the smoothed values for peaks to represent "fitted" height
                # But we might want to refine the frequency index using quadratic fit

                # Simple quadratic refinement for frequency
                refined_freqs = []
                refined_vals = []

                for pk in peaks:
                    # Default to grid values
                    f_pk = axis0[pk]
                    v_pk = p_smooth[pk]

                    # Try 3-point Gaussian fit (Parabola on Log)
                    if 0 < pk < len(p_log_smooth) - 1:
                        y1 = p_log_smooth[pk - 1]
                        y2 = p_log_smooth[pk]
                        y3 = p_log_smooth[pk + 1]

                        denom = 2 * (y1 - 2 * y2 + y3)
                        if denom != 0:
                            delta = (y1 - y3) / denom
                            # Check if delta is within reasonable bounds (+/- 0.5 bin)
                            if -0.5 <= delta <= 0.5:
                                # Interpolate frequency
                                df = axis0[1] - axis0[0]
                                f_pk = axis0[pk] + delta * df
                                # Interpolate value (parabola peak)
                                # y_peak = y2 - 0.25 * (y1 - y3) * delta
                                # v_pk = 10**y_peak

                    refined_freqs.append(f_pk)
                    refined_vals.append(v_pk)

                # Convert to arrays
                r_freqs = _np.array(refined_freqs)
                r_vals = _np.array(refined_vals)

                # Filter by max_peaks if requested
                if config.max_peaks is not None and len(peaks) > config.max_peaks:
                    # Sort by value (descending)
                    top_indices = _np.argsort(r_vals)[-config.max_peaks :]
                    # Sort indices back to frequency order (optional but nice)
                    top_indices = _np.sort(top_indices)

                    peaks = peaks[top_indices]
                    r_freqs = r_freqs[top_indices]
                    r_vals = r_vals[top_indices]

                    # Filter properties
                    for k, v in props.items():
                        props[k] = v[top_indices]

                peaks_info[m] = {
                    "indices": peaks,
                    "frequencies": r_freqs,
                    "values": r_vals,
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
