"""qphase_sde: Power Spectral Density
---------------------------------------------------------
Compute power spectral density (PSD) from multi-trajectory time series for one
or more modes using FFT-based periodograms.

Behavior
--------
- Support two input interpretations: complex-valued directly (``kind='complex'``)
    or magnitude-based (``kind='modular'``).
- Provide common PSD conventions: unitary/symmetric (angular frequency 锠? and
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
from .peak_finding import (
    RationalPeakFinderConfig,
    ScipyPeakFinderConfig,
    create_peak_finder,
)
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
    method: Literal["periodogram", "welch", "multitaper"] = Field(
        "periodogram",
        description="PSD estimation method: periodogram, Welch, or multitaper",
    )
    nperseg: int | None = Field(
        None, ge=1, description="Welch segment length (default: n_time // 4)"
    )
    noverlap: int | None = Field(
        None, ge=0, description="Welch overlap (default: nperseg // 2)"
    )
    nfft: int | None = Field(
        None, ge=1, description="FFT length for Welch (default: nperseg)"
    )
    nw: float = Field(
        2.5, gt=0.0, description="Multitaper time-half-bandwidth product NW"
    )
    k_tapers: int | None = Field(
        None, ge=1, description="Number of DPSS tapers (default: int(2*nw)-1)"
    )

    # Peak finding configuration
    # Supports bool (legacy), string ("scipy", "rational"), or specific config objects
    find_peaks: bool | str | ScipyPeakFinderConfig | RationalPeakFinderConfig = Field(
        False, description="Peak finding method or configuration"
    )

    # Deprecated fields retained for backward compatibility with 'find_peaks=True'
    min_height: float | None = Field(
        None, description="[Deprecated] Minimum peak height"
    )
    prominence: float | None = Field(None, description="[Deprecated] Peak prominence")
    distance: int | None = Field(
        None, description="[Deprecated] Minimum horizontal distance"
    )
    smooth_window: int | None = Field(
        None, description="[Deprecated] Window length for smoothing"
    )
    noise_threshold: float | None = Field(
        None, description="[Deprecated] Threshold vs noise floor"
    )
    max_peaks: int | None = Field(
        None, description="[Deprecated] Maximum number of peaks to return"
    )

    @model_validator(mode="after")
    def validate_modes(self) -> "PsdAnalyzerConfig":
        if not self.modes:
            raise ValueError("modes must be non-empty")
        return self


class PsdAnalyzer(Analyzer):
    """Analyzer for Power Spectral Density."""

    name: ClassVar[str] = "psd"
    description: ClassVar[str] = "Power Spectral Density analyzer"
    config_schema: ClassVar[type[PsdAnalyzerConfig]] = PsdAnalyzerConfig

    def __init__(self, config: PsdAnalyzerConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

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

        method = config.method
        nperseg = config.nperseg
        noverlap = config.noverlap
        nfft = config.nfft
        nw = config.nw
        k_tapers = config.k_tapers

        # Compute first to get axis
        axis0, P0 = self._compute_single(
            data_arr[:, :, modes[0]],
            dt,
            kind=kind,
            convention=convention,
            window=config.window,
            method=method,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            nw=nw,
            k_tapers=k_tapers,
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
                method=method,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                nw=nw,
                k_tapers=k_tapers,
                backend=backend,
            )
            P_list.append(Pm)
        P_mat = _np.vstack(P_list).T  # shape (n_freq, n_modes)

        # Peak finding
        peaks_info = {}

        # Instantiate peak finder
        finder = None
        pf_conf = config.find_peaks

        # Backward compatibility logic.
        # Build a Scipy config when using legacy bool/string peak-finding flags.
        legacy_scipy = False
        if isinstance(pf_conf, bool) and pf_conf:
            legacy_scipy = True
        elif isinstance(pf_conf, str) and pf_conf.lower() in ["scipy", "standard"]:
            legacy_scipy = True

        if legacy_scipy:
            # Map legacy fields
            scipy_conf = ScipyPeakFinderConfig(
                method="scipy",
                min_height=config.min_height,
                prominence=config.prominence,
                distance=config.distance,
                smooth_window=config.smooth_window,
                noise_threshold=config.noise_threshold,
                max_peaks=config.max_peaks,
            )
            finder = create_peak_finder(scipy_conf)
        else:
            # Just try creating directly
            finder = create_peak_finder(pf_conf)

        if finder:
            for i, m in enumerate(modes):
                p_data = P_mat[:, i]
                # Delegate to finder
                try:
                    p_info = finder.find_peaks(axis0, p_data)
                    peaks_info[m] = p_info.model_dump()
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(
                        f"Peak finding failed for mode {m}: {e}"
                    )
                    # Fallback to empty
                    peaks_info[m] = {
                        "indices": [],
                        "frequencies": [],
                        "values": [],
                        "properties": {},
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
        method: str = "periodogram",
        nperseg: int | None = None,
        noverlap: int | None = None,
        nfft: int | None = None,
        nw: float = 2.5,
        k_tapers: int | None = None,
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

        if method == "periodogram":
            return self._compute_periodogram(x_proc, dt, convention, window, backend)

        # Welch and multitaper are implemented on NumPy arrays.
        x_np = convert_to_numpy(x_proc)
        if method == "welch":
            return self._compute_welch(
                x_np, dt, convention, window, nperseg, noverlap, nfft
            )
        if method == "multitaper":
            return self._compute_multitaper(x_np, dt, convention, nw, k_tapers)

        raise ValueError(f"Unsupported PSD method: {method}")

    def _get_window(self, window: str | None, n: int) -> _np.ndarray:
        """Return a NumPy window of length ``n``."""
        if window is None:
            return _np.ones(n)
        try:
            return getattr(_np, window)(n)
        except AttributeError:
            # Unknown window name: fall back to rectangular
            return _np.ones(n)

    def _scale_and_shift(
        self,
        axis: _np.ndarray,
        P: _np.ndarray,
        dt: float,
        convention: str,
        n_fft: int,
        energy: float,
    ) -> tuple[_np.ndarray, _np.ndarray]:
        """Apply PSD scaling, shift zero frequency to center, and return."""
        if energy <= 0.0:
            energy = 1.0
        if convention in ("symmetric", "unitary"):
            scale_p = n_fft * dt / (2.0 * _np.pi * energy)
            P = P * scale_p
            axis = axis * 2.0 * _np.pi
        else:
            scale_p = dt / energy
            P = P * scale_p
        axis = _np.fft.fftshift(axis)
        P = _np.fft.fftshift(P)
        return axis, P

    def _compute_periodogram(
        self,
        x_proc: Any,
        dt: float,
        convention: str,
        window: str | None,
        backend: Any,
    ) -> tuple[_np.ndarray, _np.ndarray]:
        """Classical averaged periodogram over trajectories."""
        n_time = int(x_proc.shape[-1])

        if window:
            w = self._get_window(window, n_time)
            w_backend = backend.asarray(w)
            x_proc = x_proc * w_backend
        else:
            w = _np.ones(n_time)

        norm: Literal["backward", "ortho", "forward"] | None
        if convention in ("symmetric", "unitary"):
            norm = "ortho"
        else:
            norm = None

        X = backend.fft(x_proc, axis=-1, norm=norm)
        P_backend = backend.mean(backend.abs(X) ** 2, axis=0)
        axis_backend = backend.fftfreq(n_time, d=dt)

        energy = float(_np.sum(w * w))
        if convention in ("symmetric", "unitary"):
            scale_p = n_time * dt / (2.0 * backend.pi * energy)
            P_backend = P_backend * scale_p
            axis_backend = axis_backend * 2.0 * backend.pi
        else:
            scale_p = dt / energy
            P_backend = P_backend * scale_p

        axis = convert_to_numpy(axis_backend)
        P = convert_to_numpy(P_backend)
        axis = _np.fft.fftshift(axis)
        P = _np.fft.fftshift(P, axes=0)
        return axis, P

    def _compute_welch(
        self,
        x: _np.ndarray,
        dt: float,
        convention: str,
        window: str | None,
        nperseg: int | None,
        noverlap: int | None,
        nfft: int | None,
    ) -> tuple[_np.ndarray, _np.ndarray]:
        """Welch's method: average periodograms over overlapping segments."""
        n_traj, n_time = x.shape
        if nperseg is None:
            nperseg = max(1, n_time // 4)
        if noverlap is None:
            noverlap = nperseg // 2
        if nfft is None:
            nfft = nperseg

        if nperseg > n_time:
            nperseg = n_time
            noverlap = 0
        step = nperseg - noverlap
        if step <= 0:
            raise ValueError("noverlap must be smaller than nperseg")

        w = self._get_window(window, nperseg)
        if window is None:
            # Welch defaults to a Hann window.
            w = _np.hanning(nperseg)
        w = w.astype(x.real.dtype if _np.iscomplexobj(x) else x.dtype)
        energy = float(_np.sum(w * w))

        norm: Literal["backward", "ortho", "forward"] | None
        if convention in ("symmetric", "unitary"):
            norm = "ortho"
        else:
            norm = None

        spectra: list[_np.ndarray] = []
        for traj in x:
            start = 0
            while start + nperseg <= n_time:
                seg = traj[start : start + nperseg] * w
                if nfft > nperseg:
                    seg = _np.pad(seg, (0, nfft - nperseg), mode="constant")
                X = _np.fft.fft(seg, norm=norm)
                spectra.append(_np.abs(X) ** 2)
                start += step

        if not spectra:
            # Not enough data for even one segment; fall back to one segment.
            seg = x[0, :nperseg] * w
            if nfft > nperseg:
                seg = _np.pad(seg, (0, nfft - nperseg), mode="constant")
            X = _np.fft.fft(seg, norm=norm)
            spectra.append(_np.abs(X) ** 2)

        P = _np.mean(spectra, axis=0)
        axis = _np.fft.fftfreq(nfft, d=dt)
        return self._scale_and_shift(axis, P, dt, convention, nfft, energy)

    def _compute_multitaper(
        self,
        x: _np.ndarray,
        dt: float,
        convention: str,
        nw: float,
        k_tapers: int | None,
    ) -> tuple[_np.ndarray, _np.ndarray]:
        """Multitaper PSD estimate using discrete prolate spheroidal sequences."""
        from scipy.signal.windows import dpss

        n_traj, n_time = x.shape
        if k_tapers is None:
            k_tapers = max(1, int(2.0 * nw) - 1)

        tapers = dpss(n_time, nw, Kmax=k_tapers, sym=False)
        tapers = tapers.astype(x.real.dtype if _np.iscomplexobj(x) else x.dtype)
        energy = 1.0  # dpss windows are normalized to unit energy

        norm: Literal["backward", "ortho", "forward"] | None
        if convention in ("symmetric", "unitary"):
            norm = "ortho"
        else:
            norm = None

        spectra: list[_np.ndarray] = []
        for traj in x:
            for taper in tapers:
                seg = traj * taper
                X = _np.fft.fft(seg, norm=norm)
                spectra.append(_np.abs(X) ** 2)

        P = _np.mean(spectra, axis=0)
        axis = _np.fft.fftfreq(n_time, d=dt)
        return self._scale_and_shift(axis, P, dt, convention, n_time, energy)
