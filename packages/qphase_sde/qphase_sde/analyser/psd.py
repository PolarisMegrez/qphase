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

from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

import numpy as _np
from pydantic import Field, model_validator
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy
from qphase.core.protocols import PluginConfigBase

from ..utils import resolve_mode_columns
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


@dataclass(frozen=True)
class _PsdEstimate:
    """PSD mean and cross-trajectory uncertainty for one mode."""

    axis: _np.ndarray
    mean: _np.ndarray
    std: _np.ndarray
    sem: _np.ndarray
    n_independent: int


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
    expected_freq_max: float | None = Field(
        None,
        gt=0.0,
        description=(
            "Optional maximum physical frequency expected in the output-axis "
            "units; analysis fails if it reaches the Nyquist limit"
        ),
    )
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
        mode_columns = resolve_mode_columns(data, modes)
        kind = config.kind
        convention = config.convention
        if dt <= 0.0:
            raise ValueError("PSD sampling interval must be positive")
        nyquist = _np.pi / dt if convention in ("symmetric", "unitary") else 0.5 / dt
        if (
            config.expected_freq_max is not None
            and config.expected_freq_max >= nyquist
        ):
            raise ValueError(
                f"expected_freq_max={config.expected_freq_max:.6g} reaches or "
                f"exceeds the PSD Nyquist limit {nyquist:.6g} for sample dt={dt:.6g}; "
                "reduce save_stride or the PSD dt override"
            )

        # Extract data array. Some array-like objects (e.g. TrajectorySet) wrap
        # the actual array in a ``.data`` attribute. NumPy/CuPy arrays also
        # expose a ``.data`` attribute, but it is a memoryview/MemoryPointer,
        # not an array. If ``data`` itself is already array-like (has ndim),
        # use it directly; otherwise unwrap ``data.data``.
        if hasattr(data, "ndim") and hasattr(data, "shape"):
            data_arr = data
        elif (
            hasattr(data, "data")
            and hasattr(data.data, "ndim")
            and hasattr(data.data, "shape")
        ):
            data_arr = data.data
        else:
            data_arr = data

        method = config.method
        nperseg = config.nperseg
        noverlap = config.noverlap
        nfft = config.nfft
        nw = config.nw
        k_tapers = config.k_tapers

        # Compute first to get the common axis and uncertainty metadata.
        estimate0 = self._estimate_single(
            data_arr[:, :, mode_columns[0]],
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
        axis0 = estimate0.axis
        P_list = [estimate0.mean]
        P_std_list = [estimate0.std]
        P_sem_list = [estimate0.sem]
        for column in mode_columns[1:]:
            estimate = self._estimate_single(
                data_arr[:, :, column],
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
            P_list.append(estimate.mean)
            P_std_list.append(estimate.std)
            P_sem_list.append(estimate.sem)
        P_mat = _np.vstack(P_list).T  # shape (n_freq, n_modes)
        P_std_mat = _np.vstack(P_std_list).T
        P_sem_mat = _np.vstack(P_sem_list).T

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
            "psd_std": P_std_mat,
            "psd_sem": P_sem_mat,
            "modes": modes,
            "kind": kind,
            "convention": convention,
            "sample_dt": dt,
            "nyquist": nyquist,
            "peaks": peaks_info,
            "uncertainty": {
                "kind": "standard_error",
                "field": "psd_sem",
                "std_field": "psd_std",
                "independent_unit": "trajectory",
                "n_independent": estimate0.n_independent,
                "ddof": 1,
                "available": estimate0.n_independent > 1,
            },
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
        """Compute mean two-sided PSD for one mode (compatibility wrapper)."""
        estimate = self._estimate_single(
            x,
            dt,
            kind=kind,
            convention=convention,
            window=window,
            method=method,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            nw=nw,
            k_tapers=k_tapers,
            backend=backend,
        )
        return estimate.axis, estimate.mean

    def _estimate_single(
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
    ) -> _PsdEstimate:
        """Compute PSD and cross-trajectory uncertainty for a single mode."""
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

    @staticmethod
    def _trajectory_statistics(
        spectra: _np.ndarray,
    ) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
        """Return mean, sample standard deviation, and SEM across trajectories."""
        n_independent = int(spectra.shape[0])
        mean = _np.mean(spectra, axis=0)
        if n_independent < 2:
            unavailable = _np.full(mean.shape, _np.nan, dtype=mean.dtype)
            return mean, unavailable, unavailable.copy()

        std = _np.std(spectra, axis=0, ddof=1)
        sem = std / _np.sqrt(float(n_independent))
        return mean, std, sem

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

    def _scale_and_shift_estimate(
        self,
        axis: _np.ndarray,
        mean: _np.ndarray,
        std: _np.ndarray,
        sem: _np.ndarray,
        dt: float,
        convention: str,
        n_fft: int,
        energy: float,
        n_independent: int,
    ) -> _PsdEstimate:
        """Apply the same linear PSD scaling to its mean and uncertainty."""
        shifted_axis, shifted_mean = self._scale_and_shift(
            axis, mean, dt, convention, n_fft, energy
        )
        _, shifted_std = self._scale_and_shift(
            axis, std, dt, convention, n_fft, energy
        )
        _, shifted_sem = self._scale_and_shift(
            axis, sem, dt, convention, n_fft, energy
        )
        return _PsdEstimate(
            axis=shifted_axis,
            mean=shifted_mean,
            std=shifted_std,
            sem=shifted_sem,
            n_independent=n_independent,
        )

    def _compute_periodogram(
        self,
        x_proc: Any,
        dt: float,
        convention: str,
        window: str | None,
        backend: Any,
    ) -> _PsdEstimate:
        """Classical averaged periodogram over trajectories.

        All heavy array operations stay on the active backend until the very end,
        minimizing device-to-host transfers for GPU backends.
        """
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
        trajectory_psd = backend.abs(X) ** 2
        del X
        mean_backend = backend.mean(trajectory_psd, axis=0)
        n_independent = int(x_proc.shape[0])

        mean = convert_to_numpy(mean_backend)
        if n_independent > 1:
            # Reuse the power buffer to avoid another n_traj x n_freq allocation.
            trajectory_psd -= mean_backend
            trajectory_psd *= trajectory_psd
            variance_backend = backend.mean(trajectory_psd, axis=0) * (
                n_independent / (n_independent - 1)
            )
            variance = convert_to_numpy(variance_backend)
            std = _np.sqrt(_np.maximum(variance, 0.0))
            sem = std / _np.sqrt(float(n_independent))
        else:
            std = _np.full(mean.shape, _np.nan, dtype=mean.dtype)
            sem = std.copy()

        axis = convert_to_numpy(backend.fftfreq(n_time, d=dt))
        energy = float(_np.sum(w * w))
        return self._scale_and_shift_estimate(
            axis,
            mean,
            std,
            sem,
            dt,
            convention,
            n_time,
            energy,
            n_independent,
        )

    def _compute_welch(
        self,
        x: _np.ndarray,
        dt: float,
        convention: str,
        window: str | None,
        nperseg: int | None,
        noverlap: int | None,
        nfft: int | None,
    ) -> _PsdEstimate:
        """Welch PSD with segments averaged before cross-trajectory statistics."""
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

        trajectory_spectra: list[_np.ndarray] = []
        for traj in x:
            segment_spectra: list[_np.ndarray] = []
            start = 0
            while start + nperseg <= n_time:
                seg = traj[start : start + nperseg] * w
                if nfft > nperseg:
                    seg = _np.pad(seg, (0, nfft - nperseg), mode="constant")
                X = _np.fft.fft(seg, norm=norm)
                segment_spectra.append(_np.abs(X) ** 2)
                start += step
            trajectory_spectra.append(_np.mean(segment_spectra, axis=0))

        spectra = _np.stack(trajectory_spectra, axis=0)
        mean, std, sem = self._trajectory_statistics(spectra)
        axis = _np.fft.fftfreq(nfft, d=dt)
        return self._scale_and_shift_estimate(
            axis,
            mean,
            std,
            sem,
            dt,
            convention,
            nfft,
            energy,
            n_traj,
        )

    def _compute_multitaper(
        self,
        x: _np.ndarray,
        dt: float,
        convention: str,
        nw: float,
        k_tapers: int | None,
    ) -> _PsdEstimate:
        """Multitaper PSD with tapers averaged within each trajectory."""
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

        trajectory_spectra: list[_np.ndarray] = []
        for traj in x:
            tapered_spectra: list[_np.ndarray] = []
            for taper in tapers:
                seg = traj * taper
                X = _np.fft.fft(seg, norm=norm)
                tapered_spectra.append(_np.abs(X) ** 2)
            trajectory_spectra.append(_np.mean(tapered_spectra, axis=0))

        spectra = _np.stack(trajectory_spectra, axis=0)
        mean, std, sem = self._trajectory_statistics(spectra)
        axis = _np.fft.fftfreq(n_time, d=dt)
        return self._scale_and_shift_estimate(
            axis,
            mean,
            std,
            sem,
            dt,
            convention,
            n_time,
            energy,
            n_traj,
        )
