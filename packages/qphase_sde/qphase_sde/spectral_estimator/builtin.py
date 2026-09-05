"""Periodogram, Welch, and multitaper estimator implementations."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field
from qphase.backend.xputil import convert_to_numpy
from qphase.core.protocols import PluginConfigBase

from .base import PsdEstimate, SpectralEstimatorCapabilities


class PeriodogramEstimatorConfig(PluginConfigBase):
    window: str | None = Field(None, description="Window function name")
    fft_chunk_trajectories: int | None = Field(
        None,
        ge=1,
        description="Trajectories per FFT chunk; None transforms the full batch",
    )


class WelchEstimatorConfig(PluginConfigBase):
    window: str | None = Field(None, description="Window function name")
    nperseg: int | None = Field(None, ge=1, description="Samples per segment")
    noverlap: int | None = Field(None, ge=0, description="Overlapping samples")
    nfft: int | None = Field(None, ge=1, description="FFT length")


class MultitaperEstimatorConfig(PluginConfigBase):
    nw: float = Field(2.5, gt=0.0, description="Time-half-bandwidth product")
    k_tapers: int | None = Field(None, ge=1, description="Number of DPSS tapers")


class _EstimatorMath:
    @staticmethod
    def window(name: str | None, n: int) -> np.ndarray:
        if name is None:
            return np.ones(n)
        try:
            return getattr(np, name)(n)
        except AttributeError:
            return np.ones(n)

    @staticmethod
    def trajectory_statistics(
        spectra: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = int(spectra.shape[0])
        mean = np.mean(spectra, axis=0)
        if count < 2:
            unavailable = np.full(mean.shape, np.nan, dtype=mean.dtype)
            return mean, unavailable, unavailable.copy()
        std = np.std(spectra, axis=0, ddof=1)
        return mean, std, std / np.sqrt(float(count))

    @staticmethod
    def scaled(
        axis: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        sem: np.ndarray,
        *,
        dt: float,
        convention: str,
        n_fft: int,
        energy: float,
        count: int,
    ) -> PsdEstimate:
        if energy <= 0.0:
            energy = 1.0
        if convention in {"symmetric", "unitary"}:
            scale = n_fft * dt / (2.0 * np.pi * energy)
            axis = axis * 2.0 * np.pi
        else:
            scale = dt / energy
        return PsdEstimate(
            axis=np.fft.fftshift(axis),
            mean=np.fft.fftshift(mean * scale),
            std=np.fft.fftshift(std * scale),
            sem=np.fft.fftshift(sem * scale),
            n_independent=count,
        )


class PeriodogramEstimator(_EstimatorMath):
    name: ClassVar[str] = "periodogram"
    description: ClassVar[str] = "Full-length averaged trajectory periodogram"
    config_schema: ClassVar[type[PeriodogramEstimatorConfig]] = (
        PeriodogramEstimatorConfig
    )

    def __init__(self, config: PeriodogramEstimatorConfig | None = None) -> None:
        self.config = config or PeriodogramEstimatorConfig()

    def capabilities(self) -> SpectralEstimatorCapabilities:
        return SpectralEstimatorCapabilities(backend_native=True)

    def estimate(self, x, dt, convention, backend) -> PsdEstimate:
        x = backend.asarray(x)
        if x.ndim == 1:
            x = x[None, :]
        n_time = int(x.shape[-1])
        window = self.window(self.config.window, n_time)
        if self.config.window:
            x = x * backend.asarray(window)
        norm: Literal["ortho"] | None = (
            "ortho" if convention in {"symmetric", "unitary"} else None
        )
        count = int(x.shape[0])
        chunk = self.config.fft_chunk_trajectories
        if chunk is None or int(chunk) >= count:
            mean, std, sem = self._power_moments(x, norm, backend)
        else:
            mean, std, sem = self._chunked_power_moments(x, norm, int(chunk), backend)
        axis = convert_to_numpy(backend.fftfreq(n_time, d=dt))
        return self.scaled(
            axis,
            mean,
            std,
            sem,
            dt=dt,
            convention=convention,
            n_fft=n_time,
            energy=float(np.sum(window * window)),
            count=count,
        )

    def _power_moments(self, x, norm, backend) -> tuple[Any, Any, Any]:
        """Mean/std/sem of per-trajectory powers from one full-batch FFT."""
        transformed = backend.fft(x, axis=-1, norm=norm)
        powers = backend.abs(transformed) ** 2
        del transformed
        mean_backend = backend.mean(powers, axis=0)
        count = int(x.shape[0])
        mean = convert_to_numpy(mean_backend)
        if count > 1:
            powers -= mean_backend
            powers *= powers
            variance = convert_to_numpy(backend.mean(powers, axis=0))
            variance *= count / (count - 1)
            std = np.sqrt(np.maximum(variance, 0.0))
            sem = std / np.sqrt(float(count))
        else:
            std = np.full(mean.shape, np.nan, dtype=mean.dtype)
            sem = std.copy()
        return mean, std, sem

    def _chunked_power_moments(
        self, x, norm, chunk: int, backend
    ) -> tuple[Any, Any, Any]:
        """Trajectory-chunked FFT with Chan parallel variance combination.

        Peak FFT workspace scales with ``chunk`` trajectories instead of the
        full batch; per-chunk (count, mean, M2) statistics are combined
        exactly on the host, so the result matches the single-pass estimator.
        """
        total = 0
        mean_acc: Any = None
        m2_acc: Any = None
        for start in range(0, int(x.shape[0]), chunk):
            transformed = backend.fft(x[start : start + chunk], axis=-1, norm=norm)
            powers = backend.abs(transformed) ** 2
            del transformed
            n_c = int(powers.shape[0])
            mean_c = convert_to_numpy(backend.mean(powers, axis=0))
            if n_c > 1:
                centered = powers - backend.asarray(mean_c)
                m2_c = convert_to_numpy(backend.mean(centered * centered, axis=0)) * n_c
            else:
                m2_c = np.zeros_like(mean_c)
            del powers
            if total == 0:
                total, mean_acc, m2_acc = n_c, mean_c, m2_c
                continue
            delta = mean_c - mean_acc
            new_total = total + n_c
            mean_acc = mean_acc + delta * (n_c / new_total)
            m2_acc = m2_acc + m2_c + delta * delta * (total * n_c / new_total)
            total = new_total
        mean = mean_acc
        if total > 1:
            std = np.sqrt(np.maximum(m2_acc / (total - 1), 0.0))
            sem = std / np.sqrt(float(total))
        else:
            std = np.full(mean.shape, np.nan, dtype=mean.dtype)
            sem = std.copy()
        return mean, std, sem


class WelchEstimator(_EstimatorMath):
    name: ClassVar[str] = "welch"
    description: ClassVar[str] = "Overlapping-segment Welch PSD estimator"
    config_schema: ClassVar[type[WelchEstimatorConfig]] = WelchEstimatorConfig

    def __init__(self, config: WelchEstimatorConfig | None = None) -> None:
        self.config = config or WelchEstimatorConfig()

    def capabilities(self) -> SpectralEstimatorCapabilities:
        # estimate() materializes the full record before segmenting; do not
        # claim time streaming until an overlap-buffer implementation lands.
        return SpectralEstimatorCapabilities(time_streaming=False)

    def estimate(self, x, dt, convention, backend) -> PsdEstimate:
        del backend
        x = np.asarray(convert_to_numpy(x))
        if x.ndim == 1:
            x = x[None, :]
        count, n_time = x.shape
        nperseg = self.config.nperseg or max(1, n_time // 4)
        noverlap = self.config.noverlap
        noverlap = nperseg // 2 if noverlap is None else noverlap
        nfft = self.config.nfft or nperseg
        if nperseg > n_time:
            nperseg, noverlap = n_time, 0
        step = nperseg - noverlap
        if step <= 0:
            raise ValueError("noverlap must be smaller than nperseg")
        window = self.window(self.config.window, nperseg)
        if self.config.window is None:
            window = np.hanning(nperseg)
        window = window.astype(x.real.dtype if np.iscomplexobj(x) else x.dtype)
        norm: Literal["backward", "ortho", "forward"] | None = (
            "ortho" if convention in {"symmetric", "unitary"} else None
        )
        trajectory_spectra = []
        for trajectory in x:
            segments = []
            for start in range(0, n_time - nperseg + 1, step):
                segment = trajectory[start : start + nperseg] * window
                if nfft > nperseg:
                    segment = np.pad(segment, (0, nfft - nperseg))
                segments.append(np.abs(np.fft.fft(segment, norm=norm)) ** 2)
            trajectory_spectra.append(np.mean(segments, axis=0))
        mean, std, sem = self.trajectory_statistics(np.stack(trajectory_spectra))
        return self.scaled(
            np.fft.fftfreq(nfft, d=dt),
            mean,
            std,
            sem,
            dt=dt,
            convention=convention,
            n_fft=nfft,
            energy=float(np.sum(window * window)),
            count=count,
        )


class MultitaperEstimator(_EstimatorMath):
    name: ClassVar[str] = "multitaper"
    description: ClassVar[str] = "DPSS multitaper PSD estimator"
    config_schema: ClassVar[type[MultitaperEstimatorConfig]] = MultitaperEstimatorConfig

    def __init__(self, config: MultitaperEstimatorConfig | None = None) -> None:
        self.config = config or MultitaperEstimatorConfig()

    def capabilities(self) -> SpectralEstimatorCapabilities:
        return SpectralEstimatorCapabilities()

    def estimate(self, x, dt, convention, backend) -> PsdEstimate:
        del backend
        from scipy.signal.windows import dpss

        x = np.asarray(convert_to_numpy(x))
        if x.ndim == 1:
            x = x[None, :]
        count, n_time = x.shape
        k_tapers = self.config.k_tapers or max(1, int(2 * self.config.nw) - 1)
        tapers = dpss(n_time, self.config.nw, Kmax=k_tapers, sym=False)
        tapers = tapers.astype(x.real.dtype if np.iscomplexobj(x) else x.dtype)
        norm: Literal["backward", "ortho", "forward"] | None = (
            "ortho" if convention in {"symmetric", "unitary"} else None
        )
        trajectory_spectra = []
        for trajectory in x:
            spectra = [
                np.abs(np.fft.fft(trajectory * taper, norm=norm)) ** 2
                for taper in tapers
            ]
            trajectory_spectra.append(np.mean(spectra, axis=0))
        mean, std, sem = self.trajectory_statistics(np.stack(trajectory_spectra))
        return self.scaled(
            np.fft.fftfreq(n_time, d=dt),
            mean,
            std,
            sem,
            dt=dt,
            convention=convention,
            n_fft=n_time,
            energy=1.0,
            count=count,
        )


def create_builtin_estimator(name: str, values: dict[str, Any]):
    """Construct a built-in estimator for direct Python compatibility."""
    implementations: dict[str, Any] = {
        "periodogram": PeriodogramEstimator,
        "welch": WelchEstimator,
        "multitaper": MultitaperEstimator,
    }
    estimator = implementations[name]
    return estimator(config=estimator.config_schema(**values))
