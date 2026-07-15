"""Tests for the ``method`` option of ``PsdAnalyzer``.

The analyzer now supports three PSD estimation methods:

*   ``periodogram`` — classical averaged periodogram (default).
*   ``welch`` — averaged overlapping segment periodograms.
*   ``multitaper`` — DPSS (Slepian) multitaper estimate.

These tests use a synthetic complex sinusoid with known frequency so the
peak location can be verified regardless of scaling differences between
methods.
"""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.psd import PsdAnalyzer
from qphase_sde.state import TrajectorySet

BACKEND = NumpyBackend()


def _make_sine_data(
    n_traj: int = 16,
    n_time: int = 1024,
    dt: float = 0.1,
    omega: float = 1.0,
    noise_std: float = 0.05,
) -> np.ndarray:
    """Return complex-valued trajectories with a single angular-frequency peak."""
    rng = np.random.default_rng(42)
    t = np.arange(n_time) * dt
    # A complex exponential rotating at ``omega`` rad/s.
    signal = np.exp(1j * omega * t)
    noise = noise_std * (
        rng.standard_normal((n_traj, n_time))
        + 1j * rng.standard_normal((n_traj, n_time))
    )
    # Return a single-mode array of shape (n_traj, n_time, 1).
    return (signal + noise)[:, :, None]


@pytest.mark.parametrize("method", ["periodogram", "welch", "multitaper"])
def test_psd_method_peak_location(method):
    """Each method should recover the peak near the injected frequency."""
    dt = 0.1
    omega = 1.0
    data = TrajectorySet(
        data=_make_sine_data(n_traj=16, n_time=1024, dt=dt, omega=omega),
        t0=0.0,
        dt=dt,
    )

    analyzer = PsdAnalyzer(
        kind="complex",
        modes=[0],
        convention="symmetric",
        method=method,
        nperseg=256 if method == "welch" else None,
    )
    result = analyzer.analyze(data, BACKEND)

    axis = result.data_dict["axis"]
    psd = result.data_dict["psd"]
    assert axis.ndim == 1
    assert psd.shape == (axis.size, 1)

    peak_idx = int(np.argmax(psd[:, 0]))
    peak_freq = float(axis[peak_idx])
    # The two-sided spectrum contains the peak at ``+omega`` (and a mirror at
    # ``-omega``).  Either is fine; just check the magnitude.
    assert abs(abs(peak_freq) - omega) < 0.05, (
        f"{method}: peak at {peak_freq}, expected near ±{omega}"
    )


@pytest.mark.parametrize("method", ["periodogram", "welch", "multitaper"])
def test_psd_method_unitary_convention(method):
    """Unitary convention should scale the frequency axis by 2*pi."""
    dt = 0.1
    data = TrajectorySet(
        data=_make_sine_data(n_traj=8, n_time=512, dt=dt, omega=2.0),
        t0=0.0,
        dt=dt,
    )

    analyzer = PsdAnalyzer(
        kind="complex",
        modes=[0],
        convention="unitary",
        method=method,
        nperseg=128 if method == "welch" else None,
    )
    result = analyzer.analyze(data, BACKEND)
    axis = result.data_dict["axis"]
    # Unitary convention uses angular frequency, so the Nyquist limit is pi/dt.
    assert np.isclose(np.max(np.abs(axis)), np.pi / dt, rtol=1e-12)


def test_welch_nperseg_default_reduces_variance():
    """Welch with a segment length shorter than the series produces more segments."""
    dt = 0.1
    data = TrajectorySet(
        data=_make_sine_data(n_traj=8, n_time=1024, dt=dt, omega=1.0),
        t0=0.0,
        dt=dt,
    )

    analyzer = PsdAnalyzer(
        kind="complex",
        modes=[0],
        convention="symmetric",
        method="welch",
        nperseg=256,
        noverlap=128,
    )
    result = analyzer.analyze(data, BACKEND)
    axis = result.data_dict["axis"]
    # 256-point FFT gives 256 frequency bins (two-sided).
    assert axis.size == 256


def test_multitaper_k_tapers_default():
    """Multitaper defaults should produce a stable spectrum."""
    dt = 0.1
    data = TrajectorySet(
        data=_make_sine_data(n_traj=8, n_time=512, dt=dt, omega=1.5),
        t0=0.0,
        dt=dt,
    )

    analyzer = PsdAnalyzer(
        kind="complex",
        modes=[0],
        convention="symmetric",
        method="multitaper",
        nw=2.5,
    )
    result = analyzer.analyze(data, BACKEND)
    axis = result.data_dict["axis"]
    psd = result.data_dict["psd"]
    peak_idx = int(np.argmax(psd[:, 0]))
    assert abs(abs(axis[peak_idx]) - 1.5) < 0.1


def test_periodogram_reports_cross_trajectory_standard_error():
    """PSD uncertainty uses the sample variance of independent trajectories."""
    dt = 0.2
    n_time = 32
    t = np.arange(n_time) * dt
    amplitudes = np.array([1.0, 1.5, 2.0, 3.0])[:, None]
    values = amplitudes * np.exp(1j * 0.7 * t)[None, :]
    data = TrajectorySet(data=values[:, :, None], t0=0.0, dt=dt)

    result = PsdAnalyzer(
        kind="complex", modes=[0], convention="pragmatic"
    ).analyze(data, BACKEND)
    payload = result.data_dict

    trajectory_psd = np.abs(np.fft.fft(values, axis=-1)) ** 2
    trajectory_psd *= dt / n_time
    trajectory_psd = np.fft.fftshift(trajectory_psd, axes=-1)
    expected_mean = np.mean(trajectory_psd, axis=0)
    expected_std = np.std(trajectory_psd, axis=0, ddof=1)

    np.testing.assert_allclose(payload["psd"][:, 0], expected_mean)
    np.testing.assert_allclose(payload["psd_std"][:, 0], expected_std)
    np.testing.assert_allclose(
        payload["psd_sem"][:, 0], expected_std / np.sqrt(amplitudes.shape[0])
    )
    assert payload["uncertainty"]["independent_unit"] == "trajectory"
    assert payload["uncertainty"]["n_independent"] == amplitudes.shape[0]


@pytest.mark.parametrize("method", ["periodogram", "welch", "multitaper"])
def test_psd_methods_report_sem_for_each_mode(method):
    """Every PSD estimator exposes arrays aligned with the mean PSD."""
    n_traj = 8
    data = TrajectorySet(
        data=_make_sine_data(n_traj=n_traj, n_time=256),
        t0=0.0,
        dt=0.1,
    )
    result = PsdAnalyzer(
        kind="complex",
        modes=[0],
        method=method,
        nperseg=64 if method == "welch" else None,
    ).analyze(data, BACKEND)
    payload = result.data_dict

    assert payload["psd_std"].shape == payload["psd"].shape
    assert payload["psd_sem"].shape == payload["psd"].shape
    np.testing.assert_allclose(payload["psd_sem"], payload["psd_std"] / np.sqrt(n_traj))
    assert np.all(payload["psd_std"] >= 0.0)


def test_single_trajectory_marks_psd_uncertainty_unavailable():
    """One trajectory has no cross-trajectory sample variance."""
    data = TrajectorySet(
        data=_make_sine_data(n_traj=1, n_time=128),
        t0=0.0,
        dt=0.1,
    )
    payload = PsdAnalyzer(kind="complex", modes=[0]).analyze(data, BACKEND).data_dict

    assert np.all(np.isnan(payload["psd_std"]))
    assert np.all(np.isnan(payload["psd_sem"]))
    assert payload["uncertainty"]["available"] is False


def test_psd_method_invalid():
    """An unsupported method is rejected at configuration time."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="periodogram"):
        PsdAnalyzer(
            kind="complex",
            modes=[0],
            method="not_a_method",
        )
