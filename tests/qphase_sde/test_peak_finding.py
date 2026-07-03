"""Tests for PSD peak finding algorithms."""

from __future__ import annotations

import numpy as np
from qphase_sde.analyser.peak_finding import (
    PeakInfo,
    RationalPeakFinder,
    RationalPeakFinderConfig,
    ScipyPeakFinder,
    ScipyPeakFinderConfig,
)


def _lorentzian(
    axis: np.ndarray, center: float, gamma: float, amplitude: float
) -> np.ndarray:
    return amplitude * (gamma**2 / ((axis - center) ** 2 + gamma**2))


def test_scipy_no_peaks():
    """Flat noise floor below the threshold should yield no peaks."""
    freqs = np.linspace(0.0, 1.0, 100)
    psd = np.ones_like(freqs) * 1e-12
    finder = ScipyPeakFinder(ScipyPeakFinderConfig(noise_threshold=10.0))

    info = finder.find_peaks(freqs, psd)

    assert info.indices == []
    assert info.frequencies == []
    assert info.values == []


def test_scipy_single_peak():
    """A single Lorentzian peak should be detected."""
    freqs = np.linspace(-2.0, 2.0, 401)
    psd = _lorentzian(freqs, center=0.5, gamma=0.1, amplitude=5.0) + 0.01
    finder = ScipyPeakFinder(ScipyPeakFinderConfig(noise_threshold=2.0))

    info = finder.find_peaks(freqs, psd)

    assert len(info.indices) == 1
    assert np.isclose(info.frequencies[0], 0.5, atol=0.05)
    assert info.values[0] > 1.0


def test_scipy_multiple_peaks():
    """Multiple well-separated peaks should all be detected."""
    freqs = np.linspace(-3.0, 3.0, 601)
    psd = (
        _lorentzian(freqs, center=-1.5, gamma=0.1, amplitude=4.0)
        + _lorentzian(freqs, center=0.2, gamma=0.08, amplitude=6.0)
        + _lorentzian(freqs, center=1.8, gamma=0.12, amplitude=3.0)
        + 0.01
    )
    finder = ScipyPeakFinder(ScipyPeakFinderConfig(noise_threshold=2.0))

    info = finder.find_peaks(freqs, psd)

    assert len(info.indices) == 3
    detected = sorted(info.frequencies)
    assert np.isclose(detected[0], -1.5, atol=0.1)
    assert np.isclose(detected[1], 0.2, atol=0.1)
    assert np.isclose(detected[2], 1.8, atol=0.1)


def test_scipy_max_peaks_filter():
    """max_peaks should limit the number of returned peaks."""
    freqs = np.linspace(-3.0, 3.0, 601)
    psd = (
        _lorentzian(freqs, center=-1.5, gamma=0.1, amplitude=4.0)
        + _lorentzian(freqs, center=0.2, gamma=0.08, amplitude=6.0)
        + _lorentzian(freqs, center=1.8, gamma=0.12, amplitude=3.0)
        + 0.01
    )
    finder = ScipyPeakFinder(ScipyPeakFinderConfig(noise_threshold=2.0, max_peaks=2))

    info = finder.find_peaks(freqs, psd)

    assert len(info.indices) == 2


def test_rational_single_peak():
    """Rational finder should detect a dominant peak in a smooth spectrum."""
    freqs = np.linspace(-2.0, 2.0, 201)
    psd = _lorentzian(freqs, center=0.6, gamma=0.15, amplitude=4.0) + 0.02
    finder = RationalPeakFinder(
        RationalPeakFinderConfig(num_order=2, den_order=4, parity="even")
    )

    info = finder.find_peaks(freqs, psd)

    assert len(info.indices) >= 1
    assert any(np.isclose(abs(float(f)), 0.6, atol=0.2) for f in info.frequencies)


def test_rational_empty_on_invalid_data():
    """Rational finder should return empty info when the fit cannot proceed."""
    freqs = np.linspace(-1.0, 1.0, 10)
    psd = np.full_like(freqs, np.nan)
    finder = RationalPeakFinder(RationalPeakFinderConfig())

    info = finder.find_peaks(freqs, psd)

    assert info.indices == []
    assert info.frequencies == []
    assert info.values == []


def test_peak_info_serialization_no_numpy_or_complex():
    """PeakInfo.model_dump() must not leave numpy arrays or complex numbers."""
    info = PeakInfo(
        indices=[1, 2],
        frequencies=[0.1, 0.2],
        values=[1.0, 2.0],
        properties={
            "array": np.array([1, 2, 3]),
            "nested": {"complex": 1 + 2j, "values": np.array([0.5, 1.5])},
            "scalar": np.float64(3.14),
        },
    )

    dumped = info.model_dump()
    props = dumped["properties"]

    assert props["array"] == [1, 2, 3]
    assert props["nested"]["complex"] == {"real": 1.0, "imag": 2.0}
    assert props["nested"]["values"] == [0.5, 1.5]
    assert props["scalar"] == 3.14

    # Ensure no numpy or complex types remain
    def _no_special_types(obj):
        if isinstance(obj, dict):
            return all(_no_special_types(v) for v in obj.values())
        if isinstance(obj, list):
            return all(_no_special_types(v) for v in obj)
        return not isinstance(obj, np.ndarray | np.complexfloating | complex)

    assert _no_special_types(props)


def test_scipy_properties_serialized():
    """Scipy finder returns ndarray properties; model_dump should convert them."""
    freqs = np.linspace(-2.0, 2.0, 401)
    psd = _lorentzian(freqs, center=0.5, gamma=0.1, amplitude=5.0) + 0.01
    finder = ScipyPeakFinder(ScipyPeakFinderConfig(noise_threshold=2.0))

    info = finder.find_peaks(freqs, psd)
    dumped = info.model_dump()

    assert len(dumped["indices"]) == 1
    # SciPy find_peaks properties contain arrays; they must be lists now.
    for value in dumped["properties"].values():
        assert not isinstance(value, np.ndarray)
