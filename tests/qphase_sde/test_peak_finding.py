"""Tests for PSD peak finding algorithms."""

from __future__ import annotations

import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.peaks import PeaksAnalyzer, PeaksAnalyzerConfig
from qphase_sde.peak_finder import (
    PeakInfo,
    RationalPeakFinder,
    RationalPeakFinderConfig,
    ScipyPeakFinder,
    ScipyPeakFinderConfig,
    TypedPeakFinder,
)
from qphase_sde.result import SDEResult


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


def test_builtin_finder_emits_grouped_typed_candidate_table():
    freqs = np.linspace(-1.0, 1.0, 101)
    power = np.vstack(
        [np.exp(-(((freqs - 0.2) / 0.08) ** 2)), np.exp(-(((freqs + 0.3) / 0.1) ** 2))]
    )
    finder = ScipyPeakFinder(ScipyPeakFinderConfig(noise_threshold=None, max_peaks=1))
    assert isinstance(finder, TypedPeakFinder)
    table = finder.find_candidate_table(
        freqs,
        power,
        scan_index=0,
        scan_count=1,
        channel_names=["mode_0", "mode_1"],
    )
    assert table["candidate_offsets"].tolist() == [0, 1, 2]
    assert table["group_scan_index"].tolist() == [0, 0]
    assert table["group_channel_index"].tolist() == [0, 1]
    assert table["columns"]["sample_index"].dtype == np.int64
    assert table["columns"]["status_code"].tolist() == [3, 3]


def test_peaks_analyser_publishes_grouped_product():
    axis = np.linspace(-1.0, 1.0, 101)
    inputs = {
        f"point_{index}": SDEResult(
            analysis={
                "psd": {
                    "axis": axis,
                    "psd": np.exp(-(((axis - 0.2 - 0.1 * index) / 0.08) ** 2))[:, None],
                    "modes": [0],
                    "orientation": "phase_decreasing",
                }
            },
            meta={"params": {"epsilon": float(index)}},
        )
        for index in range(2)
    }
    analyzer = PeaksAnalyzer(PeaksAnalyzerConfig(scan_param="epsilon", finder="scipy"))
    payload = analyzer.analyze(inputs, NumpyBackend()).data_dict
    products = analyzer.build_products(payload, scan_size=2, label="peaks")
    assert products is not None
    product = products["peaks"]
    assert product.attributes["graph_ready"] is True
    assert product.axis("scan").size == 2
    assert product.axis("group").size == 2


def test_peak_paths_analyser_publishes_typed_path_product():
    from qphase_sde.analyser.peak_paths import (
        PeakPathsAnalyzer,
        PeakPathsAnalyzerConfig,
    )
    from qphase_sde.peak_tracker import RidgePeakTrackerConfig
    from qphase_sde.result import SDEDataBundle, SDEProvenance

    axis = np.linspace(-1.0, 1.0, 101)
    peak_analyzer = PeaksAnalyzer(
        PeaksAnalyzerConfig(scan_param="epsilon", finder="scipy")
    )
    payload = peak_analyzer.analyze(
        {
            f"point_{index}": SDEResult(
                analysis={
                    "psd": {
                        "axis": axis,
                        "psd": np.exp(-(((axis - 0.2 - 0.1 * index) / 0.08) ** 2))[
                            :, None
                        ],
                        "modes": [0],
                    }
                },
                meta={"params": {"epsilon": float(index)}},
            )
            for index in range(2)
        },
        NumpyBackend(),
    ).data_dict
    candidates = peak_analyzer.build_products(payload, scan_size=2, label="peaks")
    assert candidates is not None
    source = SDEDataBundle(
        candidates,
        SDEProvenance(dt=1.0),
        scan_axes={"epsilon": np.asarray([0.0, 1.0])},
        scan_shape=(2,),
    )
    analyzer = PeakPathsAnalyzer(
        PeakPathsAnalyzerConfig(
            scan_param="epsilon",
            tracker=RidgePeakTrackerConfig(path_count=1),
        )
    )
    result = analyzer.analyze(source, NumpyBackend())
    products = analyzer.build_products(result.data_dict, scan_size=None, label="paths")

    assert products is not None
    product = products["paths"]
    assert product.attributes["graph_ready"] is True
    assert product.schema.attributes["table"] == "peak_paths"
    assert product.axis("path").size == 1
