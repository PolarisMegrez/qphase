"""Tests for PSD peak finding algorithms."""

from __future__ import annotations

import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.peaks import PeaksAnalyzer, PeaksAnalyzerConfig
from qphase_sde.analyser.psd import PsdAnalyzer, _build_psd_products
from qphase_sde.peak_finder import (
    PeakInfo,
    RationalPeakFinder,
    RationalPeakFinderConfig,
    ScipyPeakFinder,
    ScipyPeakFinderConfig,
    TypedPeakFinder,
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
    analyzer = PeaksAnalyzer(
        PeaksAnalyzerConfig(scan_param="epsilon"),
        subplugins={"finder": ScipyPeakFinder()},
    )
    products = analyzer.execute(
        {"psd": _typed_psd_product([0.0, 1.0], points=101)},
        backend=NumpyBackend(),
        label="peaks",
    )
    product = products["peaks"]
    assert product.attributes["graph_ready"] is True
    assert product.axis("scan").size == 2
    assert product.axis("group").size == 2
    np.testing.assert_array_equal(
        product.handle("sample_index").materialize(),
        np.asarray([60, 65]),
    )
    np.testing.assert_array_equal(
        product.handle("status_code").materialize(),
        np.asarray([3, 3]),
    )


def test_peaks_analyser_resolves_finder_through_registry():
    from qphase.core.registry import registry

    class RegistryFinder(ScipyPeakFinder):
        name = "registry_test"

    registry.register("peak_finder", "registry_test", RegistryFinder, overwrite=True)
    analyzer = PeaksAnalyzer(
        PeaksAnalyzerConfig(scan_param="epsilon"),
        subplugins={"finder": RegistryFinder()},
    )
    result = analyzer.execute(
        {"psd": _typed_psd_product([0.0], points=101)},
        backend=NumpyBackend(),
        label="peaks",
    )["peaks"]
    assert result.axis("group").size == 1


def test_peak_paths_analyser_publishes_typed_path_product():
    from qphase_sde.analyser.peak_paths import (
        PeakPathsAnalyzer,
        PeakPathsAnalyzerConfig,
    )
    from qphase_sde.peak_tracker import RidgePeakTracker, RidgePeakTrackerConfig
    peak_analyzer = PeaksAnalyzer(
        PeaksAnalyzerConfig(scan_param="epsilon"),
        subplugins={"finder": ScipyPeakFinder()},
    )
    candidates = peak_analyzer.execute(
        {"psd": _typed_psd_product([0.0, 1.0], points=101)},
        backend=NumpyBackend(),
        label="peaks",
    )
    analyzer = PeakPathsAnalyzer(
        PeakPathsAnalyzerConfig(
            scan_param="epsilon",
            tracker=RidgePeakTrackerConfig(path_count=1).model_dump(),
        ),
        subplugins={"tracker": RidgePeakTracker(path_count=1)},
    )
    products = analyzer.execute(
        {"peaks": candidates["peaks"]},
        backend=NumpyBackend(),
        label="paths",
    )

    assert products is not None
    product = products["paths"]
    assert product.attributes["graph_ready"] is True
    assert product.schema.attributes["table"] == "peak_paths"
    assert product.axis("path").size == 1


def _typed_psd_product(scan_values: list[float], *, points: int = 401):
    """Typed PSD scan product for execute()."""
    axis = np.linspace(-1.0, 1.0, points)
    payload = []
    for value in scan_values:
        psd = np.exp(-(((axis - 0.2 - 0.1 * value) / 0.08) ** 2))
        payload.append(
            {
                "axis": axis,
                "psd": psd[:, None],
                "psd_std": np.full((axis.size, 1), 0.01),
                "psd_sem": np.full((axis.size, 1), 0.001),
                "modes": [0],
                "orientation": "phase_decreasing",
                "uncertainty": {"n_independent": 4},
            }
        )
    analyzer = PsdAnalyzer(modes=[0], kind="complex")
    products = _build_psd_products(
        payload[0] if len(payload) == 1 else payload,
        scan_size=len(scan_values),
        label="psd",
        analyser=analyzer,
    )
    assert products is not None
    return products["psd"]


class _LoopAbortError(Exception):
    """Raised by the fake cancellation token mid-loop."""


class _FakeCancellation:
    """Cancellation token that raises after a fixed number of polls."""

    def __init__(self, abort_after: int) -> None:
        self.calls = 0
        self._abort_after = abort_after

    def raise_if_cancelled(self) -> None:
        self.calls += 1
        if self.calls > self._abort_after:
            raise _LoopAbortError


class _FakeProgress:
    """Progress reporter recording structured updates."""

    def __init__(self) -> None:
        self.updates: list[dict] = []

    def update(
        self,
        completed=None,
        total=None,
        *,
        unit=None,
        stage=None,
        message="",
        metadata=None,
    ):
        self.updates.append(
            {
                "completed": completed,
                "total": total,
                "unit": unit,
                "stage": stage,
                "message": message,
            }
        )


def test_peaks_execute_aborts_mid_loop_on_cancellation():
    from types import SimpleNamespace

    import pytest

    analyzer = PeaksAnalyzer(
        PeaksAnalyzerConfig(scan_param="epsilon"),
        subplugins={"finder": ScipyPeakFinder()},
    )
    context = SimpleNamespace(
        cancellation=_FakeCancellation(abort_after=2), progress=None
    )

    with pytest.raises(_LoopAbortError):
        analyzer.execute(
            {"psd": _typed_psd_product([0.0, 1.0, 2.0, 3.0, 4.0])},
            backend=NumpyBackend(),
            label="peaks",
            context=context,
        )
    assert context.cancellation.calls < 5


def test_peaks_execute_reports_spectrum_progress():
    from types import SimpleNamespace

    analyzer = PeaksAnalyzer(
        PeaksAnalyzerConfig(scan_param="epsilon"),
        subplugins={"finder": ScipyPeakFinder()},
    )
    progress = _FakeProgress()
    context = SimpleNamespace(cancellation=None, progress=progress)

    products = analyzer.execute(
        {"psd": _typed_psd_product([0.0, 1.0, 2.0])},
        backend=NumpyBackend(),
        label="peaks",
        context=context,
    )

    assert products
    assert progress.updates
    completed = [update["completed"] for update in progress.updates]
    assert completed == sorted(completed)
    assert all(update["unit"] == "spectrum" for update in progress.updates)
    assert all(
        str(update["stage"]).startswith("analyze") for update in progress.updates
    )
    assert all(update["total"] is None for update in progress.updates)
