"""Tests for adaptive band-limited coherence-carrier estimation."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ProductSchema,
    SpectralDataset,
    VariableSchema,
)
from qphase_sde.analyser.band_limited_carrier import (
    BandLimitedCarrierAnalyzer,
    BandLimitedCarrierCandidate,
    BandLimitedCarrierConfig,
    BandLimitedCarrierEstimate,
    BandLimitedCarrierPlatform,
    RidgeConditionedCenterConfig,
    _select_platforms,
    estimate_band_limited_carrier,
    track_band_limited_carrier,
)
from qphase_sde.analyser.spectral_ridge import SpectralRidgeConfig
from qphase_sde.products import add_scan_parameter_coordinates


def _lorentzian(
    axis: np.ndarray,
    *,
    center: float,
    hwhm: float,
    amplitude: float = 2.0,
    baseline: float = 0.03,
) -> np.ndarray:
    return baseline + amplitude * hwhm**2 / ((axis - center) ** 2 + hwhm**2)


def _spectral_product(axis, spectra, scan_values, modes):
    spectra = np.asarray(spectra)
    scan_size, _, channels = spectra.shape
    schema = ProductSchema(
        kind=DataKind.SPECTRAL,
        axes=[
            AxisSchema(name="scan", role=AxisRole.PARAMETER, size=scan_size),
            AxisSchema(name="frequency", role=AxisRole.COORDINATE, size=axis.size),
            AxisSchema(name="channel", role=AxisRole.COMPONENT, size=channels),
        ],
        variables=[
            VariableSchema(
                name="axis",
                dtype="float64",
                value_domain="real",
                dims=("scan", "frequency"),
            ),
            VariableSchema(
                name="psd",
                dtype="float64",
                value_domain="real",
                dims=("scan", "frequency", "channel"),
            ),
            VariableSchema(
                name="psd_sem",
                dtype="float64",
                value_domain="real",
                dims=("scan", "frequency", "channel"),
            ),
            VariableSchema(
                name="modes",
                dtype="int64",
                value_domain="real",
                dims=("scan", "channel"),
            ),
        ],
        attributes={
            "frequency_units": "inverse_time",
            "orientation": "phase_decreasing",
            "sidedness": "two_sided",
            "normalization": "density",
            "window": "rectangular",
            "estimator": "synthetic",
            "graph_ready": True,
        },
    )
    product = SpectralDataset.from_arrays(
        schema,
        {
            "axis": np.repeat(axis[None, :], scan_size, axis=0),
            "psd": spectra,
            "psd_sem": np.full(spectra.shape, 0.001),
            "modes": np.repeat(np.asarray(modes)[None, :], scan_size, axis=0),
        },
        owner="test",
    )
    return add_scan_parameter_coordinates(product, {"epsilon": np.asarray(scan_values)})


def test_adaptive_band_recovers_lorentz_carrier_without_profile_fit():
    axis = np.linspace(-4.0, 4.0, 8192, endpoint=False)
    spectrum = _lorentzian(axis, center=-0.7, hwhm=0.08)

    estimate, candidates = estimate_band_limited_carrier(
        axis,
        spectrum,
        freq_min=-2.0,
        freq_max=0.2,
        maximum_lag=100.0,
    )

    assert estimate.status == "ok"
    assert estimate.frequency == pytest.approx(-0.7, abs=2.0e-5)
    assert estimate.selected_half_bandwidth > 0.0
    assert estimate.bandwidth_std >= 0.0
    assert len(candidates) == 9
    assert estimate.platforms
    assert estimate.phase_fit_rms < 1e-3


def test_remote_peak_outside_search_band_does_not_bias_carrier():
    axis = np.linspace(-4.0, 4.0, 8192, endpoint=False)
    spectrum = _lorentzian(axis, center=-0.7, hwhm=0.08)
    spectrum += _lorentzian(
        axis,
        center=1.1,
        hwhm=0.04,
        amplitude=8.0,
        baseline=0.0,
    )

    estimate, _ = estimate_band_limited_carrier(
        axis,
        spectrum,
        freq_min=-2.0,
        freq_max=0.2,
        maximum_lag=100.0,
    )

    assert estimate.status == "ok"
    assert estimate.frequency == pytest.approx(-0.7, abs=2.0e-5)


def test_fixed_ridge_center_recovers_selected_peak_in_multipeak_spectrum():
    axis = np.linspace(-4.0, 4.0, 8192, endpoint=False)
    spectrum = _lorentzian(axis, center=-0.7, hwhm=0.05, amplitude=2.0)
    spectrum += _lorentzian(
        axis,
        center=0.8,
        hwhm=0.06,
        amplitude=6.0,
        baseline=0.0,
    )

    estimate, _ = estimate_band_limited_carrier(
        axis,
        spectrum,
        freq_min=-2.0,
        freq_max=2.0,
        fixed_center=-0.7,
        fixed_bandwidth_max=0.25,
        maximum_lag=100.0,
    )

    assert estimate.status == "ok"
    assert estimate.peak_center == pytest.approx(-0.7)
    assert estimate.frequency == pytest.approx(-0.7, abs=2e-5)
    assert estimate.selected_half_bandwidth <= 0.25


def test_analyzer_exports_trace_rows_and_bandwidth_audit(tmp_path):
    axis = np.linspace(-3.0, 3.0, 4096, endpoint=False)
    spectra = []
    for parameter in (0.1, 0.2):
        mode_0 = _lorentzian(axis, center=-parameter, hwhm=0.07)
        mode_1 = _lorentzian(axis, center=-parameter, hwhm=0.07, amplitude=0.5)
        spectra.append(np.column_stack((mode_0, mode_1)))
    analyzer = BandLimitedCarrierAnalyzer(
        BandLimitedCarrierConfig(
            scan_param="epsilon",
            freq_min=-1.0,
            freq_max=0.2,
            maximum_lag=80.0,
            output_dir=str(tmp_path),
        )
    )

    products = analyzer.execute(
        {"psd": _spectral_product(axis, spectra, [0.1, 0.2], [0, 1])},
        backend=NumpyBackend(),
        label="carrier",
    )
    payload = products["carrier"].attributes["payload_meta"]
    candidates = products["carrier.candidates"].attributes["payload_meta"]
    platforms = products["carrier.platforms"].attributes["payload_meta"]

    assert [row["epsilon"] for row in payload["carrier_rows"]] == [0.1, 0.2]
    assert payload["carrier_rows"][0]["frequency"] == pytest.approx(-0.1, abs=3e-5)
    assert len(candidates["candidate_rows"]) == 18
    assert platforms["platform_rows"]
    assert payload["carrier_rows"][0]["tracked_status"] == "ok"
    assert (tmp_path / "carrier_results.csv").exists()
    assert (tmp_path / "carrier_candidates.csv").exists()
    assert (tmp_path / "carrier_platforms.csv").exists()
    assert payload["orientation"] == "phase_decreasing"


def test_analyzer_builds_typed_result_and_audit_products():
    axis = np.linspace(-3.0, 3.0, 4096, endpoint=False)
    spectrum = _lorentzian(axis, center=-0.1, hwhm=0.07)[None, :, None]
    analyzer = BandLimitedCarrierAnalyzer(
        BandLimitedCarrierConfig(
            scan_param="epsilon", freq_min=-1.0, freq_max=0.2, maximum_lag=80.0
        )
    )
    products = analyzer.execute(
        {"psd": _spectral_product(axis, spectrum, [0.1], [0])},
        backend=NumpyBackend(),
        label="carrier",
    )
    assert set(products) == {
        "carrier",
        "carrier.candidates",
        "carrier.platforms",
    }
    assert all(product.attributes["graph_ready"] for product in products.values())
    assert products["carrier"].axis("row").size == 1


def test_analyzer_writes_header_only_csvs_when_no_candidates(tmp_path):
    axis = np.linspace(-3.0, 3.0, 4096, endpoint=False)
    spectrum = np.full((1, axis.size, 1), 0.03)
    analyzer = BandLimitedCarrierAnalyzer(
        BandLimitedCarrierConfig(
            scan_param="epsilon",
            freq_min=-1.0,
            freq_max=0.2,
            maximum_lag=80.0,
            output_dir=str(tmp_path),
        )
    )

    products = analyzer.execute(
        {"psd": _spectral_product(axis, spectrum, [0.1], [0])},
        backend=NumpyBackend(),
        label="carrier",
    )
    payload = products["carrier"].attributes["payload_meta"]
    candidates_payload = products["carrier.candidates"].attributes["payload_meta"]
    platforms_payload = products["carrier.platforms"].attributes["payload_meta"]

    assert payload["carrier_rows"]
    assert candidates_payload["candidate_rows"] == []
    assert platforms_payload["platform_rows"] == []
    candidates_csv = (
        (tmp_path / "carrier_candidates.csv").read_text(encoding="utf-8").splitlines()
    )
    platforms_csv = (
        (tmp_path / "carrier_platforms.csv").read_text(encoding="utf-8").splitlines()
    )
    assert len(candidates_csv) == 1
    assert "half_bandwidth" in candidates_csv[0]
    assert len(platforms_csv) == 1
    assert "platform_index" in platforms_csv[0]


def test_analyzer_outputs_each_retained_ridge_candidate(tmp_path):
    axis = np.linspace(-3.0, 3.0, 8192, endpoint=False)
    spectra = []
    for shift in (0.0, 0.04, 0.08):
        spectrum = _lorentzian(axis, center=-0.7 + shift, hwhm=0.045, amplitude=2.0)
        spectrum += _lorentzian(
            axis,
            center=0.65 + shift,
            hwhm=0.05,
            amplitude=2.0,
            baseline=0.0,
        )
        spectra.append(spectrum[:, None])
    ridge_config = SpectralRidgeConfig(
        scan_param="epsilon",
        readouts=[0],
        freq_min=-1.2,
        freq_max=1.2,
        smoothing_scale_bins=[2.0, 4.0, 8.0],
        minimum_prominence_fraction=0.01,
        tracking_path_count=2,
    )
    analyzer = BandLimitedCarrierAnalyzer(
        BandLimitedCarrierConfig(
            scan_param="epsilon",
            readout=0,
            freq_min=-1.2,
            freq_max=1.2,
            maximum_lag=80.0,
            center=RidgeConditionedCenterConfig(spectral_ridge=ridge_config),
            output_dir=str(tmp_path),
        )
    )

    products = analyzer.execute(
        {"psd": _spectral_product(axis, spectra, [0.0, 1.0, 2.0], [0])},
        backend=NumpyBackend(),
        label="carrier",
    )
    payload = products["carrier"].attributes["payload_meta"]

    rows = payload["carrier_rows"]
    assert len(rows) >= 6
    assert {row["epsilon"] for row in rows} == {0.0, 1.0, 2.0}
    assert all("ridge_candidate_index" in row for row in rows)
    assert all(row["tracked_status"] == "ridge_conditioned" for row in rows)
    assert all(row["ridge_bandwidth_max"] > 0.0 for row in rows)
    assert all(row["ridge_conditioned_uncertainty_upper"] > 0.0 for row in rows)
    assert all(
        abs(row["ridge_carrier_correction"]) < 2e-4
        for row in rows
        if row["status"] == "ok"
    )
    assert payload["carrier_rows"][0]["ridge_retention_tier"] in {
        "strict",
        "continuity_rescued",
    }


def test_config_requires_ordered_positive_bandwidth_family():
    with pytest.raises(ValueError, match="strictly increasing"):
        BandLimitedCarrierConfig(scan_param="epsilon", bandwidth_multipliers=[1.0, 0.5])


def _candidate(bandwidth, frequency, *, status="ok", rms=0.01):
    return BandLimitedCarrierCandidate(
        half_bandwidth=bandwidth,
        frequency=frequency,
        regression_std=1e-4,
        phase_fit_rms=rms,
        frequency_drift=1e-5,
        decay_rate=0.02,
        lag_points=64,
        lag_start=1.0,
        lag_end=64.0,
        status=status,
    )


def test_unresolved_bandwidth_family_has_no_reference_width_fallback():
    candidates = [
        _candidate(0.01, -0.20),
        _candidate(0.02, -0.17),
        _candidate(0.04, -0.14),
    ]

    platforms, status = _select_platforms(
        candidates,
        consensus_count=3,
        minimum_log_bandwidth_span=0.2,
        stability_fraction=0.01,
        stability_sigma=2.0,
        platform_ambiguity_delta=0.5,
    )

    assert platforms == ()
    assert status == "no_bandwidth_plateau"


def test_nonlinear_phase_outlier_is_excluded_from_bandwidth_platform():
    candidates = [
        _candidate(0.01, -0.1850),
        _candidate(0.02, -0.1738, status="nonlinear_phase", rms=0.3),
        _candidate(0.04, -0.1842),
        _candidate(0.08, -0.1851),
    ]

    platforms, status = _select_platforms(
        candidates,
        consensus_count=3,
        minimum_log_bandwidth_span=0.2,
        stability_fraction=0.02,
        stability_sigma=2.0,
        platform_ambiguity_delta=0.5,
    )

    assert status == "ok"
    assert platforms[0].frequency == pytest.approx(-0.18477, abs=5e-4)
    assert platforms[0].candidate_count == 3


def _platform(frequency, score=0.0):
    return BandLimitedCarrierPlatform(
        frequency=frequency,
        regression_std=1e-4,
        bandwidth_std=1e-4,
        diagnostic_uncertainty=2e-4,
        score=score,
        first_candidate=0,
        last_candidate=3,
        candidate_count=4,
        minimum_half_bandwidth=0.01,
        maximum_half_bandwidth=0.04,
        log_bandwidth_span=np.log(4.0),
        phase_fit_rms=0.01,
        frequency_drift=1e-4,
        decay_rate=0.02,
        lag_start=2.0,
        lag_end=60.0,
        touched_maximum_lag=False,
    )


def test_scan_tracker_uses_continuity_without_target_frequency():
    estimates = [
        BandLimitedCarrierEstimate(
            spectral_width=0.02,
            platforms=(_platform(-0.20),),
            status="ok",
        ),
        BandLimitedCarrierEstimate(
            spectral_width=0.02,
            platforms=(_platform(-0.19, 0.2), _platform(-0.15, 0.0)),
            status="ambiguous_multiband",
        ),
        BandLimitedCarrierEstimate(
            spectral_width=0.02,
            platforms=(_platform(-0.18),),
            status="ok",
        ),
    ]

    tracked = track_band_limited_carrier(
        np.asarray([0.0, 1.0, 2.0]),
        estimates,
        frequency_scale=0.01,
        curvature_weight=1.0,
    )

    assert tracked[1]["tracked_frequency"] == pytest.approx(-0.19)
    assert tracked[1]["tracked_platform_index"] == 0


def test_scan_tracker_rejects_unsupported_curvature_jump():
    estimates = [
        BandLimitedCarrierEstimate(
            spectral_width=0.02,
            platforms=(_platform(-0.20),),
            status="ok",
        ),
        BandLimitedCarrierEstimate(
            spectral_width=0.02,
            platforms=(_platform(-0.19),),
            status="ok",
        ),
        BandLimitedCarrierEstimate(
            spectral_width=0.02,
            platforms=(_platform(-0.12),),
            status="ok",
        ),
    ]

    tracked = track_band_limited_carrier(
        np.asarray([0.0, 1.0, 2.0]),
        estimates,
        frequency_scale=0.01,
        max_normalized_curvature=3.0,
    )

    assert np.isnan(tracked[2]["tracked_frequency"])
    assert tracked[2]["tracked_status"] == "discontinuous_path"


def _typed_psd_product(scan_values: list[float]):
    """Typed PSD scan product with parameter coordinates for execute()."""
    axis = np.linspace(-3.0, 3.0, 1024, endpoint=False)
    payload = []
    for value in scan_values:
        psd = _lorentzian(axis, center=-0.1 - 0.05 * value, hwhm=0.07)
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
    spectra = [point["psd"] for point in payload]
    return _spectral_product(axis, spectra, scan_values, [0])


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


def test_band_limited_carrier_execute_aborts_mid_loop_on_cancellation():
    from types import SimpleNamespace

    psd = _typed_psd_product([0.0, 1.0, 2.0, 3.0, 4.0])
    analyzer = BandLimitedCarrierAnalyzer(
        BandLimitedCarrierConfig(
            scan_param="epsilon", readout=0, maximum_lag=80.0, export=[]
        )
    )
    context = SimpleNamespace(
        cancellation=_FakeCancellation(abort_after=2), progress=None
    )

    with pytest.raises(_LoopAbortError):
        analyzer.execute(
            {"psd": psd},
            backend=NumpyBackend(),
            label="carrier",
            context=context,
        )
    assert context.cancellation.calls < 5


def test_band_limited_carrier_execute_reports_spectrum_progress():
    from types import SimpleNamespace

    psd = _typed_psd_product([0.0, 1.0, 2.0])
    analyzer = BandLimitedCarrierAnalyzer(
        BandLimitedCarrierConfig(
            scan_param="epsilon", readout=0, maximum_lag=80.0, export=[]
        )
    )
    progress = _FakeProgress()
    context = SimpleNamespace(cancellation=None, progress=progress)

    products = analyzer.execute(
        {"psd": psd},
        backend=NumpyBackend(),
        label="carrier",
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


def test_band_limited_tracking_aborts_mid_loop_on_cancellation():
    from types import SimpleNamespace

    estimates = [
        BandLimitedCarrierEstimate(
            spectral_width=0.02,
            platforms=(_platform(-0.20 + 0.01 * index),),
            status="ok",
        )
        for index in range(8)
    ]
    context = SimpleNamespace(
        cancellation=_FakeCancellation(abort_after=3), progress=None
    )

    with pytest.raises(_LoopAbortError):
        track_band_limited_carrier(
            np.asarray([float(index) for index in range(8)]),
            estimates,
            frequency_scale=0.05,
            context=context,
        )
