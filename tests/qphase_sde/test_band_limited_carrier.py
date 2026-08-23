"""Tests for adaptive band-limited coherence-carrier estimation."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
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
from qphase_sde.result import SDEResult


def _lorentzian(
    axis: np.ndarray,
    *,
    center: float,
    hwhm: float,
    amplitude: float = 2.0,
    baseline: float = 0.03,
) -> np.ndarray:
    return baseline + amplitude * hwhm**2 / ((axis - center) ** 2 + hwhm**2)


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
    results: dict[str, SDEResult] = {}
    for index, parameter in enumerate((0.1, 0.2)):
        mode_0 = _lorentzian(axis, center=-parameter, hwhm=0.07)
        mode_1 = _lorentzian(axis, center=-parameter, hwhm=0.07, amplitude=0.5)
        results[f"point_{index}"] = SDEResult(
            analysis={
                "psd": {
                    "axis": axis,
                    "psd": np.column_stack((mode_0, mode_1)),
                    "modes": [0, 1],
                    "orientation": "phase_decreasing",
                }
            },
            meta={"params": {"epsilon": parameter}},
        )
    analyzer = BandLimitedCarrierAnalyzer(
        BandLimitedCarrierConfig(
            scan_param="epsilon",
            freq_min=-1.0,
            freq_max=0.2,
            maximum_lag=80.0,
            output_dir=str(tmp_path),
        )
    )

    payload = analyzer.analyze(results, NumpyBackend()).data_dict

    assert [row["epsilon"] for row in payload["carrier_rows"]] == [0.1, 0.2]
    assert payload["carrier_rows"][0]["frequency"] == pytest.approx(-0.1, abs=3e-5)
    assert len(payload["candidate_rows"]) == 18
    assert payload["platform_rows"]
    assert payload["carrier_rows"][0]["tracked_status"] == "ok"
    assert (tmp_path / "carrier_results.csv").exists()
    assert (tmp_path / "carrier_candidates.csv").exists()
    assert (tmp_path / "carrier_platforms.csv").exists()
    assert payload["orientation"] == "phase_decreasing"


def test_analyzer_outputs_each_retained_ridge_candidate(tmp_path):
    axis = np.linspace(-3.0, 3.0, 8192, endpoint=False)
    results: dict[str, SDEResult] = {}
    for index, shift in enumerate((0.0, 0.04, 0.08)):
        spectrum = _lorentzian(axis, center=-0.7 + shift, hwhm=0.045, amplitude=2.0)
        spectrum += _lorentzian(
            axis,
            center=0.65 + shift,
            hwhm=0.05,
            amplitude=2.0,
            baseline=0.0,
        )
        results[f"point_{index}"] = SDEResult(
            analysis={
                "psd": {
                    "axis": axis,
                    "psd": spectrum[:, None],
                    "psd_sem": np.full((axis.size, 1), 0.001),
                    "modes": [0],
                    "orientation": "phase_decreasing",
                }
            },
            meta={"params": {"epsilon": float(index)}},
        )
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

    payload = analyzer.analyze(results, NumpyBackend()).data_dict

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
