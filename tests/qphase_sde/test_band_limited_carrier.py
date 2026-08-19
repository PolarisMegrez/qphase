"""Tests for adaptive band-limited coherence-carrier estimation."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.band_limited_carrier import (
    BandLimitedCarrierAnalyzer,
    BandLimitedCarrierConfig,
    estimate_band_limited_carrier,
)
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
    assert len(candidates) == 5


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


def test_analyzer_exports_trace_rows_and_bandwidth_audit(tmp_path):
    axis = np.linspace(-3.0, 3.0, 4096, endpoint=False)
    results: dict[str, SDEResult] = {}
    for index, parameter in enumerate((0.1, 0.2)):
        mode_0 = _lorentzian(axis, center=-parameter, hwhm=0.07)
        mode_1 = _lorentzian(
            axis, center=-parameter, hwhm=0.07, amplitude=0.5
        )
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
    assert len(payload["candidate_rows"]) == 10
    assert (tmp_path / "carrier_results.csv").exists()
    assert (tmp_path / "carrier_candidates.csv").exists()
    assert payload["orientation"] == "phase_decreasing"


def test_config_requires_ordered_positive_bandwidth_family():
    with pytest.raises(ValueError, match="strictly increasing"):
        BandLimitedCarrierConfig(
            scan_param="epsilon", bandwidth_multipliers=[1.0, 0.5]
        )
