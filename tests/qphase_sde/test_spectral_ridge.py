from pathlib import Path

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.spectral_ridge import (
    SpectralRidgeAnalyzer,
    SpectralRidgeConfig,
    _tracking_segments,
    estimate_spectral_ridges,
)
from qphase_sde.result import SDEResult


def _gaussian(axis: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((axis - center) / width) ** 2)


def test_scale_space_ridge_recovers_smooth_peak():
    axis = np.linspace(-2.0, 2.0, 4001)
    spectrum = 0.02 + 3.0 * _gaussian(axis, 0.37, 0.12)
    sem = np.full(axis.shape, 0.002)

    estimates = estimate_spectral_ridges(
        axis,
        spectrum,
        sem=sem,
        smoothing_scale_bins=[2.0, 4.0, 8.0],
    )

    estimate = estimates[0]
    assert estimate.frequency == pytest.approx(0.37, abs=2e-3)
    assert estimate.frequency_std < 2e-3
    assert estimate.curvature > 0.0
    assert estimate.curvature_significance > 2.0
    assert estimate.scale_support == 3
    assert estimate.plateau_lower < estimate.frequency < estimate.plateau_upper


def test_spectral_ridge_analyzer_handles_multiple_readouts(tmp_path: Path):
    axis = np.linspace(-2.0, 2.0, 4001)
    results = {}
    for point, shift in enumerate((0.0, 0.1)):
        spectrum = np.column_stack(
            [
                0.01 + _gaussian(axis, -0.4 + shift, 0.08),
                0.02 + 0.7 * _gaussian(axis, 0.6 + shift, 0.1),
            ]
        )
        results[f"point_{point}"] = SDEResult(
            analysis={
                "psd": {
                    "axis": axis,
                    "psd": spectrum,
                    "psd_sem": np.full(spectrum.shape, 0.001),
                    "modes": [0, 1],
                    "orientation": "phase_decreasing",
                }
            },
            meta={"params": {"epsilon": float(point)}},
        )
    analyzer = SpectralRidgeAnalyzer(
        SpectralRidgeConfig(
            scan_param="epsilon",
            readouts=[0, 1],
            smoothing_scale_bins=[2.0, 4.0, 8.0],
            output_dir=str(tmp_path),
        )
    )

    payload = analyzer.analyze(results, NumpyBackend()).data_dict

    assert len(payload["ridge_rows"]) == 4
    mode_0 = [
        row for row in payload["ridge_rows"] if row["measurement_name"] == "mode_0"
    ]
    np.testing.assert_allclose(
        [row["frequency"] for row in mode_0], [-0.4, -0.3], atol=2e-3
    )
    assert all(row["status"] == "ok" for row in payload["ridge_rows"])
    assert (tmp_path / "spectral_ridge.csv").exists()
    assert (tmp_path / "spectral_ridge_candidates.csv").exists()


def test_spectral_ridge_rejects_duplicate_readouts():
    with pytest.raises(ValueError, match="unique"):
        SpectralRidgeConfig(scan_param="epsilon", readouts=[0, 0])


def test_tracking_segments_split_explicit_scan_gap():
    values = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])

    segments = _tracking_segments(values, 1.5)

    assert [(item.start, item.stop) for item in segments] == [(0, 3), (3, 6)]
