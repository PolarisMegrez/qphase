from pathlib import Path

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.finite_delay_carrier import (
    FiniteDelayCarrierAnalyzer,
    FiniteDelayCarrierConfig,
    finite_delay_carrier_from_spectrum,
)
from qphase_sde.result import SDEResult


def test_finite_delay_carrier_recovers_single_frequency_for_all_rates():
    axis = np.linspace(-4.0, 4.0, 8192, endpoint=False)
    center = -0.7
    spectrum = np.exp(-0.5 * ((axis - center) / 0.08) ** 2)
    rates = np.asarray([0.01, 0.1, 1.0])

    result = finite_delay_carrier_from_spectrum(
        axis,
        spectrum,
        rates,
        maximum_lag=500.0,
        tail_time_constants=12.0,
    )

    np.testing.assert_allclose(result["frequency"], center, atol=2e-5)
    assert result["instantaneous_frequency"] == pytest.approx(center, abs=2e-5)


def test_finite_delay_carrier_analyzer_exports_scan_rows(tmp_path: Path):
    axis = np.linspace(-3.0, 3.0, 4096, endpoint=False)
    results = {}
    for index, center in enumerate((-0.3, -0.2)):
        spectrum = np.exp(-0.5 * ((axis - center) / 0.06) ** 2)
        results[f"point_{index}"] = SDEResult(
            analysis={
                "psd": {
                    "axis": axis,
                    "psd": spectrum[:, None],
                    "modes": [0],
                    "orientation": "phase_decreasing",
                }
            },
            meta={"params": {"epsilon": float(index)}},
        )
    analyzer = FiniteDelayCarrierAnalyzer(
        FiniteDelayCarrierConfig(
            scan_param="epsilon",
            detector_rates=[0.1, 1.0],
            output_dir=str(tmp_path),
        )
    )

    payload = analyzer.analyze(results, NumpyBackend()).data_dict

    assert len(payload["carrier_rows"]) == 4
    np.testing.assert_allclose(
        [row["frequency"] for row in payload["carrier_rows"]],
        [-0.3, -0.3, -0.2, -0.2],
        atol=2e-5,
    )
    assert (tmp_path / "finite_delay_carrier.csv").exists()
