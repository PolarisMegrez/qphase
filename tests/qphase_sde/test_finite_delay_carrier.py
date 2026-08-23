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


def test_finite_delay_carrier_analyzer_evaluates_multiple_readouts():
    axis = np.linspace(-3.0, 3.0, 4096, endpoint=False)
    spectrum = np.column_stack(
        [
            np.exp(-0.5 * ((axis + 0.4) / 0.06) ** 2),
            np.exp(-0.5 * ((axis - 0.7) / 0.08) ** 2),
        ]
    )
    result = SDEResult(
        analysis={
            "psd": {
                "axis": axis,
                "psd": spectrum,
                "modes": [0, 1],
                "orientation": "phase_decreasing",
            }
        },
        meta={"params": {"epsilon": 0.0}},
    )
    analyzer = FiniteDelayCarrierAnalyzer(
        FiniteDelayCarrierConfig(
            scan_param="epsilon",
            readouts=[0, 1],
            detector_rates=[0.1],
        )
    )

    payload = analyzer.analyze({"point": result}, NumpyBackend()).data_dict

    assert [row["measurement_name"] for row in payload["carrier_rows"]] == [
        "mode_0",
        "mode_1",
    ]
    np.testing.assert_allclose(
        [row["frequency"] for row in payload["carrier_rows"]],
        [-0.4, 0.7],
        atol=2e-5,
    )
    assert payload["carrier_rows"][0]["measurement_kind"] == "bare_mode"
