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


def _typed_psd_product(scan_values: list[float]):
    """Typed PSD scan product with parameter coordinates for execute()."""
    from qphase_sde.analyser.psd import PsdAnalyzer
    from qphase_sde.products import add_scan_parameter_coordinates

    axis = np.linspace(-3.0, 3.0, 1024, endpoint=False)
    payload = []
    for value in scan_values:
        psd = 0.03 + 2.0 * 0.08**2 / ((axis + 0.1 + 0.05 * value) ** 2 + 0.08**2)
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
    products = PsdAnalyzer(modes=[0], kind="complex").build_products(
        payload, scan_size=len(scan_values), label="psd"
    )
    assert products is not None
    return add_scan_parameter_coordinates(
        products["psd"], {"epsilon": np.asarray(scan_values, dtype=float)}
    )


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


def test_finite_delay_carrier_execute_aborts_mid_loop_on_cancellation():
    from types import SimpleNamespace

    psd = _typed_psd_product([0.0, 1.0, 2.0, 3.0, 4.0])
    analyzer = FiniteDelayCarrierAnalyzer(
        FiniteDelayCarrierConfig(scan_param="epsilon", readout=0, export=[])
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


def test_finite_delay_carrier_execute_reports_spectrum_progress():
    from types import SimpleNamespace

    psd = _typed_psd_product([0.0, 1.0, 2.0])
    analyzer = FiniteDelayCarrierAnalyzer(
        FiniteDelayCarrierConfig(scan_param="epsilon", readout=0, export=[])
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
