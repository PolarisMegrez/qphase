"""Tests for automatic white-FM window and scan-scaling analysis."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.artifacts import ArtifactStore
from qphase.core.result_loader import load_result
from qphase_sde.analyser.allan_scaling import (
    AllanScalingAnalyzer,
    AllanScalingConfig,
    NormalFormExpectation,
)
from qphase_sde.analyser.allan_variance import _build_allan_products
from qphase_sde.contracts.bundle import SDEProvenance
from qphase_sde.products import add_scan_parameter_coordinates
from qphase_sde.result import SDEDataBundle


def _result(
    epsilon: float,
    *,
    allan_tau_slope: float = -1.0,
    trajectories: int = 24,
) -> dict:
    tau = np.geomspace(10.0, 10000.0, 13)
    expected_intensity = epsilon ** (-4.0 / 3.0)
    multipliers = np.linspace(0.9, 1.1, trajectories)
    per_trajectory = (
        expected_intensity * multipliers[:, None] * tau[None, :] ** allan_tau_slope
    )
    variance = np.mean(per_trajectory, axis=0)
    sem = np.std(per_trajectory, axis=0, ddof=1) / np.sqrt(trajectories)
    frequency = 0.2 + 0.4 * epsilon ** (1.0 / 3.0)
    phase_frequency = frequency + np.linspace(-1e-5, 1e-5, trajectories)
    payload = {
        "t0": 0.0,
        "dt": 1.0,
        "orientation": "phase_increasing",
        "measurement_model": "phase_frequency",
        "n_traj": trajectories,
        "n_samples": 100000,
        "mode_results": {
            0: {
                "allan": {
                    "tau": tau,
                    "angular_frequency_variance": variance,
                    "angular_frequency_variance_sem": sem,
                    "per_trajectory": per_trajectory,
                    "nonoverlap_per_trajectory": per_trajectory,
                    "total_independent_window_count": np.full(tau.shape, 500),
                },
                "phase_increment": {
                    "mean_angular_frequency_per_trajectory": phase_frequency
                },
            }
        },
    }
    return payload


def _product_payload(product):
    payload = {}
    values = list(product.attributes.get("payload_meta", {}).items()) + [
        (variable.name, product.handle(variable.name).materialize())
        for variable in product.variables
    ]
    for name, value in values:
        if np.asarray(value).ndim == 0:
            value = np.asarray(value).item()
        target = payload
        parts = name.split(".")
        for part in parts[:-1]:
            key = int(part) if part.isdigit() else part
            target = target.setdefault(key, {})
        target[parts[-1]] = value
    return payload


def _execute(analyzer, inputs):
    product = analyzer.execute(
        inputs,
        backend=NumpyBackend(),
        label="allan_scaling",
    )["allan_scaling"]
    return _product_payload(product)


def _config() -> AllanScalingConfig:
    return AllanScalingConfig(
        scan_param="omega_c",
        critical_value=1.0,
        mode=0,
        min_local_r2=0.999,
        min_tau_decades=0.5,
        min_scaling_points=5,
        target_scaling_decades=1.0,
        bootstrap_samples=200,
        bootstrap_seed=7,
        normal_form=NormalFormExpectation(n=3, k=1, m=0),
    )


def test_allan_scaling_recovers_white_fm_normal_form():
    epsilon = np.geomspace(1e-3, 1e-1, 9)
    result = _execute(
        AllanScalingAnalyzer(_config()),
        {"allan_variance": _typed_allan_input(epsilon)},
    )
    summary = result["summary"]

    assert summary["status"] == "ok"
    assert summary["epsilon_window_decades"] == pytest.approx(2.0)
    assert summary["noise_fit"]["exponent"] == pytest.approx(4.0 / 3.0)
    assert summary["frequency_fit"]["exponent"] == pytest.approx(1.0 / 3.0)
    assert summary["frequency_fit"]["model"] == ("omega0 + A * abs(epsilon) ** p")
    assert "linear" not in summary["frequency_fit"]
    assert summary["normal_form"]["frequency_match"] is True
    assert summary["normal_form"]["allan_match"] is True
    assert summary["orientation"] == "phase_increasing"
    assert all(
        row["orientation"] == "phase_increasing" for row in result["rows"]
    )
    assert all(row["accepted"] for row in result["rows"])
    assert len(result["rows"]) == epsilon.size
    assert all(row["tau_decades"] > 0.0 for row in result["rows"])
    assert all(
        np.isfinite(row["allan_slope"]) for row in result["rows"] if row["accepted"]
    )
    assert summary["selected_points"] == epsilon.size


def test_allan_scaling_rejects_colored_fm_even_with_scan_power_law():
    epsilon = np.geomspace(1e-3, 1e-1, 9)
    result = _execute(
        AllanScalingAnalyzer(_config()),
        {
            "allan_variance": _typed_allan_input(
                epsilon, allan_tau_slope=-0.4
            )
        },
    )
    summary = result["summary"]

    assert summary["status"] == "target_not_met"
    assert "epsilon_window_too_narrow" in summary["gate_failures"]
    assert summary["accepted_points"] == 0
    assert summary["noise_fit"] is None
    assert all(
        row["reason"] == "no local white-FM tau window"
        for row in result["rows"]
    )


def test_normal_form_expectation_includes_observable_projection_order():
    expectation = NormalFormExpectation(n=3, k=1, m=0, observable_order=2)

    assert expectation.frequency_exponent == pytest.approx(2.0 / 3.0)
    assert expectation.allan_exponent == pytest.approx(2.0 / 3.0)


def _scan_bundle(epsilon: np.ndarray, trajectories: int = 24):
    """In-memory typed scan bundle holding synthetic per-point Allan payloads."""
    product = _typed_allan_input(epsilon, trajectories=trajectories)
    return SDEDataBundle(
        {"allan_variance": product},
        SDEProvenance(dt=1.0),
        scan_axes={"omega_c": [1.0 + float(value) for value in epsilon]},
        scan_shape=(epsilon.size,),
        meta={"params": {"omega_c": 1.0}},
        n_traj_per_point=trajectories,
    )


def test_allan_scaling_consumes_v4_scan_bundle():
    """Regression: an in-memory upstream SDEDataBundle must yield its points."""
    epsilon = np.geomspace(1e-3, 1e-1, 9)
    bundle = _scan_bundle(epsilon)

    result = _execute(AllanScalingAnalyzer(_config()), bundle.products)
    summary = result["summary"]

    assert summary["status"] == "ok"
    assert summary["accepted_points"] == epsilon.size
    assert summary["noise_fit"]["exponent"] == pytest.approx(4.0 / 3.0)
    assert summary["normal_form"]["allan_match"] is True
    assert len(result["rows"]) == epsilon.size


def test_allan_scaling_execute_consumes_typed_product_mapping():
    epsilon = np.geomspace(1e-3, 1e-1, 9)
    analyzer = AllanScalingAnalyzer(_config())
    products = _build_allan_products(
        [_result(float(value)) for value in epsilon],
        scan_size=epsilon.size,
        label="allan_variance",
    )
    assert products is not None
    typed = products["allan_variance"]
    typed.schema.attributes["orientation"] = "phase_increasing"
    typed = add_scan_parameter_coordinates(typed, {"omega_c": 1.0 + epsilon})

    products = analyzer.execute(
        {"allan_variance": typed},
        backend=NumpyBackend(),
        label="allan_scaling",
    )

    assert "allan_scaling" in products
    assert products["allan_scaling"].attributes["graph_ready"] is True


def test_allan_scaling_consumes_restored_v4_artifact(tmp_path):
    """Regression: a persisted and restored v4 artifact must yield its points."""
    epsilon = np.geomspace(1e-3, 1e-1, 9)
    job_dir = tmp_path / "upstream_sim"
    job_dir.mkdir()
    store = ArtifactStore(job_dir, SimpleNamespace(shard_target_mib=1))
    store.save_result(_scan_bundle(epsilon), "sim")
    restored = load_result("sim", job_dir)

    result = _execute(AllanScalingAnalyzer(_config()), restored.products)
    summary = result["summary"]

    assert summary["status"] == "ok"
    assert summary["accepted_points"] == epsilon.size
    assert summary["noise_fit"]["exponent"] == pytest.approx(4.0 / 3.0)
    assert len(result["rows"]) == epsilon.size
    assert all(row["accepted"] for row in result["rows"])


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


def _typed_allan_input(
    epsilon: np.ndarray,
    *,
    trajectories: int = 24,
    allan_tau_slope: float = -1.0,
):
    """Typed Allan scan product with parameter coordinates for execute()."""
    products = _build_allan_products(
        [
            _result(
                float(value),
                trajectories=trajectories,
                allan_tau_slope=allan_tau_slope,
            )
            for value in epsilon
        ],
        scan_size=epsilon.size,
        label="allan_variance",
    )
    assert products is not None
    typed = products["allan_variance"]
    typed.schema.attributes["orientation"] = "phase_increasing"
    return add_scan_parameter_coordinates(typed, {"omega_c": 1.0 + epsilon})


def test_allan_scaling_execute_aborts_mid_loop_on_cancellation():
    epsilon = np.geomspace(1e-3, 1e-1, 9)
    analyzer = AllanScalingAnalyzer(_config())
    context = SimpleNamespace(
        cancellation=_FakeCancellation(abort_after=1), progress=None
    )

    with pytest.raises(_LoopAbortError):
        analyzer.execute(
            {"allan_variance": _typed_allan_input(epsilon)},
            backend=NumpyBackend(),
            label="allan_scaling",
            context=context,
        )
    assert context.cancellation.calls < epsilon.size


def test_allan_scaling_execute_aborts_inside_bootstrap_loop():
    # Pre-bootstrap polling costs 9 (collect) + 9 (windows) + 1 (candidate run)
    # + 15 (trial fits) = 34 polls; bootstrap_samples=200 then dominates, so an
    # abort after 100 polls lands well inside the bootstrap resampling loop.
    epsilon = np.geomspace(1e-3, 1e-1, 9)
    analyzer = AllanScalingAnalyzer(_config())
    context = SimpleNamespace(
        cancellation=_FakeCancellation(abort_after=100), progress=None
    )

    with pytest.raises(_LoopAbortError):
        analyzer.execute(
            {"allan_variance": _typed_allan_input(epsilon)},
            backend=NumpyBackend(),
            label="allan_scaling",
            context=context,
        )


def test_allan_scaling_execute_reports_tau_progress():
    epsilon = np.geomspace(1e-3, 1e-1, 9)
    analyzer = AllanScalingAnalyzer(_config())
    progress = _FakeProgress()
    context = SimpleNamespace(cancellation=None, progress=progress)

    products = analyzer.execute(
        {"allan_variance": _typed_allan_input(epsilon)},
        backend=NumpyBackend(),
        label="allan_scaling",
        context=context,
    )

    assert products
    assert progress.updates
    completed = [update["completed"] for update in progress.updates]
    assert completed == sorted(completed)
    assert all(update["unit"] == "tau" for update in progress.updates)
    assert all(
        str(update["stage"]).startswith("analyze") for update in progress.updates
    )
    assert all(update["total"] is None for update in progress.updates)
