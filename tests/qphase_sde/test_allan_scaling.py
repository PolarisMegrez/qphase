"""Tests for automatic white-FM window and scan-scaling analysis."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.artifacts import ArtifactStore
from qphase.core.result_loader import load_result
from qphase.core.scan import ScanSpec
from qphase.data.resolver import default_artifact_resolver
from qphase_sde.analyser.allan_scaling import (
    AllanScalingAnalyzer,
    AllanScalingConfig,
    NormalFormExpectation,
)
from qphase_sde.contracts.bundle import SDEProvenance
from qphase_sde.result import SDEResult, bundle_from_result
from qphase_sde.runtime.scan import SDEScanResult


def _result(
    epsilon: float,
    *,
    allan_tau_slope: float = -1.0,
    trajectories: int = 24,
) -> SDEResult:
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
        }
    }
    return SDEResult(
        analysis={"allan_variance": payload},
        meta={"params": {"omega_c": 1.0 + epsilon}},
    )


def _config(tmp_path) -> AllanScalingConfig:
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
        output_dir=str(tmp_path),
    )


def test_allan_scaling_recovers_white_fm_normal_form(tmp_path):
    epsilon = np.geomspace(1e-3, 1e-1, 9)
    data = {f"point-{index}": _result(value) for index, value in enumerate(epsilon)}

    result = AllanScalingAnalyzer(_config(tmp_path)).analyze(data, NumpyBackend())
    summary = result.data_dict["summary"]

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
        row["orientation"] == "phase_increasing" for row in result.data_dict["rows"]
    )
    assert all(row["accepted"] for row in result.data_dict["rows"])
    assert (tmp_path / "allan_points.csv").exists()
    with (tmp_path / "allan_scaling.json").open(encoding="utf-8") as handle:
        assert json.load(handle)["status"] == "ok"


def test_allan_scaling_rejects_colored_fm_even_with_scan_power_law(tmp_path):
    epsilon = np.geomspace(1e-3, 1e-1, 9)
    data = {
        f"point-{index}": _result(value, allan_tau_slope=-0.4)
        for index, value in enumerate(epsilon)
    }

    result = AllanScalingAnalyzer(_config(tmp_path)).analyze(data, NumpyBackend())
    summary = result.data_dict["summary"]

    assert summary["status"] == "target_not_met"
    assert "epsilon_window_too_narrow" in summary["gate_failures"]
    assert summary["accepted_points"] == 0
    assert summary["noise_fit"] is None
    assert all(
        row["reason"] == "no local white-FM tau window"
        for row in result.data_dict["rows"]
    )


def test_normal_form_expectation_includes_observable_projection_order():
    expectation = NormalFormExpectation(n=3, k=1, m=0, observable_order=2)

    assert expectation.frequency_exponent == pytest.approx(2.0 / 3.0)
    assert expectation.allan_exponent == pytest.approx(2.0 / 3.0)


def _scan_bundle(epsilon: np.ndarray, trajectories: int = 24):
    """In-memory v4 scan bundle holding synthetic per-point Allan payloads."""
    results = [_result(float(value), trajectories=trajectories) for value in epsilon]
    combined = SDEResult(
        analysis={
            "allan_variance": [result.analysis["allan_variance"] for result in results]
        },
        meta={"params": {"omega_c": 1.0}},
    )
    grid = ScanSpec.model_validate(
        {
            "axes": {
                "omega_c": {
                    "target": "model.scan.omega_c",
                    "values": [1.0 + float(value) for value in epsilon],
                }
            }
        }
    ).compile()
    dataset = SDEScanResult(combined, grid, {"omega_c": 1.0}, trajectories)
    return bundle_from_result(dataset, provenance=SDEProvenance(dt=1.0))


def test_allan_scaling_consumes_v4_scan_bundle(tmp_path):
    """Regression: an in-memory upstream SDEDataBundle must yield its points."""
    epsilon = np.geomspace(1e-3, 1e-1, 9)
    bundle = _scan_bundle(epsilon)

    result = AllanScalingAnalyzer(_config(tmp_path)).analyze(bundle, NumpyBackend())
    summary = result.data_dict["summary"]

    assert summary["status"] == "ok"
    assert summary["accepted_points"] == epsilon.size
    assert summary["noise_fit"]["exponent"] == pytest.approx(4.0 / 3.0)
    assert summary["normal_form"]["allan_match"] is True
    assert len(result.data_dict["rows"]) == epsilon.size


def test_allan_scaling_consumes_restored_v4_artifact(tmp_path):
    """Regression: a clean-process restored v4 artifact must yield its points."""
    epsilon = np.geomspace(1e-3, 1e-1, 9)
    job_dir = tmp_path / "upstream_sim"
    job_dir.mkdir()
    store = ArtifactStore(job_dir, SimpleNamespace(shard_target_mib=1))
    store.save_result(_scan_bundle(epsilon), "sim")
    # Simulate a clean process: drop every in-process artifact binding.
    default_artifact_resolver().clear()
    restored = load_result("sim", job_dir)

    out_dir = tmp_path / "analysis"
    result = AllanScalingAnalyzer(_config(out_dir)).analyze(restored, NumpyBackend())
    summary = result.data_dict["summary"]

    assert summary["status"] == "ok"
    assert summary["accepted_points"] == epsilon.size
    assert summary["noise_fit"]["exponent"] == pytest.approx(4.0 / 3.0)
    assert (out_dir / "allan_points.csv").exists()
