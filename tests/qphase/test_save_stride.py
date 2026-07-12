"""Regression tests for ``engine.sde.save_stride``.

``save_stride`` lets the integrator use a small ``dt`` for stability while
only storing (and FFT-ing) every ``N``-th sample.  These tests verify that
increasing ``save_stride`` narrows the saved ``dt`` and leaves the fitted
Lorentzian peak parameters unchanged for a narrow low-frequency line.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from qphase.core.config_loader import load_jobs_from_files
from qphase.core.registry import discovery
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig


@pytest.fixture(scope="module")
def vdp_auto_workflow_path():
    """Path to the lightweight single-point VDP workflow."""
    path = (
        Path(__file__).parent.parent.parent / "configs" / "jobs" / "vdp_2mode_auto.yaml"
    )
    if not path.exists():
        pytest.skip("vdp_2mode_auto.yaml config not found")
    return path


def _make_system_config(tmp_path: Path) -> SystemConfig:
    """Build a SystemConfig that writes runs under ``tmp_path``."""
    repo = Path(__file__).parent.parent.parent
    return SystemConfig(
        paths={
            "output_dir": str(tmp_path / "runs"),
            "global_file": str(repo / "configs" / "global.yaml"),
            "config_dirs": [str(repo / "configs")],
            "plugin_dirs": [str(repo / "models")],
        }
    )


@pytest.mark.parametrize("save_stride", [1, 10, 50])
def test_save_stride_preserves_peak(vdp_auto_workflow_path, tmp_path, save_stride):
    """Vary ``save_stride`` and check that the fitted peak stays stable."""
    discovery.discover_plugins()
    discovery.discover_local_plugins()

    job_list = load_jobs_from_files([vdp_auto_workflow_path])
    assert len(job_list.jobs) == 2

    # Override the stride for this test.  Use a smaller ensemble so the
    # parametrized sweep stays fast while still averaging enough trajectories
    # to resolve the narrow Lorentzian line.
    sim_job = job_list.jobs[0]
    sim_job.engine["sde"]["save_stride"] = save_stride
    sim_job.engine["sde"]["n_traj"] = 10

    scheduler = Scheduler(system_config=_make_system_config(tmp_path))
    results = scheduler.run(job_list)

    assert len(results) == 2, results
    assert all(r.success for r in results), f"Job failed: {results}"

    sim_result = next(r for r in results if r.job_name == "vdp_2mode_auto_sim")
    fit_result = next(r for r in results if r.job_name == "vdp_2mode_auto_fit")

    # Saved dt must reflect the stride.
    npz_path = sim_result.run_dir / "vdp_2mode_auto_sim.npz"
    assert npz_path.exists()
    with np.load(npz_path, allow_pickle=True) as npz:
        saved_dt = float(npz["dt"])
    assert np.isclose(saved_dt, 0.1 * save_stride, rtol=1e-12)

    # The fitted peak should be stable across strides.
    fit_csv = fit_result.run_dir / "fit_results.csv"
    assert fit_csv.exists()
    with fit_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok", row

    center = float(row["center"])
    linewidth = float(row["linewidth"])
    r2 = float(row["R2"])

    expected_center = -0.08574307
    expected_linewidth = 3.54044345e-03

    assert np.isclose(center, expected_center, atol=1e-3), (
        f"save_stride={save_stride}: center {center} deviates from {expected_center}"
    )
    assert np.isclose(linewidth, expected_linewidth, atol=3e-3), (
        f"save_stride={save_stride}: linewidth {linewidth} deviates from "
        f"{expected_linewidth}"
    )
    assert r2 > 0.5, f"save_stride={save_stride}: R2 {r2} is unexpectedly low"
