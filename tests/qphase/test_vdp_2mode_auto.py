"""Automated VDP 2-mode peak-fitting regression test.

This test runs the ``configs/jobs/vdp_2mode_auto.yaml`` workflow, which keeps the
physical parameters of ``vdp_2mode.yaml`` but fixes:

* ``t1 = 1000.0``
* ``omega_a = 0.00251189``

The simulation is analyzed by ``analyser.psd`` and then fed into
``analyser.lorentz_fitter`` in analyze mode. The fit results CSV must contain
the peak position, peak height (``amplitude`` / ``peak_intensity``), linewidth
(FWHM), and ``R2``.
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
    """Return the path to the automated VDP workflow configuration."""
    path = (
        Path(__file__).parent.parent.parent / "configs" / "jobs" / "vdp_2mode_auto.yaml"
    )
    if not path.exists():
        pytest.skip("Automated VDP workflow config not found")
    return path


def _discover_plugins():
    """Discover package and local plugins (safe on CPU-only machines)."""
    discovery.discover_plugins()
    discovery.discover_local_plugins()


def test_vdp_2mode_auto_peak_fit(vdp_auto_workflow_path, tmp_path):
    """Run the automated VDP workflow and verify the fitted peak parameters."""
    _discover_plugins()

    job_list = load_jobs_from_files([vdp_auto_workflow_path])
    assert len(job_list.jobs) == 2, "Expected sim + fit jobs"

    repo = Path(__file__).parent.parent.parent
    system_config = SystemConfig(
        paths={
            "output_dir": str(tmp_path / "runs"),
            "global_file": str(repo / "configs" / "global.yaml"),
            "config_dirs": [str(repo / "configs")],
            "plugin_dirs": [str(repo / "models")],
        }
    )

    scheduler = Scheduler(system_config=system_config)
    results = scheduler.run(job_list)

    assert len(results) == 2, results
    assert all(r.success for r in results), f"Job failed: {results}"

    fit_result = next(r for r in results if r.job_name == "vdp_2mode_auto_fit")
    fit_csv = fit_result.run_dir / "fit_results.csv"
    assert fit_csv.exists(), "fit_results.csv was not produced"

    with fit_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1, "Expected a single fit row for the fixed omega_a"

    row = rows[0]
    assert row["status"] == "ok", row
    center = float(row["center"])
    linewidth = float(row["linewidth"])
    r2 = float(row["R2"])

    # Reference values from previous long-run characterizations.
    # With t1=1000 the frequency grid is coarser, so the linewidth is allowed
    # a somewhat larger absolute tolerance than the peak position.
    expected_center = -0.08574307
    expected_linewidth = 3.54044345e-03

    assert np.isclose(
        center, expected_center, atol=1e-3
    ), f"Fitted center {center} deviates too far from {expected_center}"
    assert np.isclose(
        linewidth, expected_linewidth, atol=3e-3
    ), f"Fitted linewidth {linewidth} deviates too far from {expected_linewidth}"
    assert r2 > 0.5, f"R2 {r2} is unexpectedly low"

    # The CSV should expose both the total peak height and the height above baseline.
    assert float(row["amplitude"]) > 0.0
    assert float(row["peak_intensity"]) > 0.0
