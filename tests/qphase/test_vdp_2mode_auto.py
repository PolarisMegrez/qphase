"""Automated VDP 2-mode peak-fitting regression test.

This test runs the ``vdp_2mode_auto`` workflow, which keeps the
physical parameters of ``vdp_2mode.yaml`` but fixes:

* ``t1 = 1000.0``
* ``omega_a = 0.00251189``

The simulation is analyzed by ``analyser.psd`` and then fed into
``analyser.lorentz_fitter`` in analyze mode. The fit results CSV must contain
the peak position, peak height (``amplitude`` / ``peak_intensity``), linewidth
(FWHM), and ``R2``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from qphase.core.registry import discovery
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig
from qphase.core.workflow import load_workflow
from qphase.data import load_products

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


@pytest.fixture(scope="module")
def vdp_auto_workflow_path():
    """Return the path to the automated VDP workflow configuration."""
    path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "workflows"
        / "vdp_2mode"
        / "vdp_2mode_auto.yaml"
    )
    assert path.exists(), f"VDP auto workflow config is missing: {path}"
    return path


def _discover_plugins():
    """Discover package and local plugins (safe on CPU-only machines)."""
    discovery.discover_plugins()
    discovery.discover_local_plugins()


def test_vdp_2mode_auto_peak_fit(vdp_auto_workflow_path, temp_project):
    """Run the automated VDP workflow and verify the fitted peak parameters."""
    _discover_plugins()

    job_list = load_workflow(vdp_auto_workflow_path)
    assert len(job_list.jobs) == 3, "Expected sim + fit + peaks jobs"

    scheduler = Scheduler(system_config=SystemConfig(), project=temp_project)
    results = scheduler.run(job_list)

    assert len(results) == 3, results
    assert all(r.success for r in results), f"Job failed: {results}"

    fit_result = next(r for r in results if r.job_name == "vdp_2mode_auto_fit")
    fit_products = load_products(fit_result.job_dir)
    assert "lorentz_fitter" in fit_products, sorted(fit_products)
    fit = fit_products["lorentz_fitter"]

    rows = fit.attributes["payload_meta"]["fit_rows"]
    assert len(rows) == 1, "Expected a single fit row for the fixed omega_a"

    row = rows[0]
    assert row["status"] == "ok", row
    center = float(np.asarray(fit.handle("center").materialize())[0])
    linewidth = float(np.asarray(fit.handle("linewidth").materialize())[0])
    r2 = float(np.asarray(fit.handle("R2").materialize())[0])
    amplitude = float(np.asarray(fit.handle("amplitude").materialize())[0])
    peak_intensity = float(np.asarray(fit.handle("peak_intensity").materialize())[0])

    # Reference values from previous long-run characterizations.
    # With t1=1000 the frequency grid is coarser, so the linewidth is allowed
    # a somewhat larger absolute tolerance than the peak position.
    # The default phase_decreasing convention maps exp(-i*omega*t) to +omega.
    expected_center = 0.08574307
    expected_linewidth = 3.54044345e-03

    assert np.isclose(center, expected_center, atol=1e-3), (
        f"Fitted center {center} deviates too far from {expected_center}"
    )
    assert np.isclose(linewidth, expected_linewidth, atol=3e-3), (
        f"Fitted linewidth {linewidth} deviates too far from {expected_linewidth}"
    )
    assert r2 > 0.5, f"R2 {r2} is unexpectedly low"

    # The fit row should expose both the total peak height and the height above
    # baseline.
    assert amplitude > 0.0
    assert peak_intensity > 0.0
