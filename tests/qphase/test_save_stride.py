"""Regression tests for ``engine.sde.save_stride``.

``save_stride`` lets the integrator use a small ``dt`` for stability while
only storing (and FFT-ing) every ``N``-th sample.  These tests verify that
increasing ``save_stride`` narrows the saved ``dt`` and leaves the fitted
Lorentzian peak parameters unchanged for a narrow low-frequency line.

Absolute golden values of that peak live in ``test_vdp_2mode_auto.py``; here
the two strides are compared against each other, so the assertion is about
stride invariance rather than about the fit itself.
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

#: Strides compared against each other; 10 was dropped as a third full run of
#: the same pipeline without adding an independent physical fact.
STRIDES = (1, 50)

#: Cross-stride tolerances.  Both runs used to be bounded against the same
#: absolute golden value (center 1e-3, linewidth 3e-3), so their mutual
#: difference is bounded here by those same per-run tolerances.
CENTER_ATOL = 1e-3
LINEWIDTH_ATOL = 3e-3


@pytest.fixture(scope="module")
def vdp_auto_workflow_path():
    """Path to the lightweight single-point VDP workflow."""
    path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "workflows"
        / "vdp_2mode"
        / "vdp_2mode_auto.yaml"
    )
    assert path.exists(), f"vdp_2mode_auto.yaml fixture is missing: {path}"
    return path


def _make_system_config(tmp_path: Path) -> SystemConfig:
    """Build a SystemConfig that writes runs under ``tmp_path``."""
    return SystemConfig()


def test_save_stride_preserves_peak(vdp_auto_workflow_path, temp_project):
    """Run each stride once and compare the fitted peaks against each other."""
    discovery.discover_plugins()
    discovery.discover_local_plugins()

    fits: dict[int, tuple[float, float]] = {}
    for save_stride in STRIDES:
        job_list = load_workflow(vdp_auto_workflow_path)
        assert len(job_list.jobs) == 3

        # Override the stride for this run.  Use a smaller ensemble so the
        # sweep stays fast while still averaging enough trajectories to
        # resolve the narrow Lorentzian line.
        sim_job = job_list.jobs[0]
        sim_job.engine["sde"]["save_stride"] = save_stride
        sim_job.engine["sde"]["n_traj"] = 10

        scheduler = Scheduler(
            system_config=_make_system_config(temp_project.root), project=temp_project
        )
        results = scheduler.run(job_list)

        assert len(results) == 3, results
        assert all(r.success for r in results), f"Job failed: {results}"

        sim_result = next(r for r in results if r.job_name == "vdp_2mode_auto_sim")
        fit_result = next(r for r in results if r.job_name == "vdp_2mode_auto_fit")

        # Saved dt must reflect the stride.
        products = load_products(sim_result.job_dir)
        saved_dt = float(
            np.asarray(products["psd"].handle("sample_dt").materialize())
        )
        assert np.isclose(saved_dt, 0.1 * save_stride, rtol=1e-12), save_stride

        fit_products = load_products(fit_result.job_dir)
        assert "lorentz_fitter" in fit_products, sorted(fit_products)
        fit_product = fit_products["lorentz_fitter"]
        rows = fit_product.attributes["payload_meta"]["fit_rows"]
        assert len(rows) == 1
        assert rows[0]["status"] == "ok", rows[0]

        fits[save_stride] = (
            float(np.asarray(fit_product.handle("center").materialize())[0]),
            float(np.asarray(fit_product.handle("linewidth").materialize())[0]),
        )

    (center_fast, linewidth_fast), (center_slow, linewidth_slow) = (
        fits[stride] for stride in STRIDES
    )
    assert np.isclose(center_slow, center_fast, atol=CENTER_ATOL), (
        f"center moves with save_stride: {center_fast} (stride {STRIDES[0]}) vs "
        f"{center_slow} (stride {STRIDES[-1]})"
    )
    assert np.isclose(linewidth_slow, linewidth_fast, atol=LINEWIDTH_ATOL), (
        f"linewidth moves with save_stride: {linewidth_fast} (stride "
        f"{STRIDES[0]}) vs {linewidth_slow} (stride {STRIDES[-1]})"
    )
