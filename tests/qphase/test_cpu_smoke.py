"""CPU-only end-to-end smoke tests using real models and numpy backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from qphase.core.registry import discovery
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig
from qphase.core.workflow import load_workflow
from qphase.data import load_products
from qphase.main import app
from typer.testing import CliRunner

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def cpu_workflow_path():
    """Return the path to the CPU smoke workflow configuration."""
    path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "workflows"
        / "misc"
        / "cpu_smoke_kerr_2mode_fit.yaml"
    )
    assert path.exists(), f"CPU smoke workflow config is missing: {path}"
    return path


@pytest.fixture(scope="module")
def cpu_job_path():
    """Return the path to the CPU smoke job configuration."""
    path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "workflows"
        / "misc"
        / "cpu_smoke_kerr_2mode.yaml"
    )
    assert path.exists(), f"CPU smoke job config is missing: {path}"
    return path


def _discover_plugins():
    """Discover package and local plugins (safe on CPU-only machines)."""
    discovery.discover_plugins()
    discovery.discover_local_plugins()


def test_cpu_smoke_run_and_postprocess(cpu_workflow_path, temp_project):
    """Run the CPU smoke workflow and postprocess its PSD output via analyse mode."""
    _discover_plugins()

    job_list = load_workflow(cpu_workflow_path)
    assert len(job_list.jobs) == 3

    scheduler = Scheduler(system_config=SystemConfig(), project=temp_project)
    results = scheduler.run(job_list)

    assert len(results) == 3, "Expected scan, fit, and standalone peaks jobs"
    assert all(r.success for r in results), f"Job failed: {results}"

    # Locate the fit job run directory.
    fit_result = next(r for r in results if r.job_name == "cpu_smoke_kerr_2mode_fit")
    sim_result = next(r for r in results if r.job_name == "cpu_smoke_kerr_2mode")
    assert (sim_result.job_dir / "artifact_manifest.json").exists()
    fit_products = load_products(fit_result.job_dir)
    assert "lorentz_fitter" in fit_products, sorted(fit_products)
    fitted_centers = np.asarray(
        fit_products["lorentz_fitter"].handle("center").materialize()
    )
    assert fitted_centers.size == 3, "expected one fit row per scan value"
    assert np.all(np.isfinite(fitted_centers))
    peaks_result = next(
        r for r in results if r.job_name == "cpu_smoke_kerr_2mode_peaks"
    )
    assert (peaks_result.job_dir / "artifact_manifest.json").exists()


def test_cpu_smoke_cli_list(cpu_job_path):
    """CLI can list the CPU smoke workflow."""
    _discover_plugins()

    runner = CliRunner()
    result = runner.invoke(app, ["workflow", "list"])
    assert result.exit_code == 0, result.output
    assert "cpu_smoke_kerr_2mode" in result.output
