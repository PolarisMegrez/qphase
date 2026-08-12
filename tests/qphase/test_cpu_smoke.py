"""CPU-only end-to-end smoke tests using real models and numpy backend."""

from __future__ import annotations

from pathlib import Path

import pytest
from qphase.core.registry import discovery
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig
from qphase.core.workflow import load_workflow
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
    if not path.exists():
        pytest.skip("CPU smoke workflow config not found")
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
    if not path.exists():
        pytest.skip("CPU smoke job config not found")
    return path


def _discover_plugins():
    """Discover package and local plugins (safe on CPU-only machines)."""
    discovery.discover_plugins()
    discovery.discover_local_plugins()


def test_cpu_smoke_run_and_postprocess(cpu_workflow_path, temp_project):
    """Run the CPU smoke workflow and postprocess its PSD output via analyse mode."""
    _discover_plugins()

    job_list = load_workflow(cpu_workflow_path)
    assert len(job_list.jobs) == 2

    scheduler = Scheduler(system_config=SystemConfig(), project=temp_project)
    results = scheduler.run(job_list)

    assert len(results) == 2, "Expected one logical scan job + one fit job"
    assert all(r.success for r in results), f"Job failed: {results}"

    # Locate the fit job run directory.
    fit_result = next(r for r in results if r.job_name == "cpu_smoke_kerr_2mode_fit")
    sim_result = next(r for r in results if r.job_name == "cpu_smoke_kerr_2mode")
    assert (sim_result.job_dir / "artifact_manifest.json").exists()
    assert (fit_result.job_dir / "fit_results.csv").exists()
    assert (fit_result.job_dir / "psd_merged.csv").exists()


def test_cpu_smoke_cli_list(cpu_job_path):
    """CLI can list the CPU smoke workflow."""
    _discover_plugins()

    runner = CliRunner()
    result = runner.invoke(app, ["workflow", "list"])
    assert result.exit_code == 0, result.output
    assert "cpu_smoke_kerr_2mode" in result.output
