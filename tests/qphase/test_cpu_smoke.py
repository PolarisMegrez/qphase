"""CPU-only end-to-end smoke tests using real models and numpy backend."""

from __future__ import annotations

from pathlib import Path

import pytest
from qphase.core.config_loader import load_jobs_from_files
from qphase.core.registry import discovery
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig
from qphase.main import app
from typer.testing import CliRunner

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def cpu_workflow_path():
    """Return the path to the CPU smoke workflow configuration."""
    path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "jobs"
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
        / "jobs"
        / "cpu_smoke_kerr_2mode.yaml"
    )
    if not path.exists():
        pytest.skip("CPU smoke job config not found")
    return path


def _discover_plugins():
    """Discover package and local plugins (safe on CPU-only machines)."""
    discovery.discover_plugins()
    discovery.discover_local_plugins()


def test_cpu_smoke_run_and_postprocess(cpu_workflow_path, tmp_path):
    """Run the CPU smoke workflow and postprocess its PSD output via analyse mode."""
    _discover_plugins()

    job_list = load_jobs_from_files([cpu_workflow_path])
    assert len(job_list.jobs) == 2

    # Redirect output to a temp directory so we do not pollute the repo.
    system_config = SystemConfig(
        paths={
            "output_dir": str(tmp_path / "runs"),
            "global_file": "./configs/global.yaml",
            "plugin_dirs": ["./models"],
            "config_dirs": ["./configs"],
        }
    )

    scheduler = Scheduler(system_config=system_config)
    results = scheduler.run(job_list)

    assert len(results) == 3, "Expected two expanded sim jobs + one fit job"
    assert all(r.success for r in results), f"Job failed: {results}"

    # Locate the fit job run directory.
    fit_result = next(r for r in results if r.job_name == "cpu_smoke_kerr_2mode_fit")
    assert (fit_result.run_dir / "fit_results.csv").exists()
    assert (fit_result.run_dir / "psd_merged.csv").exists()


def test_cpu_smoke_cli_list(cpu_job_path):
    """CLI can list the CPU smoke job."""
    _discover_plugins()

    runner = CliRunner()
    result = runner.invoke(app, ["run", "--list"])
    assert result.exit_code == 0, result.output
    assert "cpu_smoke_kerr_2mode" in result.output
