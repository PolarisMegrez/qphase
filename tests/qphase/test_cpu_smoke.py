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


@pytest.fixture(scope="module")
def cpu_job_path():
    """Return the path to the CPU smoke job configuration."""
    path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "jobs"
        / "cpu_smoke_kerr_3pa.yaml"
    )
    if not path.exists():
        pytest.skip("CPU smoke job config not found")
    return path


def _discover_plugins():
    """Discover package and local plugins (safe on CPU-only machines)."""
    discovery.discover_plugins()
    discovery.discover_local_plugins()


def test_cpu_smoke_run_and_postprocess(cpu_job_path, tmp_path):
    """Run the CPU smoke job and postprocess its PSD output."""
    _discover_plugins()

    job_list = load_jobs_from_files([cpu_job_path])
    assert len(job_list.jobs) == 1

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

    assert len(results) == 2, "Expected two expanded jobs for epsilon scan"
    assert all(r.success for r in results), f"Job failed: {results}"

    # Locate the session directory that contains the per-job subdirectories.
    session_dir = results[0].run_dir.parent
    npz_files = list(session_dir.glob("*/*.npz"))
    assert len(npz_files) == 2, f"Expected 2 result files, found {npz_files}"

    output_dir = tmp_path / "postprocess"
    output_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "postprocess",
            str(session_dir),
            "--scan-param",
            "epsilon",
            "--mode",
            "0",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output_dir / "fit_results.csv").exists()
    assert (output_dir / "psd_merged.csv").exists()


def test_cpu_smoke_cli_list(cpu_job_path):
    """CLI can list the CPU smoke job."""
    _discover_plugins()

    runner = CliRunner()
    result = runner.invoke(app, ["run", "--list"])
    assert result.exit_code == 0, result.output
    assert "cpu_smoke_kerr_3pa" in result.output
