"""End-to-end pipeline test for the VDP two-mode model.

This test exercises the full scheduler workflow introduced by the refactor:

* CPU-only ``backend.numpy`` + ``integrator.srk`` (Heun).
* ``model.vdp_2mode`` with a parameter scan on ``omega_a``.
* Multiple analyzers on the same simulation job: ``psd``, ``dist``, ``pdist``.
* Two downstream analyze-mode jobs using ``analyser.lorentz_fitter`` for modes 0
  and 1, including ``export_dist`` outputs.
* CLI invocation via ``qphase run <job>`` with a temporary system config.

The sequence length is intentionally capped at 1000 steps
(``t0=0.0, t1=100.0, dt=0.1``) and ``n_traj=20`` to keep the test fast on CPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from qphase.core import system_config as system_config_module
from qphase.core.config_loader import load_jobs_from_files
from qphase.core.registry import discovery
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig
from qphase.core.utils import save_yaml
from qphase.main import app
from typer.testing import CliRunner


@pytest.fixture(scope="module")
def vdp_workflow_path():
    """Return the path to the VDP smoke workflow configuration."""
    path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "jobs"
        / "vdp_2mode_smoke.yaml"
    )
    if not path.exists():
        pytest.skip("VDP smoke workflow config not found")
    return path


def _repo_root() -> Path:
    return Path(__file__).parent.parent.parent


def _discover_plugins():
    """Discover package and local plugins (safe on CPU-only machines)."""
    discovery.discover_plugins()
    discovery.discover_local_plugins()


def _make_system_config(tmp_path: Path) -> SystemConfig:
    """Build a temporary system config that keeps test outputs in tmp_path."""
    repo = _repo_root()
    return SystemConfig(
        paths={
            "output_dir": str(tmp_path / "runs"),
            "global_file": str(repo / "configs" / "global.yaml"),
            "config_dirs": [str(repo / "configs")],
            "plugin_dirs": [str(repo / "models")],
        }
    )


def _write_system_config(tmp_path: Path) -> Path:
    """Write a temporary system.yaml and return its path."""
    repo = _repo_root()
    cfg = {
        "paths": {
            "output_dir": str(tmp_path / "runs"),
            "global_file": str(repo / "configs" / "global.yaml"),
            "config_dirs": [str(repo / "configs")],
            "plugin_dirs": [str(repo / "models")],
        }
    }
    path = tmp_path / "system.yaml"
    save_yaml(cfg, path)
    return path


def _clear_system_config_cache():
    """Force ``load_system_config`` to reload on the next CLI invocation."""
    system_config_module._SYSTEM_CONFIG_CACHE = None


def _extract_json(output: str) -> dict:
    """Extract the last JSON object from mixed log + JSON CLI output."""
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in CLI output")
    return json.loads(output[start : end + 1])


def test_vdp_2mode_smoke_workflow_scheduler(vdp_workflow_path, tmp_path):
    """Run the VDP workflow through the scheduler and verify all artifacts."""
    _discover_plugins()

    job_list = load_jobs_from_files([vdp_workflow_path])
    assert len(job_list.jobs) == 3, "Expected sim + 2 fit jobs"

    scheduler = Scheduler(system_config=_make_system_config(tmp_path))
    results = scheduler.run(job_list)

    assert len(results) == 5, "Expected 3 expanded sim jobs + 2 fit jobs"
    assert all(r.success for r in results), f"Job failed: {results}"

    sim_results = [r for r in results if r.job_name.startswith("vdp_2mode_sim_")]
    assert len(sim_results) == 3
    for sim in sim_results:
        npz_files = list(sim.run_dir.glob("*.npz"))
        assert len(npz_files) == 1, "Expected exactly one saved result per sim job"
        assert (sim.run_dir / "config_snapshot.json").exists()

    for mode in (0, 1):
        fit = next(r for r in results if r.job_name == f"vdp_2mode_fit_mode{mode}")
        assert (fit.run_dir / "fit_results.csv").exists()
        assert (fit.run_dir / "psd_merged.csv").exists()
        assert (fit.run_dir / "dist_merged.npz").exists()
        assert (fit.run_dir / "pdist_merged.pkl").exists()

        with open(fit.run_dir / "fit_results.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3, "Expected one fit row per scan value"
        for row in rows:
            assert float(row["R2"]) >= 0.0
            assert row["status"] in {"ok", "low_quality", "failed"}


def test_vdp_2mode_smoke_cli_plan(vdp_workflow_path, tmp_path):
    """CLI --plan shows the expanded workflow including analyzer dependencies."""
    _discover_plugins()
    config_path = _write_system_config(tmp_path)
    _clear_system_config_cache()

    runner = CliRunner(env={"QPHASE_SYSTEM_CONFIG": str(config_path)})
    result = runner.invoke(app, ["run", "vdp_2mode_smoke", "--plan", "--json"])
    assert result.exit_code == 0, result.output

    plan = _extract_json(result.output)
    assert len(plan["original_jobs"]) == 3
    assert len(plan["expanded_jobs"]) == 5

    edges = plan["edges"]
    targets = {e["target"] for e in edges}
    assert "vdp_2mode_fit_mode0" in targets
    assert "vdp_2mode_fit_mode1" in targets


def test_vdp_2mode_smoke_cli_run(vdp_workflow_path, tmp_path):
    """CLI run executes the full pipeline and reports all jobs successful."""
    _discover_plugins()
    config_path = _write_system_config(tmp_path)
    _clear_system_config_cache()

    runner = CliRunner(env={"QPHASE_SYSTEM_CONFIG": str(config_path)})
    result = runner.invoke(app, ["run", "vdp_2mode_smoke", "--json"])
    assert result.exit_code == 0, result.output

    report = _extract_json(result.output)
    assert report["success_count"] == 5
    assert report["total_count"] == 5

    for entry in report["results"]:
        assert entry["success"], entry

    fit_dirs = [
        Path(entry["run_dir"])
        for entry in report["results"]
        if entry["job_name"].startswith("vdp_2mode_fit_mode")
    ]
    assert len(fit_dirs) == 2
    for fit_dir in fit_dirs:
        assert (fit_dir / "fit_results.csv").exists()
        assert (fit_dir / "psd_merged.csv").exists()
        assert (fit_dir / "dist_merged.npz").exists()
        assert (fit_dir / "pdist_merged.pkl").exists()
