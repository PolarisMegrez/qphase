"""End-to-end pipeline test for the VDP two-mode model.

This test exercises the full scheduler workflow introduced by the refactor:

* CPU-only ``backend.numpy`` + ``integrator.srk`` (Heun).
* ``model.vdp_2mode`` with a parameter scan on ``omega_a``.
* Multiple analyzers on the same simulation job: ``psd``, ``dist``, ``pdist``.
* Two downstream analyze-mode jobs using ``analyser.lorentz_fitter`` for modes 0
  and 1; their fit rows are persisted as typed products.
* CLI invocation via ``qphase run <job>`` with a temporary system config.

The sequence length is intentionally capped at 1000 steps
(``t0=0.0, t1=100.0, dt=0.1``) and ``n_traj=20`` to keep the test fast on CPU.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from qphase.core import system_config as system_config_module
from qphase.core.registry import discovery
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig
from qphase.core.utils import save_yaml
from qphase.core.workflow import load_workflow
from qphase.data import load_products
from qphase.main import app
from typer.testing import CliRunner

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def vdp_workflow_path():
    """Return the path to the VDP smoke workflow configuration."""
    path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "workflows"
        / "vdp_2mode"
        / "vdp_2mode_smoke.yaml"
    )
    assert path.exists(), f"VDP smoke workflow config is missing: {path}"
    return path


@pytest.fixture
def vdp_cli_project(vdp_workflow_path, temp_project, monkeypatch):
    """Install the local workflow and models in an isolated CLI project."""
    shutil.copy2(
        vdp_workflow_path,
        temp_project.workflow_root / vdp_workflow_path.name,
    )
    shutil.copytree(
        _repo_root() / "models",
        temp_project.root / "models",
        dirs_exist_ok=True,
    )
    monkeypatch.setenv("QPHASE_PROJECT", str(temp_project.root))
    return temp_project


def _repo_root() -> Path:
    return Path(__file__).parent.parent.parent


def _discover_plugins():
    """Discover package and local plugins (safe on CPU-only machines)."""
    discovery.discover_plugins()
    discovery.discover_local_plugins()


def _make_system_config(tmp_path: Path) -> SystemConfig:
    """Build a temporary system config that keeps test outputs in tmp_path."""
    return SystemConfig()


def _write_system_config(tmp_path: Path) -> Path:
    """Write a temporary system.yaml and return its path."""
    cfg = {}
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


def test_vdp_2mode_smoke_workflow_scheduler(vdp_workflow_path, temp_project):
    """Run the VDP workflow through the scheduler and verify all artifacts."""
    _discover_plugins()

    job_list = load_workflow(vdp_workflow_path)
    assert len(job_list.jobs) == 4, "Expected sim + 2 fit + peaks jobs"

    scheduler = Scheduler(
        system_config=_make_system_config(temp_project.root), project=temp_project
    )
    results = scheduler.run(job_list)

    assert len(results) == 4, "Expected one logical sim job + 2 fit + peaks jobs"
    assert all(r.success for r in results), f"Job failed: {results}"

    sim = next(r for r in results if r.job_name == "vdp_2mode_sim")
    assert (sim.job_dir / "config_snapshot.json").exists()
    assert (sim.job_dir / "artifact_manifest.json").exists()
    products = load_products(sim.job_dir)
    assert "psd" in products, "Expected the scan dataset to carry a psd product"
    # dist/pdist live where they are produced, as typed products of the sim job.
    assert "dist" in products, sorted(products)
    assert "pdist" in products, sorted(products)

    for mode in (0, 1):
        fit = next(r for r in results if r.job_name == f"vdp_2mode_fit_mode{mode}")
        fit_products = load_products(fit.job_dir)
        assert "lorentz_fitter" in fit_products, sorted(fit_products)
        fit_product = fit_products["lorentz_fitter"]
        rows = fit_product.attributes["payload_meta"]["fit_rows"]
        assert len(rows) == 3, "Expected one fit row per scan value"
        r2 = np.asarray(fit_product.handle("R2").materialize())
        assert r2.size == len(rows)
        assert np.all(r2 >= 0.0)
        for row in rows:
            assert row["status"] in {"ok", "low_quality", "failed"}


def test_vdp_2mode_smoke_cli_plan(vdp_workflow_path, vdp_cli_project, tmp_path):
    """CLI --plan shows the expanded workflow including analyzer dependencies."""
    _discover_plugins()
    config_path = _write_system_config(tmp_path)
    _clear_system_config_cache()

    runner = CliRunner(env={"QPHASE_SYSTEM_CONFIG": str(config_path)})
    result = runner.invoke(app, ["run", "vdp_2mode_smoke", "--plan", "--json"])
    assert result.exit_code == 0, result.output

    plan = _extract_json(result.output)
    assert len(plan["jobs"]) == 4
    sim_plan = next(job for job in plan["jobs"] if job["name"] == "vdp_2mode_sim")
    assert sim_plan["scan_summary"]["shape"] == [3]

    edges = plan["edges"]
    targets = {e["target"] for e in edges}
    assert "vdp_2mode_fit_mode0" in targets
    assert "vdp_2mode_fit_mode1" in targets
    assert "vdp_2mode_peaks" in targets


def test_vdp_2mode_smoke_cli_run(vdp_workflow_path, vdp_cli_project, tmp_path):
    """CLI run executes the full pipeline and reports all jobs successful."""
    _discover_plugins()
    config_path = _write_system_config(tmp_path)
    _clear_system_config_cache()

    runner = CliRunner(env={"QPHASE_SYSTEM_CONFIG": str(config_path)})
    result = runner.invoke(app, ["run", "vdp_2mode_smoke", "--json"])
    assert result.exit_code == 0, result.output

    report = _extract_json(result.output)
    assert report["completed"] == 4
    assert report["failed"] == 0
    session = Path(report["session_dir"])
    sim_products = load_products(session / "vdp_2mode_sim")
    assert {"psd", "dist", "pdist"} <= set(sim_products), sorted(sim_products)
    for mode in (0, 1):
        fit_dir = session / f"vdp_2mode_fit_mode{mode}"
        fit_products = load_products(fit_dir)
        assert "lorentz_fitter" in fit_products, sorted(fit_products)
        rows = fit_products["lorentz_fitter"].attributes["payload_meta"]["fit_rows"]
        assert len(rows) == 3, "Expected one fit row per scan value"
