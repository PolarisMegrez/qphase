"""End-to-end tests for batched SDE parameter scans."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from qphase.core.batch_negotiator import BatchNegotiator
from qphase.core.config import JobConfig, JobList
from qphase.core.registry import discovery, registry
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig

pytestmark = pytest.mark.e2e


REPO_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = REPO_ROOT / "models"


def _make_scan_job_list(
    param_values: list[float], n_traj: int = 10, n_steps: int = 50
) -> list[JobConfig]:
    """Build expanded SDE scan jobs that differ only in omega_a."""
    dt = 0.001
    jobs = []
    for idx, val in enumerate(param_values):
        jobs.append(
            JobConfig(
                name=f"vdp_scan_{idx:03d}",
                engine={
                    "sde": {
                        "t0": 0.0,
                        "t1": n_steps * dt,
                        "dt": dt,
                        "n_traj": n_traj,
                        "seed": 42,
                        "ic": [["7.0+0.0j", "0.0-7.0j"]],
                        "save_stride": n_steps,
                    }
                },
                plugins={},
                backend={"numpy": {}},
                integrator={"euler_maruyama": {}},
                model={
                    "vdp_2mode": {
                        "omega_a": val,
                        "omega_b": 0.0,
                        "gamma_a": 2.0,
                        "gamma_b": 1.0,
                        "Gamma": 0.01,
                        "g": 0.5,
                        "D": 1.0,
                    }
                },
                save=True,
            )
        )
    return jobs


@pytest.fixture(autouse=True)
def _discover_plugins():
    """Ensure qphase_sde plugins and local model plugins are discovered."""
    discovery.discover_plugins()
    discovery.discover_local_plugins()


def test_batch_plan_contains_vectorized_params():
    """The SDE batch planner produces a merged job with repeated scan params."""
    negotiator = BatchNegotiator(registry)
    jobs = _make_scan_job_list([0.001, 0.002, 0.003])
    groups = negotiator.group_jobs(jobs)

    assert len(groups) == 1
    batch = groups[0]
    batch_job = batch.plan.batch_job

    # n_traj multiplied by scan count.
    assert batch_job.engine["sde"]["n_traj"] == 30

    # The scanned parameter is repeated n_traj times per scan point.
    omega_a = batch_job.model_extra["model"]["vdp_2mode"]["omega_a"]
    expected = np.repeat([0.001, 0.002, 0.003], 10)
    np.testing.assert_array_almost_equal(np.asarray(omega_a), expected)
    # Keep config JSON-serializable: stored as a plain Python list.
    assert isinstance(omega_a, list)


def test_batched_scheduler_runs_all_jobs(tmp_path):
    """The scheduler executes a batched scan and produces one result per job."""
    system_config = SystemConfig(
        paths={
            "config_dirs": [str(tmp_path / "configs")],
            "output_dir": str(tmp_path / "runs"),
            "global_file": str(tmp_path / "global.yaml"),
            "plugin_dirs": [str(MODELS_DIR), str(tmp_path / "plugins")],
        },
        parameter_scan={"enabled": False},
    )
    scheduler = Scheduler(system_config=system_config)

    jobs = _make_scan_job_list([0.001, 0.002])
    results = scheduler.run(JobList(jobs=jobs))

    assert len(results) == 2
    assert all(r.success for r in results)
    assert {r.job_name for r in results} == {"vdp_scan_000", "vdp_scan_001"}


def test_batched_results_have_correct_shape_and_differ(tmp_path):
    """Batched results are split into correct per-scan shapes and differ."""
    param_values = [0.001, 0.002]

    system_config = SystemConfig(
        paths={
            "config_dirs": [str(tmp_path / "configs_batch")],
            "output_dir": str(tmp_path / "runs_batch"),
            "global_file": str(tmp_path / "global_batch.yaml"),
            "plugin_dirs": [str(MODELS_DIR), str(tmp_path / "plugins_batch")],
        },
        parameter_scan={"enabled": False},
    )
    scheduler = Scheduler(system_config=system_config)
    jobs = _make_scan_job_list(param_values)
    batch_results = scheduler.run(JobList(jobs=jobs))

    assert len(batch_results) == 2
    assert all(r.success for r in batch_results)

    trajs = []
    for idx in range(len(param_values)):
        traj_path = batch_results[idx].run_dir / f"vdp_scan_{idx:03d}.npz"
        traj = np.load(traj_path)["data"]
        # Original n_traj per scan point.
        assert traj.shape == (10, 2, 2)
        trajs.append(traj)

    # Different scan parameters must yield different final states.
    assert not np.allclose(trajs[0][:, -1, :], trajs[1][:, -1, :])
