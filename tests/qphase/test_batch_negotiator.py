"""Tests for the batch negotiation framework."""

from __future__ import annotations

import pytest
from qphase.core.batch_negotiator import BatchJob, BatchNegotiator, SingleJob
from qphase.core.config import JobConfig
from qphase.core.registry import RegistryCenter, discovery, registry

pytestmark = pytest.mark.integration


def _make_scan_jobs(param_values: list[float]) -> list[JobConfig]:
    """Build a list of SDE scan jobs that differ only in omega_a."""
    jobs = []
    for idx, val in enumerate(param_values):
        jobs.append(
            JobConfig(
                name=f"vdp_scan_{idx:03d}",
                engine={
                    "sde": {
                        "t0": 0.0,
                        "t1": 1.0,
                        "dt": 0.001,
                        "n_traj": 100,
                        "seed": 42,
                        "ic": [["7.0+0.0j", "0.0-7.0j"]],
                    }
                },
                plugins={},
                backend={"cupy": {"float_dtype": "float32"}},
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
            )
        )
    return jobs


def test_no_planner_yields_single_jobs():
    """When no batch planner is registered, jobs stay single."""
    reg = RegistryCenter()
    negotiator = BatchNegotiator(reg)
    jobs = _make_scan_jobs([0.001, 0.002, 0.003])
    groups = negotiator.group_jobs(jobs)

    assert len(groups) == 3
    assert all(isinstance(g, SingleJob) for g in groups)


def test_downstream_dependency_prevents_batching():
    """A job referenced by a downstream input must not be swallowed."""
    reg = RegistryCenter()
    negotiator = BatchNegotiator(reg)
    jobs = _make_scan_jobs([0.001, 0.002])
    jobs.append(
        JobConfig(
            name="aggregate",
            engine={"sde": {"mode": "analyze"}},
            input="vdp_scan_000",
        )
    )
    groups = negotiator.group_jobs(jobs)

    # First job is referenced downstream -> SingleJob.
    assert isinstance(groups[0], SingleJob)
    # Second job has no downstream reference -> could be single (no planner).
    assert isinstance(groups[1], SingleJob)


def test_sde_batch_planner_can_batch_registered():
    """With the SDE batch planner registered, scan jobs are grouped."""
    # Discover entry points so the SDE batch planner is registered, and local
    # workspace plugins so the vdp_2mode model class can be inspected.
    discovery.discover_plugins()
    discovery.discover_local_plugins()
    negotiator = BatchNegotiator(registry)
    jobs = _make_scan_jobs([0.001, 0.002, 0.003])
    groups = negotiator.group_jobs(jobs)

    assert len(groups) == 1
    assert isinstance(groups[0], BatchJob)
    batch = groups[0]
    assert batch.plan.original_names == [j.name for j in jobs]
    assert batch.plan.result_splitter == "sde_scan_splitter"
    assert batch.plan.batch_job.engine["sde"]["n_traj"] == 300
