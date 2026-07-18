"""Tests for configuration loading contract and plugin section extraction."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from qphase.core.config_loader import load_jobs_from_files
from qphase.core.errors import QPhaseConfigError
from qphase.core.registry import registry
from qphase.core.scheduler import Scheduler


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def register_dummy_integrator():
    """Register a dummy integrator for all tests in this module."""
    from tests.plugins.dummy_plugin import DummyPlugin

    registry.register(
        namespace="integrator", name="dummy", builder=DummyPlugin, overwrite=True
    )
    yield


@pytest.fixture
def register_sde_engine():
    """Ensure the real SDE engine is available in the registry."""
    from qphase_sde.engine import Engine

    registry.register(namespace="engine", name="sde", builder=Engine, overwrite=True)
    yield


@pytest.fixture
def dummy_job_dict():
    """Return a minimal job dict using dummy plugins."""
    return {
        "name": "dummy_job",
        "engine": {"sde": {"t1": 1.0, "dt": 0.01, "n_traj": 2}},
        "backend": {"dummy": {"param": 1.0}},
        "integrator": {"dummy": {"param": 1.0}},
        "model": {"dummy": {"param": 1.0}},
    }


def test_top_level_plugin_sections_extracted(register_sde_engine, dummy_job_file):
    """Top-level backend/integrator/model/analyser keys end up in JobConfig.plugins."""
    job_list = load_jobs_from_files([dummy_job_file])
    assert len(job_list.jobs) == 1
    job = job_list.jobs[0]

    assert "backend" in job.plugins
    assert "integrator" in job.plugins
    assert "model" in job.plugins
    assert job.plugins["backend"]["dummy"]["param"] == 1.0


@pytest.fixture
def dummy_job_file(temp_workspace, dummy_job_dict):
    """Create a dummy job file with top-level plugin sections."""
    job_file = temp_workspace / "configs" / "jobs" / "dummy_top_level.yaml"
    job_file.parent.mkdir(parents=True, exist_ok=True)
    with open(job_file, "w") as f:
        yaml.dump(dummy_job_dict, f)
    return job_file


def test_plugins_compatible_format_merged(dummy_job_dict):
    """The explicit ``plugins:`` block is merged with top-level plugin sections."""
    data = {
        "name": "compat_plugins",
        "engine": {"sde": {"t1": 1.0}},
        "plugins": {
            "model": {"dummy": {"param": 1.0}},
            "backend": {"dummy": {"param": 2.0}},
        },
        "integrator": {"dummy": {"param": 3.0}},
    }
    job_list = load_jobs_from_files([_write_job_file(data)])
    job = job_list.jobs[0]

    assert "model" in job.plugins
    assert "backend" in job.plugins
    assert "integrator" in job.plugins
    assert job.plugins["backend"]["dummy"]["param"] == 2.0
    assert job.plugins["integrator"]["dummy"]["param"] == 3.0


def _write_job_file(data: dict) -> Path:
    """Write a job dict to a temporary file and return its path."""
    import tempfile

    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    handle.close()
    path = Path(handle.name)
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def test_scheduler_validation_uses_top_level_plugins(
    register_sde_engine, register_dummy_integrator
):
    """Validation must see top-level plugin sections, not only ``job.plugins``."""
    data = {
        "name": "sde_validation",
        "engine": {"sde": {"t1": 1.0, "dt": 0.01, "n_traj": 2}},
        "backend": {"dummy": {"param": 1.0}},
        "integrator": {"dummy": {"param": 1.0}},
        "model": {"dummy": {"param": 1.0}},
    }
    job_list = load_jobs_from_files([_write_job_file(data)])
    scheduler = Scheduler()

    # Should not raise: backend/model/integrator are present as top-level keys.
    scheduler._validate_jobs(job_list)


def test_scheduler_validation_respects_global_defaults(
    register_sde_engine, register_dummy_integrator, temp_workspace
):
    """A required plugin supplied by global.yaml should not be reported missing."""
    global_file = temp_workspace / "configs" / "global.yaml"
    global_file.parent.mkdir(parents=True, exist_ok=True)
    with open(global_file, "w") as f:
        yaml.dump({"integrator": {"dummy": {"param": 1.0}}}, f)

    from qphase.core.system_config import load_system_config

    system_config = load_system_config(force_reload=True)

    data = {
        "name": "global_default_plugins",
        "engine": {"sde": {"t1": 1.0, "dt": 0.01, "n_traj": 2}},
        "backend": {"dummy": {"param": 1.0}},
        "model": {"dummy": {"param": 1.0}},
        # integrator is intentionally omitted; global.yaml provides it.
    }
    job_list = load_jobs_from_files([_write_job_file(data)])
    scheduler = Scheduler(system_config=system_config)

    scheduler._validate_jobs(job_list)


def test_scheduler_validation_still_reports_missing_required_plugin(
    register_sde_engine, register_dummy_integrator
):
    """If a required plugin is truly missing, validation must still fail."""
    data = {
        "name": "missing_model",
        "engine": {"sde": {"t1": 1.0, "dt": 0.01, "n_traj": 2}},
        "backend": {"dummy": {"param": 1.0}},
        "integrator": {"dummy": {"param": 1.0}},
        # model is missing
    }
    job_list = load_jobs_from_files([_write_job_file(data)])
    scheduler = Scheduler()

    with pytest.raises(QPhaseConfigError, match="missing required plugins"):
        scheduler._validate_jobs(job_list)
