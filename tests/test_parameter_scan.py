"""Test parameter scan functionality."""

import pytest
from qphase.core.config_loader import load_jobs_from_files
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig


@pytest.fixture
def cartesian_job_file(temp_workspace):
    """Create a job file for cartesian scan."""
    job_file = temp_workspace / "configs" / "jobs" / "test_cartesian.yaml"
    job_file.parent.mkdir(parents=True, exist_ok=True)

    import yaml

    with open(job_file, "w") as f:
        yaml.dump(
            {
                "name": "test_cartesian",
                "engine": {
                    "dummy": {
                        "param": [5.0, 10.0]  # 2 values
                    }
                },
                "backend": {"dummy": {"param": 1.0}},
                "model": {"dummy": {"param": [1.0, 2.0]}},  # 2 values
            },
            f,
        )
    return job_file


@pytest.fixture
def zipped_job_file(temp_workspace):
    """Create a job file for zipped scan."""
    job_file = temp_workspace / "configs" / "jobs" / "test_zipped.yaml"
    job_file.parent.mkdir(parents=True, exist_ok=True)

    import yaml

    with open(job_file, "w") as f:
        yaml.dump(
            {
                "name": "test_zipped",
                "engine": {
                    "dummy": {
                        "param": [5.0, 10.0, 15.0]  # 3 values
                    }
                },
                "backend": {"dummy": {"param": 1.0}},
                "model": {"dummy": {"param": [1.0, 2.0, 3.0]}},  # 3 values
            },
            f,
        )
    return job_file


def test_cartesian_expansion(cartesian_job_file, dummy_model):
    """Test cartesian product expansion."""
    job_list = load_jobs_from_files([cartesian_job_file])
    assert len(job_list.jobs) == 1

    # Create scheduler with cartesian method (default)
    scheduler = Scheduler()

    # Expand jobs
    expanded = scheduler._expand_parameter_scans(job_list)

    # Expected: 2 (t_end) * 2 (dt) = 4 jobs
    assert len(expanded) == 4

    # Verify names
    assert expanded[0].name == "test_cartesian_001"
    assert expanded[3].name == "test_cartesian_004"


def test_zipped_expansion(zipped_job_file, dummy_model):
    """Test zipped expansion."""
    job_list = load_jobs_from_files([zipped_job_file])
    assert len(job_list.jobs) == 1

    # Create scheduler with zipped method
    system_config = SystemConfig(parameter_scan={"method": "zipped"})
    scheduler = Scheduler(system_config=system_config)

    # Expand jobs
    expanded = scheduler._expand_parameter_scans(job_list)

    # Expected: 3 jobs (aligned lists of length 3)
    assert len(expanded) == 3

    # Verify values
    # Job 1: engine.param=5.0, model.param=1.0
    assert expanded[0].engine["dummy"]["param"] == 5.0
    assert expanded[0].plugins["model"]["dummy"]["param"] == 1.0

    # Job 3: param=15.0, model.param=3.0
    assert expanded[2].engine["dummy"]["param"] == 15.0
    assert expanded[2].plugins["model"]["dummy"]["param"] == 3.0
