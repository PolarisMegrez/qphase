"""Test scheduler validation logic."""

import pytest
from qphase.core.config_loader import load_jobs_from_files
from qphase.core.errors import QPhaseConfigError
from qphase.core.scheduler import Scheduler


@pytest.fixture
def valid_job_file(temp_workspace):
    """Create a valid job file."""
    job_file = temp_workspace / "configs" / "jobs" / "valid_job.yaml"
    job_file.parent.mkdir(parents=True, exist_ok=True)

    import yaml

    with open(job_file, "w") as f:
        yaml.dump(
            {
                "name": "valid_job",
                "engine": {"dummy": {"param": 10.0}},
                "backend": {"dummy": {"param": 1.0}},
                "model": {"dummy": {"param": 1.0}},
            },
            f,
        )
    return job_file


@pytest.fixture
def invalid_job_file(temp_workspace):
    """Create an invalid job file (multiple engines)."""
    job_file = temp_workspace / "configs" / "jobs" / "invalid_job.yaml"
    job_file.parent.mkdir(parents=True, exist_ok=True)

    import yaml

    with open(job_file, "w") as f:
        yaml.dump(
            {
                "name": "invalid_job",
                "engine": {
                    "dummy": {"param": 10.0},
                    "viz": {"param": 1.0},  # Second engine
                },
                "backend": {"dummy": {"param": 1.0}},
                "model": {"dummy": {"param": 1.0}},
            },
            f,
        )
    return job_file


def test_job_validation_success(valid_job_file, dummy_model):
    """Test successful validation."""
    job_list = load_jobs_from_files([valid_job_file])
    scheduler = Scheduler()

    # Should not raise exception
    scheduler._validate_jobs(job_list)


def test_job_validation_failure(invalid_job_file, dummy_model):
    """Test validation failure."""
    # Note: The loader might catch multiple engines first if Pydantic model enforces it
    # But if it passes loader, scheduler should catch it

    try:
        job_list = load_jobs_from_files([invalid_job_file])
        scheduler = Scheduler()

        with pytest.raises(QPhaseConfigError):
            scheduler._validate_jobs(job_list)

    except QPhaseConfigError:
        # If loader catches it, that's also fine
        pass
