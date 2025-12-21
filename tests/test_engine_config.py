"""Test engine configuration format."""

import pytest
from qphase.core.config import JobConfig
from qphase.core.errors import QPhaseConfigError


def test_engine_config_single():
    """Test that a job can have a single engine configuration."""
    job = JobConfig(
        name="test_job",
        engine={"sde": {"t_end": 10.0, "n_steps": 1000}},
        params={},
    )

    assert job.get_engine_name() == "sde"
    assert job.engine["sde"]["t_end"] == 10.0


def test_engine_config_viz():
    """Test visualization engine configuration."""
    job = JobConfig(
        name="viz_job",
        engine={"viz": {"specs": [{"kind": "re_im"}]}},
        params={},
    )

    assert job.get_engine_name() == "viz"
    assert len(job.engine["viz"]["specs"]) == 1


def test_engine_config_lowercase():
    """Test that engine names are normalized to lowercase."""
    job = JobConfig(
        name="test_job",
        engine={"SDE": {"t_end": 10.0}},  # Uppercase
        params={},
    )

    # Engine name should be normalized to lowercase
    assert job.get_engine_name() == "sde"
    assert "sde" in job.engine
    assert "SDE" not in job.engine  # Original case should be replaced


def test_engine_config_multiple_error():
    """Test that multiple engines cause an error."""
    with pytest.raises(QPhaseConfigError, match="multiple"):
        JobConfig(
            name="test_job",
            engine={
                "sde": {"t_end": 10.0},
                "viz": {"specs": []},  # Second engine - should error
            },
            params={},
        )


def test_engine_config_empty_error():
    """Test that empty engine configuration causes an error."""
    with pytest.raises(QPhaseConfigError, match="empty"):
        JobConfig(
            name="test_job",
            engine={},  # Empty - should error
            params={},
        )
