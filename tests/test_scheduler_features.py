from unittest.mock import MagicMock, patch

import pytest
from qphase.core.config import JobConfig, JobList
from qphase.core.scheduler import JobResult, Scheduler
from qphase.core.system_config import SystemConfig


@pytest.fixture
def mock_system_config(tmp_path):
    config = MagicMock(spec=SystemConfig)
    # Mock the paths attribute which is a nested object
    config.paths = MagicMock()
    config.paths.output_dir = str(tmp_path / "runs")
    config.parameter_scan = {"enabled": True, "method": "cartesian"}
    config.auto_save_results = True
    config.progress_update_interval = 0.1
    return config


@pytest.fixture
def simple_job_list():
    job1 = JobConfig(name="job1", engine={"test_engine": {}})
    job2 = JobConfig(name="job2", engine={"test_engine": {}}, input="job1")
    return JobList(jobs=[job1, job2])


def test_dry_run(mock_system_config, simple_job_list):
    scheduler = Scheduler(system_config=mock_system_config)

    # Mock validation and expansion to avoid needing real plugins
    with (
        patch.object(scheduler, "_validate_jobs"),
        patch.object(
            scheduler, "_expand_parameter_scans", return_value=simple_job_list.jobs
        ),
    ):
        results = scheduler.run(simple_job_list, dry_run=True)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.run_id == "dry_run" for r in results)

        # Verify session was initialized
        assert scheduler.session_id is not None
        assert scheduler.session_dir is not None
        assert scheduler.session_dir.exists()


def test_resume_capability(mock_system_config, simple_job_list, tmp_path):
    # 1. Create a fake previous session
    session_dir = tmp_path / "runs" / "old_session"
    session_dir.mkdir(parents=True)

    manifest = {
        "session_id": "old_session",
        "start_time": "2025-01-01T00:00:00",
        "status": "failed",
        "jobs": {
            "job1": {"status": "completed", "run_id": "run1", "output_dir": "job1"},
            "job2": {"status": "failed"},
        },
    }

    import json

    with open(session_dir / "session_manifest.json", "w") as f:
        json.dump(manifest, f)

    # 2. Run scheduler with resume
    scheduler = Scheduler(system_config=mock_system_config)

    with (
        patch.object(scheduler, "_validate_jobs"),
        patch.object(
            scheduler, "_expand_parameter_scans", return_value=simple_job_list.jobs
        ),
        patch.object(scheduler, "_run_job") as mock_run_job,
        patch.object(scheduler, "_resolve_input", return_value=MagicMock()),
        patch.object(scheduler, "_handle_job_output"),
    ):
        # Mock run_job to return success for job2
        # Note: run_dir must be relative to session_dir for relative_to check to pass
        mock_run_job.return_value = (
            JobResult(1, "job2", session_dir / "job2", "run2", True),
            MagicMock(),
        )

        results = scheduler.run(simple_job_list, resume_from=session_dir)

        # Should only run job2
        assert len(results) == 1
        assert results[0].job_name == "job2"
        assert mock_run_job.call_count == 1
        assert mock_run_job.call_args[0][0].name == "job2"

        # Verify manifest updated
        with open(session_dir / "session_manifest.json") as f:
            new_manifest = json.load(f)
            assert new_manifest["jobs"]["job2"]["status"] == "completed"
            assert new_manifest["status"] == "completed"


def test_validate_command_logic(mock_system_config, simple_job_list):
    scheduler = Scheduler(system_config=mock_system_config)

    # We just want to ensure _validate_jobs is called
    with patch.object(scheduler, "_validate_jobs") as mock_validate:
        scheduler._validate_jobs(simple_job_list)
        mock_validate.assert_called_once()
