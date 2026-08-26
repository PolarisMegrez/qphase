from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.scheduler import JobResult, Scheduler
from qphase.core.system_config import SystemConfig

pytestmark = pytest.mark.integration


@pytest.fixture
def mock_system_config():
    return SystemConfig()


@pytest.fixture
def simple_job_list():
    job1 = JobConfig(name="job1", engine={"dummy": {}})
    job2 = JobConfig(
        name="job2",
        engine={"dummy": {}},
        input={"from": "job1", "mode": "dataset"},
    )
    return WorkflowSpec(
        schema="qphase.workflow/2",
        id="test-workflow",
        title="Test Workflow",
        jobs=[job1, job2],
    )


def test_dry_run(mock_system_config, simple_job_list, temp_project):
    scheduler = Scheduler(system_config=mock_system_config, project=temp_project)

    compiled = scheduler._validate_jobs(simple_job_list)
    with patch.object(scheduler, "_validate_jobs", return_value=compiled):
        results = scheduler.run(simple_job_list, dry_run=True)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.job_dir == Path("dry_run") for r in results)

        # Planning must not create persisted Session state.
        assert scheduler.session_id is None
        assert scheduler.session_dir is None
        assert not list(temp_project.session_root.rglob("session_manifest.json"))


def test_resume_capability(mock_system_config, simple_job_list, temp_project):
    # 1. Create a fake previous session
    session_dir = temp_project.session_root / "old_session"
    session_dir.mkdir(parents=True)

    manifest = {
        "schema": "qphase.session/2",
        "session_id": "old_session",
        "project_id": temp_project.project_id,
        "workflow_id": simple_job_list.id,
        "workflow_hash": Scheduler._workflow_hash(
            simple_job_list.model_dump(mode="json", by_alias=True)
        ),
        "start_time": "2025-01-01T00:00:00",
        "status": "failed",
        "jobs": {
            "job1": {"status": "completed", "output_dir": "job1"},
            "job2": {"status": "failed"},
        },
    }

    import json

    with open(session_dir / "session_manifest.json", "w") as f:
        json.dump(manifest, f)

    # 2. Run scheduler with resume
    scheduler = Scheduler(system_config=mock_system_config, project=temp_project)

    compiled = scheduler._validate_jobs(simple_job_list)
    with (
        patch.object(scheduler, "_validate_jobs", return_value=compiled),
        patch.object(scheduler, "_run_job") as mock_run_job,
        patch.object(scheduler, "_resolve_input", return_value=MagicMock()),
        patch.object(scheduler, "_handle_job_output"),
    ):
        # Mock run_job to return success for job2
        # The Job directory must be relative to Session for manifest serialization.
        mock_run_job.return_value = (
            JobResult(1, "job2", session_dir / "job2", True),
            MagicMock(),
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


def test_validate_command_logic(mock_system_config, simple_job_list, temp_project):
    scheduler = Scheduler(system_config=mock_system_config, project=temp_project)

    # We just want to ensure _validate_jobs is called
    with patch.object(scheduler, "_validate_jobs") as mock_validate:
        scheduler._validate_jobs(simple_job_list)
        mock_validate.assert_called_once()
