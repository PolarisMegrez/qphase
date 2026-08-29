from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from qphase.core.catalog import CatalogQuery, ProjectObjectCatalog
from qphase.core.compiler import WorkflowCompiler
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.errors import QPhaseConfigError
from qphase.core.persistence import ProjectStateStore
from qphase.core.progress import ProgressSnapshot
from qphase.core.project import ProjectContext
from qphase.core.system_config import SystemConfig
from qphase.core.tags import execution_tag_assignment_id
from qphase.service.execution import ExecutionManager
from qphase.service.models import ExecutionPlan
from qphase.service.scheduler import SchedulerService


def _job(name: str, value: float) -> JobConfig:
    return JobConfig.model_validate(
        {
            "name": name,
            "engine": {"dummy": {}},
            "plugins": {
                "backend": {"dummy": {}},
                "model": {"dummy": {"param": value}},
            },
        }
    )


class _BoundaryScheduler:
    def __init__(self) -> None:
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.second_value: float | None = None
        self.system_config = SimpleNamespace(
            reporting=SimpleNamespace(progress=SimpleNamespace(refresh_interval=0.01))
        )
        self.project = ProjectContext.discover()

    def load_workflow(self, _: str) -> WorkflowSpec:
        return WorkflowSpec(
            schema="qphase.workflow/2",
            id="test-workflow",
            title="Test Workflow",
            jobs=[_job("first", 1.0), _job("second", 2.0)],
        )

    def build_plan(self, _: WorkflowSpec) -> ExecutionPlan:
        return ExecutionPlan()

    def run(
        self,
        job_list: WorkflowSpec,
        progress_callback: Any,
        cancellation: Any,
        before_job: Any,
        on_scheduler: Any,
        **_: Any,
    ) -> list[Any]:
        scheduler = SimpleNamespace(session_id=None, session_dir=None)
        on_scheduler(scheduler)
        for index, original in enumerate(job_list.jobs):
            job = before_job(original, index, len(job_list.jobs))
            progress_callback(
                ProgressSnapshot(
                    kind="job_started",
                    job_name=job.name,
                    job_index=index,
                    total_jobs=2,
                    stage="running",
                )
            )
            if job.name == "first":
                self.first_started.set()
                assert self.release_first.wait(2.0)
            else:
                self.second_value = job.plugins["model"]["dummy"]["param"]
            progress_callback(
                ProgressSnapshot(
                    kind="job_completed",
                    job_name=job.name,
                    job_index=index,
                    total_jobs=2,
                    stage="completed",
                )
            )
        return [SimpleNamespace(status="completed")]


def _wait_for_state(manager: ExecutionManager, execution_id: str, state: str) -> None:
    for _ in range(200):
        if manager.get(execution_id).state == state:
            return
        time.sleep(0.01)
    pytest.fail(f"execution did not enter {state!r}")


def test_pause_boundary_allows_pending_job_revision() -> None:
    scheduler = _BoundaryScheduler()
    manager = ExecutionManager(scheduler)  # type: ignore[arg-type]
    try:
        execution = manager.submit("ignored")
        assert scheduler.first_started.wait(1.0)
        manager.request_pause(execution.execution_id)
        scheduler.release_first.set()
        _wait_for_state(manager, execution.execution_id, "paused")

        revised = _job("second", 9.0).model_dump(mode="python")
        manager.revise_pending_job(execution.execution_id, "second", revised)
        with pytest.raises(ValueError, match="already started"):
            manager.revise_pending_job(
                execution.execution_id,
                "first",
                _job("first", 8.0).model_dump(mode="python"),
            )
        manager.resume(execution.execution_id)
        _wait_for_state(manager, execution.execution_id, "completed")

        assert scheduler.second_value == 9.0
        kinds = [
            event.payload["kind"] for event in manager.events(execution.execution_id)
        ]
        assert "execution_paused" in kinds
        assert "pending_job_revised" in kinds
    finally:
        manager.close()


def test_invalid_execution_plan_is_rejected_before_queueing() -> None:
    scheduler = _BoundaryScheduler()
    scheduler.build_plan = lambda _: ExecutionPlan(  # type: ignore[method-assign]
        validation_issues=[
            {"path": "jobs", "message": "invalid dependency", "source": "test"}
        ]
    )
    manager = ExecutionManager(scheduler)  # type: ignore[arg-type]
    try:
        with pytest.raises(ValueError, match="invalid dependency"):
            manager.submit("ignored")
    finally:
        manager.close()


def test_manager_marks_running_record_interrupted_on_restart(tmp_path) -> None:
    project = ProjectContext.create(tmp_path / "project")
    scheduler = _BoundaryScheduler()
    scheduler.project = project
    scheduler.state_store = ProjectStateStore(project)
    workflow = scheduler.load_workflow("ignored")
    compiled = WorkflowCompiler(project, SystemConfig()).compile(workflow)
    scheduler.state_store.save_execution(
        {
            "schema": "qphase.execution/1",
            "execution_id": "interrupted-1",
            "source_workflow": "test-workflow",
            "workflow": workflow.model_dump(mode="json", by_alias=True),
            "compiled_workflow": compiled.to_payload(),
            "submitted_at": "2026-08-26T10:00:00+08:00",
            "state": "running",
            "session_id": None,
            "session_dir": None,
            "started_at": "2026-08-26T10:01:00+08:00",
            "finished_at": None,
            "current_job": "first",
            "current_stage": "running",
            "latest_message": "working",
            "error": None,
            "pause_requested": False,
            "started_jobs": ["first"],
            "revisions": {},
        }
    )

    manager = ExecutionManager(scheduler)  # type: ignore[arg-type]
    try:
        summary = manager.get("interrupted-1")
        assert summary.state == "failed"
        assert summary.error == "execution worker interrupted by process restart"
        assert scheduler.state_store.load_executions()[0]["state"] == "failed"
    finally:
        manager.close()


def test_submission_tags_are_validated_at_submit() -> None:
    scheduler = _BoundaryScheduler()
    manager = ExecutionManager(scheduler)  # type: ignore[arg-type]
    try:
        with pytest.raises(QPhaseConfigError, match="namespace:value"):
            manager.submit("ignored", tags=["no-namespace"])
    finally:
        manager.close()


def test_submission_tags_update_while_queued_and_freeze_on_start() -> None:
    scheduler = _BoundaryScheduler()
    manager = ExecutionManager(scheduler)  # type: ignore[arg-type]
    try:
        first = manager.submit("ignored", tags=["Task:Urgent"])
        assert first.submission_tags == ["task:urgent"]
        assert scheduler.first_started.wait(1.0)

        second = manager.submit("ignored", tags=["task:queued"])
        updated = manager.update_submission_tags(
            second.execution_id, ["task:revised"]
        )
        assert updated.submission_tags == ["task:revised"]
        with pytest.raises(ValueError, match="while queued"):
            manager.update_submission_tags(first.execution_id, ["task:late"])

        scheduler.release_first.set()
        _wait_for_state(manager, first.execution_id, "completed")
        _wait_for_state(manager, second.execution_id, "completed")

        first_kinds = [
            event.payload["kind"] for event in manager.events(first.execution_id)
        ]
        assert "submission_tags_frozen" in first_kinds
        second_payloads = [
            event.payload for event in manager.events(second.execution_id)
        ]
        kinds = [payload["kind"] for payload in second_payloads]
        assert "submission_tags_updated" in kinds
        frozen = second_payloads[kinds.index("submission_tags_frozen")]
        assert frozen["tags"] == ["task:revised"]
    finally:
        manager.close()


def test_submission_tags_persist_in_execution_record(tmp_path) -> None:
    project = ProjectContext.create(tmp_path / "project")
    scheduler = _BoundaryScheduler()
    scheduler.project = project
    scheduler.state_store = ProjectStateStore(project)
    scheduler.compile_workflow = lambda workflow: WorkflowCompiler(  # type: ignore[method-assign]
        project, SystemConfig()
    ).compile(workflow)
    manager = ExecutionManager(scheduler)  # type: ignore[arg-type]
    try:
        blocker = manager.submit("ignored")
        assert scheduler.first_started.wait(1.0)
        # The worker is parked inside the first execution, so the queued
        # record is stable on disk (Windows forbids replace-during-read).
        execution = manager.submit("ignored", tags=["task:urgent"])
        payload = next(
            item
            for item in scheduler.state_store.load_executions()
            if item["execution_id"] == execution.execution_id
        )
        assert payload["submission_tags"] == ["task:urgent"]
        scheduler.release_first.set()
        _wait_for_state(manager, blocker.execution_id, "completed")
        _wait_for_state(manager, execution.execution_id, "completed")

        restored = ExecutionManager(scheduler)  # type: ignore[arg-type]
        try:
            assert restored.get(execution.execution_id).submission_tags == [
                "task:urgent"
            ]
        finally:
            restored.close()
    finally:
        manager.close()


def _direct_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        schema="qphase.workflow/2",
        id="direct-run",
        title="Direct Run",
        jobs=[JobConfig(name="sim", engine={"dummy": {}})],
    )


def test_direct_run_opens_execution_record_and_links_session(tmp_path) -> None:
    """A synchronous SchedulerService run owns a full execution record.

    The record precedes the session, the session manifest links back to it,
    and both persist the same deterministic submission-tag assignment ids.
    """
    project = ProjectContext.create(tmp_path / "project")
    service = SchedulerService(SystemConfig(), project=project)
    workflow = _direct_workflow()

    results = service.run(workflow, submission_tags=["task:urgent"])

    assert all(result.success for result in results)
    records = service.state_store.load_executions()
    assert len(records) == 1
    record = records[0]
    assert record["state"] == "completed"
    handle = service.last_session_handle
    assert handle is not None and handle.session_dir is not None
    assert record["session_id"] == handle.session_id
    manifest = service.state_store.load_session_manifest(Path(handle.session_dir))
    assert manifest["execution_id"] == record["execution_id"]
    expected = {
        "task:urgent": execution_tag_assignment_id(
            record["execution_id"], "task:urgent"
        )
    }
    assert record["submission_tag_assignments"] == expected
    assert manifest["submission_tag_assignments"] == expected

    catalog = ProjectObjectCatalog(project)
    sessions = catalog.query(CatalogQuery(object_kind="session"))
    assert [row["execution_id"] for row in sessions] == [record["execution_id"]]
    executions = catalog.query(CatalogQuery(object_kind="execution"))
    assert [row["id"] for row in executions] == [record["execution_id"]]
    execution_tags = {
        tag.tag: tag
        for tag in catalog.effective_tags("execution", record["execution_id"])
    }
    assert execution_tags["task:urgent"].assignment_id == expected["task:urgent"]
    session_tags = {
        tag.tag: tag
        for tag in catalog.effective_tags("session", str(handle.session_id))
    }
    assert session_tags["task:urgent"].assignment_id == expected["task:urgent"]
    assert session_tags["task:urgent"].inherited


def test_plan_and_dry_run_create_no_execution_record(tmp_path) -> None:
    project = ProjectContext.create(tmp_path / "project")
    service = SchedulerService(SystemConfig(), project=project)
    workflow = _direct_workflow()

    service.build_plan(workflow)
    service.dry_run(workflow)

    assert service.state_store.load_executions() == []


def test_queued_run_reuses_execution_identity(tmp_path) -> None:
    """The queued path passes its record identity down to the session.

    Exactly one record exists afterwards: the scheduler service must not
    open a second execution for the same run.
    """
    project = ProjectContext.create(tmp_path / "project")
    workflow_root = project.workflow_root
    workflow_root.mkdir(parents=True, exist_ok=True)
    (workflow_root / "example.yaml").write_text(
        "schema: qphase.workflow/2\n"
        "id: example\n"
        "title: Example\n"
        "jobs:\n"
        "  - name: sim\n"
        "    engine:\n"
        "      dummy: {}\n",
        encoding="utf-8",
    )
    service = SchedulerService(SystemConfig(), project=project)
    manager = ExecutionManager(service)
    try:
        execution = manager.submit("example", tags=["task:queued"])
        _wait_for_state(manager, execution.execution_id, "completed")
    finally:
        manager.close()

    records = service.state_store.load_executions()
    assert len(records) == 1
    record = records[0]
    assert record["execution_id"] == execution.execution_id
    assert record["state"] == "completed"
    assert record["session_dir"] is not None
    manifest = service.state_store.load_session_manifest(
        project.session_root / record["session_dir"]
    )
    assert manifest["execution_id"] == execution.execution_id
    expected = {
        "task:queued": execution_tag_assignment_id(
            execution.execution_id, "task:queued"
        )
    }
    assert manifest["submission_tag_assignments"] == expected
    assert record["submission_tag_assignments"] == expected


def test_update_submission_tags_regenerates_assignment_ids(tmp_path) -> None:
    """Replacing queued tags retires old ids and derives fresh ones."""
    project = ProjectContext.create(tmp_path / "project")
    scheduler = _BoundaryScheduler()
    scheduler.project = project
    scheduler.state_store = ProjectStateStore(project)
    scheduler.compile_workflow = lambda workflow: WorkflowCompiler(  # type: ignore[method-assign]
        project, SystemConfig()
    ).compile(workflow)
    manager = ExecutionManager(scheduler)  # type: ignore[arg-type]
    try:
        blocker = manager.submit("ignored")
        assert scheduler.first_started.wait(1.0)
        # The worker is parked inside the first execution, so the queued
        # record is stable on disk (Windows forbids replace-during-read).
        execution = manager.submit("ignored", tags=["task:alpha"])
        manager.update_submission_tags(execution.execution_id, ["task:beta"])
        payload = next(
            item
            for item in scheduler.state_store.load_executions()
            if item["execution_id"] == execution.execution_id
        )
        assert payload["submission_tags"] == ["task:beta"]
        assert payload["submission_tag_assignments"] == {
            "task:beta": execution_tag_assignment_id(
                execution.execution_id, "task:beta"
            )
        }
        scheduler.release_first.set()
        _wait_for_state(manager, blocker.execution_id, "completed")
        _wait_for_state(manager, execution.execution_id, "completed")
    finally:
        manager.close()
