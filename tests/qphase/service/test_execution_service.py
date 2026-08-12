from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.progress import ProgressSnapshot
from qphase.core.project import ProjectContext
from qphase.service.execution import ExecutionManager
from qphase.service.models import ExecutionPlan


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
