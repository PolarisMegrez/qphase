"""Local workstation execution queue and durable progress journal."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.execution import CancellationController
from qphase.core.progress import ProgressSnapshot
from qphase.core.scheduler import Scheduler

from .models import (
    ExecutionEvent,
    ExecutionJobState,
    ExecutionSummary,
    PluginActivity,
)
from .scheduler import SchedulerService

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now().astimezone()


@dataclass
class _ExecutionRecord:
    execution_id: str
    workflow: WorkflowSpec
    source_workflow: str
    submitted_at: datetime = field(default_factory=_now)
    state: str = "queued"
    session_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    current_job: str | None = None
    current_stage: str | None = None
    latest_message: str = ""
    error: str | None = None
    controller: CancellationController = field(default_factory=CancellationController)
    events: deque[ExecutionEvent] = field(default_factory=lambda: deque(maxlen=1000))
    sequence: int = 0
    scheduler: Scheduler | None = None
    pause_requested: bool = False
    gate: threading.Condition = field(default_factory=threading.Condition)
    revisions: dict[str, JobConfig] = field(default_factory=dict)
    started_jobs: set[str] = field(default_factory=set)
    last_persisted_progress: float = 0.0
    persisted_sequence: int = 0


class ExecutionManager:
    """Execute one Workflow at a time and keep a bounded FIFO queue."""

    def __init__(
        self,
        scheduler: SchedulerService,
        *,
        queue_capacity: int = 16,
        retained_executions: int = 100,
    ) -> None:
        self.scheduler = scheduler
        self.queue_capacity = queue_capacity
        self.retained_executions = retained_executions
        self._records: dict[str, _ExecutionRecord] = {}
        self._queue: deque[str] = deque()
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._closed = False
        self._worker = threading.Thread(
            target=self._worker_loop, name="qphase-run-manager", daemon=True
        )
        self._worker.start()

    def close(self) -> None:
        with self._wake:
            self._closed = True
            self._wake.notify_all()
        self._worker.join(timeout=5.0)

    def submit(
        self, workflow_reference: str, *, resume_from: str | None = None
    ) -> ExecutionSummary:
        del resume_from  # Resume support remains on the synchronous service for now.
        workflow = self.scheduler.load_workflow(workflow_reference)
        self._validate_plan(workflow)
        with self._wake:
            if len(self._queue) >= self.queue_capacity:
                raise RuntimeError("execution queue is full")
            execution_id = uuid.uuid4().hex[:12]
            record = _ExecutionRecord(
                execution_id=execution_id,
                workflow=workflow,
                source_workflow=workflow_reference,
            )
            self._records[execution_id] = record
            self._queue.append(execution_id)
            self._append_event(record, {"kind": "execution_queued"}, persist=False)
            self._trim_records()
            self._wake.notify()
            return self._summary(record)

    def list_executions(self) -> list[ExecutionSummary]:
        with self._lock:
            return [self._summary(record) for record in self._records.values()]

    def get(self, execution_id: str) -> ExecutionSummary:
        with self._lock:
            return self._summary(self._require(execution_id))

    def events(self, execution_id: str, *, after: int = 0) -> list[ExecutionEvent]:
        with self._lock:
            record = self._require(execution_id)
            return [event for event in record.events if event.sequence > after]

    def cancel(self, execution_id: str) -> ExecutionSummary:
        with self._wake:
            record = self._require(execution_id)
            record.controller.cancel_execution()
            if record.state == "queued":
                if execution_id in self._queue:
                    self._queue.remove(execution_id)
                record.state = "cancelled"
                record.finished_at = _now()
            with record.gate:
                record.gate.notify_all()
            self._append_event(record, {"kind": "cancellation_requested"})
            return self._summary(record)

    def cancel_job(self, execution_id: str, job_name: str) -> ExecutionSummary:
        with self._lock:
            record = self._require(execution_id)
            record.controller.cancel_job(job_name)
            self._append_event(
                record, {"kind": "job_cancellation_requested", "job_name": job_name}
            )
            return self._summary(record)

    def request_pause(self, execution_id: str) -> ExecutionSummary:
        with self._lock:
            record = self._require(execution_id)
            if record.state not in {"queued", "running"}:
                raise ValueError("only queued or running executions can be paused")
            record.pause_requested = True
            if record.state == "queued":
                record.state = "paused"
            else:
                record.state = "pause_requested"
            self._append_event(record, {"kind": "pause_requested"})
            return self._summary(record)

    def resume(self, execution_id: str) -> ExecutionSummary:
        record = self._require(execution_id)
        with record.gate:
            if record.state != "paused":
                raise ValueError("execution is not paused at a job boundary")
            record.pause_requested = False
            record.state = "running" if record.started_at else "queued"
            record.gate.notify_all()
        with self._wake:
            if record.started_at is None and execution_id not in self._queue:
                self._queue.append(execution_id)
                self._wake.notify()
        self._append_event(record, {"kind": "execution_resumed"})
        return self._summary(record)

    def revise_pending_job(
        self, execution_id: str, job_name: str, payload: dict[str, Any]
    ) -> ExecutionSummary:
        record = self._require(execution_id)
        if record.state not in {"queued", "paused"}:
            raise ValueError("pending jobs can only be revised while queued or paused")
        if job_name in record.started_jobs:
            raise ValueError(f"job {job_name!r} has already started")
        replacement = JobConfig.model_validate(payload)
        if replacement.name != job_name:
            raise ValueError("the revised job must preserve its logical name")
        jobs = [
            replacement
            if item.name == job_name
            else record.revisions.get(item.name, item)
            for item in record.workflow.jobs
        ]
        if not any(item.name == job_name for item in record.workflow.jobs):
            raise KeyError(f"unknown job {job_name!r}")
        self._validate_plan(record.workflow.model_copy(update={"jobs": jobs}))
        record.revisions[job_name] = replacement
        self._append_event(
            record, {"kind": "pending_job_revised", "job_name": job_name}
        )
        return self._summary(record)

    def _worker_loop(self) -> None:
        while True:
            with self._wake:
                self._wake.wait_for(lambda: self._closed or bool(self._queue))
                if self._closed:
                    return
                execution_id = self._queue.popleft()
                record = self._records[execution_id]
                if record.state == "paused":
                    continue
            self._execute(record)

    def _execute(self, record: _ExecutionRecord) -> None:
        record.state = "running"
        record.started_at = _now()
        self._append_event(record, {"kind": "execution_started"}, persist=False)

        def _scheduler_ready(scheduler: Scheduler) -> None:
            record.scheduler = scheduler

        def _before_job(job: JobConfig, index: int, total: int) -> JobConfig:
            del index, total
            with record.gate:
                if record.pause_requested and not record.controller.execution.cancelled:
                    record.state = "paused"
                    self._append_event(
                        record,
                        {"kind": "execution_paused", "before_job": job.name},
                    )
                    record.gate.wait_for(
                        lambda: not record.pause_requested
                        or record.controller.execution.cancelled
                    )
                    if not record.controller.execution.cancelled:
                        record.state = "running"
                record.started_jobs.add(job.name)
                return record.revisions.get(job.name, job)

        try:
            results = self.scheduler.run(
                record.workflow,
                progress_callback=lambda snapshot: self._on_progress(record, snapshot),
                cancellation=record.controller,
                before_job=_before_job,
                on_scheduler=_scheduler_ready,
            )
            record.session_id = (
                record.scheduler.session_id if record.scheduler is not None else None
            )
            statuses = {result.status for result in results}
            record.state = (
                "failed"
                if "failed" in statuses
                else "cancelled"
                if "cancelled" in statuses
                else "partial"
                if "skipped_dependency" in statuses
                else "completed"
            )
        except Exception as exc:
            record.state = "failed"
            record.error = str(exc)
            self._append_event(record, {"kind": "execution_failed", "error": str(exc)})
        finally:
            record.finished_at = _now()
            self._append_event(record, {"kind": f"execution_{record.state}"})

    def _on_progress(
        self, record: _ExecutionRecord, snapshot: ProgressSnapshot
    ) -> None:
        record.current_job = snapshot.job_name
        record.current_stage = snapshot.stage
        record.latest_message = snapshot.message
        if record.scheduler is not None:
            record.session_id = record.scheduler.session_id
        payload = snapshot.to_dict()
        now = time.monotonic()
        persist = snapshot.kind != "job_progress" or (
            now - record.last_persisted_progress
            >= self.scheduler.system_config.reporting.progress.refresh_interval
        )
        if persist:
            record.last_persisted_progress = now
        self._append_event(record, payload, persist=persist)

    def _append_event(
        self, record: _ExecutionRecord, payload: dict[str, Any], *, persist: bool = True
    ) -> None:
        with self._lock:
            record.sequence += 1
            event = ExecutionEvent(
                sequence=record.sequence,
                timestamp=_now(),
                execution_id=record.execution_id,
                session_id=record.session_id,
                payload=deepcopy(payload),
            )
            record.events.append(event)
            if (
                persist
                and record.scheduler is not None
                and record.scheduler.session_dir
            ):
                self._persist_events(record)

    def _persist_events(self, record: _ExecutionRecord) -> None:
        assert record.scheduler is not None
        assert record.scheduler.session_dir is not None
        pending = [
            event
            for event in record.events
            if event.sequence > record.persisted_sequence
        ]
        if not pending:
            return
        path = record.scheduler.session_dir / "events.jsonl"
        try:
            with path.open("a", encoding="utf-8") as handle:
                for event in pending:
                    handle.write(
                        json.dumps(event.model_dump(mode="python"), default=str) + "\n"
                    )
            record.persisted_sequence = pending[-1].sequence
        except OSError as exc:
            # Monitoring must never turn an otherwise valid numerical run into a
            # failed execution. The in-memory event ring remains available.
            log.warning("Failed to persist execution events: %s", exc)

    def _validate_plan(self, workflow: WorkflowSpec) -> None:
        plan = self.scheduler.build_plan(workflow)
        if plan.validation_issues:
            details = "; ".join(
                f"{issue.path}: {issue.message}" for issue in plan.validation_issues
            )
            raise ValueError(f"execution plan is invalid: {details}")

    def _summary(self, record: _ExecutionRecord) -> ExecutionSummary:
        positions = list(self._queue)
        position = (
            positions.index(record.execution_id) + 1
            if record.execution_id in positions
            else None
        )
        latest_by_job: dict[str, dict[str, Any]] = {}
        for event in record.events:
            job_name = event.payload.get("job_name")
            if job_name:
                latest_by_job[str(job_name)] = event.payload
        jobs = []
        for original in record.workflow.jobs:
            configured = record.revisions.get(original.name, original)
            current = latest_by_job.get(configured.name, {})
            plugin_status = (
                "active"
                if configured.name == record.current_job
                and record.state in {"running", "pause_requested"}
                else "configured"
            )
            plugins = [
                PluginActivity(path=f"{namespace}.{name}", status=plugin_status)
                for namespace, entries in configured.plugins.items()
                for name in entries
            ]
            jobs.append(
                ExecutionJobState(
                    name=configured.name,
                    engine=configured.get_engine_name(),
                    status=self._job_status(record, configured.name, current),
                    stage=current.get("stage"),
                    fraction=current.get("fraction"),
                    message=current.get("message", ""),
                    plugins=plugins,
                )
            )
        return ExecutionSummary(
            execution_id=record.execution_id,
            project_id=self.scheduler.project.project_id,
            workflow_id=record.workflow.id,
            session_id=record.session_id,
            state=record.state,
            submitted_at=record.submitted_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            position=position,
            jobs=jobs,
            current_job=record.current_job,
            current_stage=record.current_stage,
            latest_message=record.latest_message,
            error=record.error,
        )

    @staticmethod
    def _job_status(record: _ExecutionRecord, name: str, event: dict[str, Any]) -> str:
        kind = event.get("kind")
        if kind == "job_completed":
            return "completed"
        if kind == "job_failed":
            return "failed"
        if kind == "job_skipped":
            return (
                "cancelled"
                if "cancel" in event.get("message", "").lower()
                else "skipped"
            )
        if name == record.current_job and record.state in {
            "running",
            "pause_requested",
        }:
            return "running"
        return "pending"

    def _require(self, execution_id: str) -> _ExecutionRecord:
        try:
            return self._records[execution_id]
        except KeyError as exc:
            raise KeyError(f"unknown execution {execution_id!r}") from exc

    def _trim_records(self) -> None:
        completed = [
            key
            for key, value in self._records.items()
            if value.state in {"completed", "partial", "failed", "cancelled"}
        ]
        for key in completed[: max(0, len(completed) - self.retained_executions)]:
            self._records.pop(key, None)
