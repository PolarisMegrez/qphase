"""Local workstation execution queue and durable progress journal."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from qphase.core.compiler import CompiledWorkflow
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.execution import CancellationController
from qphase.core.persistence import ProjectStateStore
from qphase.core.progress import ProgressSnapshot
from qphase.core.scheduler import Scheduler
from qphase.core.tags import (
    execution_tag_assignment_id,
    freeze_tag_rules,
    load_tag_policy,
    validate_declared_tags,
)

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


def initial_execution_payload(
    *,
    execution_id: str,
    workflow: WorkflowSpec,
    source_workflow: str,
    submitted_at: datetime,
    compiled_workflow: CompiledWorkflow | None,
    submission_tags: list[str],
    tag_policy_revision: str | None,
    submission_tag_rules: dict[str, dict[str, Any]],
    state: str = "queued",
) -> dict[str, Any]:
    """Build the initial ``qphase.execution/1`` record payload.

    This is the single construction point shared by the queued
    (:class:`ExecutionManager`) and the synchronous
    (:meth:`SchedulerService.run`) entry paths, so both persist the same
    record shape. Submission tag assignment ids derive deterministically
    from the execution id, which lets the session manifest and the catalog
    read model cite the same ids without reading this record back.
    """
    return {
        "schema": "qphase.execution/1",
        "execution_id": execution_id,
        "source_workflow": source_workflow,
        "submission_tags": list(submission_tags),
        "submission_tag_assignments": {
            tag: execution_tag_assignment_id(execution_id, tag)
            for tag in submission_tags
        },
        "tag_policy_revision": tag_policy_revision,
        "submission_tag_rules": dict(submission_tag_rules),
        "workflow": workflow.model_dump(mode="json", by_alias=True),
        "compiled_workflow": compiled_workflow.to_payload()
        if compiled_workflow is not None
        else None,
        "submitted_at": submitted_at.isoformat(),
        "state": state,
        "session_id": None,
        "session_dir": None,
        "started_at": None,
        "finished_at": None,
        "current_job": None,
        "current_stage": None,
        "latest_message": "",
        "error": None,
        "pause_requested": False,
        "started_jobs": [],
        "revisions": {},
    }


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
    compiled_workflow: CompiledWorkflow | None = None
    last_persisted_progress: float = 0.0
    persisted_sequence: int = 0
    submission_tags: list[str] = field(default_factory=list)
    tag_policy_revision: str | None = None
    submission_tag_rules: dict[str, dict[str, Any]] = field(default_factory=dict)


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
        self.state_store: ProjectStateStore | None = getattr(
            scheduler, "state_store", None
        )
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._closed = False
        self._restore_persisted()
        if self._records:
            # Records restored from disk were persisted without a catalog
            # trigger; reindex once so they are visible immediately.
            self._reindex_catalog()
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
        self,
        workflow_reference: str,
        *,
        resume_from: str | None = None,
        tags: list[str] | None = None,
    ) -> ExecutionSummary:
        del resume_from  # Resume support remains on the synchronous service for now.
        workflow = self.scheduler.load_workflow(workflow_reference)
        submission_tags, tag_policy_revision, submission_tag_rules = (
            self._validate_submission_tags(tags)
        )
        compiled_workflow = None
        if self.state_store is not None:
            compiled_workflow = self.scheduler.compile_workflow(workflow)
        else:
            self._validate_plan(workflow)
        with self._wake:
            if len(self._queue) >= self.queue_capacity:
                raise RuntimeError("execution queue is full")
            execution_id = uuid.uuid4().hex[:12]
            record = _ExecutionRecord(
                execution_id=execution_id,
                workflow=workflow,
                source_workflow=workflow_reference,
                compiled_workflow=compiled_workflow,
                submission_tags=submission_tags,
                tag_policy_revision=tag_policy_revision,
                submission_tag_rules=submission_tag_rules,
            )
            self._records[execution_id] = record
            self._queue.append(execution_id)
            self._append_event(record, {"kind": "execution_queued"}, persist=False)
            try:
                self._save_execution(record)
                self._trim_records()
            except Exception:
                self._queue.remove(execution_id)
                self._records.pop(execution_id, None)
                raise
            self._reindex_catalog()
            self._wake.notify()
            return self._summary(record)

    def _validate_submission_tags(
        self, tags: list[str] | None
    ) -> tuple[list[str], str | None, dict[str, dict[str, Any]]]:
        """Policy-validate submission tags at the service boundary.

        Returns the canonical tags, the revision of the policy that
        validated them, and the frozen minimal namespace rules governing
        their effective-tag resolution, so the frozen provenance travels
        with the record and survives later policy edits.
        """
        policy = load_tag_policy(self.scheduler.project)
        validated = validate_declared_tags(list(tags or []), "execution", policy)
        rules = freeze_tag_rules(policy, validated)
        return validated, (policy.revision if policy is not None else None), rules

    def update_submission_tags(
        self, execution_id: str, tags: list[str]
    ) -> ExecutionSummary:
        """Replace submission tags while the execution is still queued."""
        with self._wake:
            record = self._require(execution_id)
            if record.state != "queued":
                raise ValueError("submission tags can only be updated while queued")
            (
                record.submission_tags,
                record.tag_policy_revision,
                record.submission_tag_rules,
            ) = self._validate_submission_tags(tags)
            self._save_execution(record)
            self._append_event(
                record,
                {
                    "kind": "submission_tags_updated",
                    "tags": list(record.submission_tags),
                },
                persist=False,
            )
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
            self._save_execution(record)
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
            self._save_execution(record)
            self._append_event(record, {"kind": "pause_requested"})
            return self._summary(record)

    def resume(self, execution_id: str) -> ExecutionSummary:
        record = self._require(execution_id)
        with record.gate:
            if record.state != "paused":
                raise ValueError("execution is not paused at a job boundary")
            record.pause_requested = False
            record.state = "running" if record.started_at else "queued"
            self._save_execution(record)
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
        revised_workflow = record.workflow.model_copy(update={"jobs": jobs})
        compiled_workflow = None
        if self.state_store is not None:
            compiled_workflow = self.scheduler.compile_workflow(revised_workflow)
        else:
            self._validate_plan(revised_workflow)
        record.revisions[job_name] = replacement
        record.compiled_workflow = compiled_workflow
        self._save_execution(record)
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
        self._save_execution(record)
        self._append_event(record, {"kind": "execution_started"}, persist=False)
        if record.submission_tags:
            self._append_event(
                record,
                {
                    "kind": "submission_tags_frozen",
                    "tags": list(record.submission_tags),
                },
                persist=False,
            )

        def _scheduler_ready(scheduler: Scheduler) -> None:
            record.scheduler = scheduler

        def _before_job(job: JobConfig, index: int, total: int) -> JobConfig:
            del index, total
            with record.gate:
                if record.pause_requested and not record.controller.execution.cancelled:
                    record.state = "paused"
                    self._save_execution(record)
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
                        self._save_execution(record)
                record.started_jobs.add(job.name)
                self._save_execution(record)
                return record.revisions.get(job.name, job)

        try:
            results = self.scheduler.run(
                record.compiled_workflow.workflow
                if record.compiled_workflow is not None
                else record.workflow,
                progress_callback=lambda snapshot: self._on_progress(record, snapshot),
                cancellation=record.controller,
                before_job=_before_job,
                on_scheduler=_scheduler_ready,
                compiled_workflow=record.compiled_workflow,
                submission_tags=list(record.submission_tags),
                submission_tag_policy_revision=record.tag_policy_revision,
                submission_tag_rules=record.submission_tag_rules,
                execution_id=record.execution_id,
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
            self._save_execution(record)
            # A reindex failure must not kill the worker thread; the read
            # model recovers via its fingerprint probe on the next query.
            try:
                self._reindex_catalog()
            except Exception:
                log.exception("catalog reindex after execution failed")

    def _on_progress(
        self, record: _ExecutionRecord, snapshot: ProgressSnapshot
    ) -> None:
        record.current_job = snapshot.job_name
        record.current_stage = snapshot.stage
        record.latest_message = snapshot.message
        if record.scheduler is not None:
            previous_session_id = record.session_id
            record.session_id = record.scheduler.session_id
            if record.session_id != previous_session_id:
                self._save_execution(record)
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
        record.scheduler.state_store.append_events(
            record.scheduler.session_dir,
            (event.model_dump(mode="json") for event in pending),
        )
        record.persisted_sequence = pending[-1].sequence

    def _validate_plan(self, workflow: WorkflowSpec) -> None:
        plan = self.scheduler.build_plan(workflow)
        if plan.validation_issues:
            details = "; ".join(
                f"{issue.path}: {issue.message}" for issue in plan.validation_issues
            )
            raise ValueError(f"execution plan is invalid: {details}")

    def _save_execution(self, record: _ExecutionRecord) -> None:
        if self.state_store is not None:
            self.state_store.save_execution(self._execution_payload(record))

    def _reindex_catalog(self) -> None:
        """Refresh the object catalog after execution records are persisted."""
        if self.state_store is None:
            return
        # Late import: the catalog is a read model layered above the service.
        from qphase.core.catalog import ProjectObjectCatalog

        ProjectObjectCatalog(self.scheduler.project).reindex()

    def _execution_payload(self, record: _ExecutionRecord) -> dict[str, Any]:
        session_dir: str | None = None
        if record.scheduler is not None and record.scheduler.session_dir is not None:
            session_dir = (
                Path(record.scheduler.session_dir)
                .resolve()
                .relative_to(self.scheduler.project.session_root.resolve())
                .as_posix()
            )
        payload = initial_execution_payload(
            execution_id=record.execution_id,
            workflow=record.workflow,
            source_workflow=record.source_workflow,
            submitted_at=record.submitted_at,
            compiled_workflow=record.compiled_workflow,
            submission_tags=record.submission_tags,
            tag_policy_revision=record.tag_policy_revision,
            submission_tag_rules=record.submission_tag_rules,
            state=record.state,
        )
        payload.update(
            session_id=record.session_id,
            session_dir=session_dir,
            started_at=record.started_at.isoformat()
            if record.started_at is not None
            else None,
            finished_at=record.finished_at.isoformat()
            if record.finished_at is not None
            else None,
            current_job=record.current_job,
            current_stage=record.current_stage,
            latest_message=record.latest_message,
            error=record.error,
            pause_requested=record.pause_requested,
            started_jobs=sorted(record.started_jobs),
            revisions={
                name: job.model_dump(mode="json", by_alias=True)
                for name, job in record.revisions.items()
            },
        )
        return payload

    def _restore_persisted(self) -> None:
        if self.state_store is None:
            return
        for payload in self.state_store.load_executions():
            record = self._record_from_payload(payload)
            if record.state in {"running", "pause_requested"} or (
                record.state == "paused" and record.started_at is not None
            ):
                record.state = "failed"
                record.error = "execution worker interrupted by process restart"
                record.finished_at = _now()
                self._save_execution(record)
            self._records[record.execution_id] = record
            if record.state == "queued":
                self._queue.append(record.execution_id)
        self._trim_records()

    def _record_from_payload(self, payload: dict[str, Any]) -> _ExecutionRecord:
        if payload.get("schema") != "qphase.execution/1":
            raise ValueError("unsupported execution record schema")
        workflow = WorkflowSpec.model_validate(payload["workflow"])
        compiled_payload = payload.get("compiled_workflow")
        compiled_workflow = (
            CompiledWorkflow.from_payload(compiled_payload)
            if compiled_payload is not None
            else None
        )
        if compiled_workflow is None:
            raise ValueError("execution record is missing compiled workflow")
        if compiled_workflow.project_id != self.scheduler.project.project_id:
            raise ValueError("execution record belongs to a different project")
        if compiled_workflow.workflow.id != workflow.id:
            raise ValueError("compiled workflow does not match execution workflow")
        state = payload["state"]
        allowed_states = {
            "queued",
            "running",
            "pause_requested",
            "paused",
            "completed",
            "partial",
            "failed",
            "cancelled",
        }
        if state not in allowed_states:
            raise ValueError(f"unsupported execution state: {state!r}")
        revisions = {
            name: JobConfig.model_validate(job)
            for name, job in dict(payload.get("revisions", {})).items()
        }
        record = _ExecutionRecord(
            execution_id=str(payload["execution_id"]),
            workflow=workflow,
            source_workflow=str(payload["source_workflow"]),
            submitted_at=datetime.fromisoformat(str(payload["submitted_at"])),
            state=state,
            session_id=payload.get("session_id"),
            started_at=self._parse_time(payload.get("started_at")),
            finished_at=self._parse_time(payload.get("finished_at")),
            current_job=payload.get("current_job"),
            current_stage=payload.get("current_stage"),
            latest_message=str(payload.get("latest_message", "")),
            error=payload.get("error"),
            pause_requested=bool(payload.get("pause_requested", False)),
            revisions=revisions,
            started_jobs=set(payload.get("started_jobs", [])),
            compiled_workflow=compiled_workflow,
            submission_tags=[str(item) for item in payload.get("submission_tags", [])],
            tag_policy_revision=payload.get("tag_policy_revision"),
            submission_tag_rules={
                str(tag): dict(rule)
                for tag, rule in dict(payload.get("submission_tag_rules") or {}).items()
            },
        )
        session_dir = payload.get("session_dir")
        if isinstance(session_dir, str):
            root = self.scheduler.project.session_root.resolve()
            candidate = (root / session_dir).resolve()
            if candidate.is_relative_to(root) and candidate.exists():
                assert self.state_store is not None
                events = self.state_store.read_events(candidate)
                for item in events:
                    event = ExecutionEvent.model_validate(item)
                    record.events.append(event)
                    record.sequence = max(record.sequence, event.sequence)
                record.persisted_sequence = record.sequence
        return record

    @staticmethod
    def _parse_time(value: Any) -> datetime | None:
        return datetime.fromisoformat(str(value)) if value is not None else None

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
            submission_tags=list(record.submission_tags),
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
            if self.state_store is not None:
                self.state_store.delete_execution(key)
