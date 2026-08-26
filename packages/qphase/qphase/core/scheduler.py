"""qphase: Job Scheduler
---------------------------------------------------------
Orchestrates Workflow Jobs from dependency resolution through Artifact persistence.
The Scheduler handles serial logical-Job execution, passes scans to resource
engines, manages Session/Job directories, aggregates progress events, and builds
structured error reports.

Public API
----------
`Scheduler` : Main class for job execution and lifecycle management.
`JobResult` : Dataclass containing job execution results and metadata.
`execute_workflow` : Convenience function to execute a WorkflowSpec.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, cast

from .compiler import CompiledJob, CompiledWorkflow, WorkflowCompiler
from .config import JobConfig, WorkflowSpec
from .errors import (
    QPhaseConfigError,
    attach_session_log,
    get_logger,
)
from .execution import CancellationController, ExecutionContext
from .job_runner import (
    JobResult,
    _JobOutcome,
    configured_plugin_paths,
    fail_job,
    run_job,
    scan_summary,
)
from .persistence import ProjectStateStore
from .progress import ProgressSnapshot
from .project import ProjectContext
from .protocols import ResultProtocol
from .registry import RegistryCenter, registry
from .result_router import ResultRouter
from .system_config import SystemConfig, load_system_config
from .utils import save_yaml

log = get_logger()


class SessionManifest(TypedDict):
    """Type definition for session manifest."""

    schema: str
    session_id: str
    project_id: str
    workflow_id: str
    workflow_hash: str
    start_time: str
    status: str
    jobs: dict[str, dict[str, Any]]


class Scheduler:
    """Scheduler for executing simulation jobs.

    Manages serial job execution with dependency resolution, parameter scanning,
    configuration merging, structured progress aggregation, and structured
    error reporting.

    Parameters
    ----------
    system_config : SystemConfig | None, optional
        System configuration. If None, loads from system.yaml.
    on_progress : Callable[[ProgressSnapshot], None] | None, optional
        Callback for progress snapshots during job execution.
    on_job_dir : Callable[[Path], None] | None, optional
        Callback invoked with the Session Job directory after each Job completes.

    """

    system_config: SystemConfig
    session_id: str | None
    session_dir: Path | None
    manifest: SessionManifest | None

    def __init__(
        self,
        system_config: SystemConfig | None = None,
        project: ProjectContext | None = None,
        on_progress: Callable[[ProgressSnapshot], None] | None = None,
        on_job_dir: Callable[[Path], None] | None = None,
        cancellation: CancellationController | None = None,
        before_job: Callable[[JobConfig, int, int], JobConfig] | None = None,
        state_store: ProjectStateStore | None = None,
        registry_center: RegistryCenter | None = None,
    ):
        if system_config is None:
            self.system_config = load_system_config()
        else:
            self.system_config = system_config

        self.project = project or ProjectContext.discover()
        self.state_store = state_store or ProjectStateStore(self.project)

        self.on_progress = on_progress
        self.on_job_dir = on_job_dir
        self.cancellation = cancellation or CancellationController()
        self.before_job = before_job
        self._registry = registry_center or registry.snapshot()
        self._result_router = ResultRouter(self.system_config)
        self.session_id = None
        self.session_dir = None
        self.manifest = None
        self._session_log_path: Path | None = None
        self._session_log_handler: Any | None = None
        self._job_statuses: dict[str, str] = {}
        self._compiled_workflow: CompiledWorkflow | None = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def _initialize_session(self, workflow: WorkflowSpec) -> None:
        """Initialize a new execution session."""
        # Generate session ID
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        short_uuid = uuid.uuid4().hex[:6]
        self.session_id = f"{ts}_{short_uuid}"

        # Create session directory
        output_root = self.project.session_root
        now = datetime.now()
        self.session_dir = output_root / f"{now:%Y}" / f"{now:%m}" / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        workflow_payload = workflow.model_dump(mode="json", by_alias=True)
        workflow_hash = self._workflow_hash(workflow_payload)
        save_yaml(workflow_payload, self.session_dir / "workflow_snapshot.yaml")

        # Attach the per-session log file (full DEBUG content). A failure here
        # surfaces one explicit warning and never blocks the run.
        self._attach_session_log()

        # Initialize manifest
        self.manifest = {
            "schema": "qphase.session/2",
            "session_id": self.session_id,
            "project_id": self.project.project_id,
            "workflow_id": workflow.id,
            "workflow_hash": workflow_hash,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "jobs": {},
        }
        self._save_manifest()
        self._start_session_heartbeat()
        log.debug(f"Initialized session {self.session_id} at {self.session_dir}")

    @staticmethod
    def _workflow_hash(payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _attach_session_log(self) -> None:
        """Attach the session log file handler per reporting config."""
        if self.session_dir is None:
            return
        try:
            logging_cfg = self.system_config.reporting.logging
        except AttributeError:
            return
        if not getattr(logging_cfg, "session_file", False):
            return
        self._session_log_path, self._session_log_handler = attach_session_log(
            self.session_dir,
            filename=str(getattr(logging_cfg, "filename", "qphase.log")),
            level=str(getattr(logging_cfg, "file_level", "DEBUG")),
            as_json=str(getattr(logging_cfg, "format", "text")) == "json",
        )

    def _detach_session_log(self) -> None:
        """Remove the session log handler so sessions do not accumulate."""
        if self._session_log_handler is not None:
            try:
                log.removeHandler(self._session_log_handler)
                self._session_log_handler.close()
            except Exception:
                pass
            self._session_log_handler = None

    def _save_manifest(self) -> None:
        """Save session manifest to disk."""
        if self.session_dir and self.manifest:
            self.state_store.save_session_manifest(self.session_dir, self.manifest)

    def _start_session_heartbeat(self) -> None:
        if self.session_dir is None:
            return
        self._heartbeat_stop.clear()

        def _heartbeat() -> None:
            while not self._heartbeat_stop.is_set():
                self._write_session_lock()
                self._heartbeat_stop.wait(10.0)

        self._heartbeat_thread = threading.Thread(
            target=_heartbeat,
            name=f"qphase-heartbeat-{self.session_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _write_session_lock(self) -> None:
        if self.session_dir is None:
            return
        path = self.session_dir / "session.lock"
        temporary = path.with_suffix(".tmp")
        payload = {
            "pid": os.getpid(),
            "session_id": self.session_id,
            "heartbeat": datetime.now().astimezone().isoformat(),
        }
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(path)
        except OSError as exc:
            log.warning("Failed to update session heartbeat: %s", exc)

    def _stop_session_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None
        if self.session_dir is not None:
            try:
                (self.session_dir / "session.lock").unlink(missing_ok=True)
            except OSError as exc:
                log.warning("Failed to remove session heartbeat: %s", exc)

    def _update_job_status(
        self, job_name: str, status: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Update job status in manifest."""
        if self.manifest:
            if job_name not in self.manifest["jobs"]:
                self.manifest["jobs"][job_name] = {}

            self.manifest["jobs"][job_name]["status"] = status
            if metadata:
                self.manifest["jobs"][job_name].update(metadata)
            self._save_manifest()

    def run(
        self,
        workflow: WorkflowSpec,
        dry_run: bool = False,
        resume_from: Path | None = None,
        compiled_workflow: CompiledWorkflow | None = None,
    ) -> list[JobResult]:
        """Execute all jobs in the workflow serially.

        Parameters
        ----------
        workflow : WorkflowSpec
            List of jobs to execute
        dry_run : bool, optional
            If True, simulate execution without running engines.
        resume_from : Path | None, optional
            Path to a previous session directory to resume from.
        compiled_workflow : CompiledWorkflow | None, optional
            Previously resolved execution request. When provided, the scheduler
            does not re-read project defaults or recompile the workflow.

        Returns
        -------
        list[JobResult]
            Results for each executed job, in order

        """
        if compiled_workflow is not None:
            if compiled_workflow.project_id != self.project.project_id:
                raise QPhaseConfigError(
                    "compiled workflow belongs to a different project"
                )
            if compiled_workflow.workflow.id != workflow.id:
                raise QPhaseConfigError(
                    "compiled workflow does not match the requested workflow"
                )

        if dry_run:
            self.session_id = None
            self.session_dir = None
            self.manifest = None
            compiled = compiled_workflow or self._validate_jobs(workflow)
            self._compiled_workflow = compiled
            dry_results: list[JobResult] = []
            dry_job_results: dict[str, ResultProtocol] = {}
            logical_jobs = [
                compiled.job(name).job for name in compiled.topological_order
            ]
            effective_jobs = list(logical_jobs)
            for job_idx, original in enumerate(logical_jobs):
                job = (
                    self.before_job(original, job_idx, len(logical_jobs))
                    if self.before_job is not None
                    else original
                )
                if job != original:
                    effective_jobs[job_idx] = job
                    compiled = self._compile_revised_jobs(workflow, effective_jobs)
                    self._compiled_workflow = compiled
                self._run_single(
                    job,
                    job_idx,
                    len(logical_jobs),
                    effective_jobs,
                    dry_job_results,
                    dry_results,
                    compiled_job=compiled.job(job.name),
                    dry_run=True,
                )
            return dry_results

        # Compile before creating a Session or starting a worker.
        compiled = compiled_workflow or self._validate_jobs(workflow)
        if compiled.project_id != self.project.project_id:
            raise QPhaseConfigError(
                "compiled workflow belongs to a different project"
            )
        if compiled.workflow.id != workflow.id:
            raise QPhaseConfigError(
                "compiled workflow does not match the requested workflow"
            )
        self._compiled_workflow = compiled

        # Step 0: Initialize Session
        if resume_from:
            self._resume_session(resume_from, workflow)
        else:
            self._initialize_session(workflow)

        # Seed per-Session Job statuses from the manifest so that Jobs depending
        # on a previously failed upstream are marked skipped_dependency.
        self._job_statuses = {}
        if self.manifest:
            for name, entry in self.manifest["jobs"].items():
                status = entry.get("status")
                if status in ("failed", "skipped_dependency"):
                    self._job_statuses[name] = status

        results: list[JobResult] = []
        job_results: dict[str, ResultProtocol] = {}
        logical_jobs = [compiled.job(name).job for name in compiled.topological_order]
        effective_jobs = list(logical_jobs)
        try:
            for job_idx, job in enumerate(effective_jobs):
                if self.cancellation.execution.cancelled:
                    self._cancel_pending_jobs(logical_jobs[job_idx:], job_idx, results)
                    break
                if self.before_job is not None:
                    replacement = self.before_job(job, job_idx, len(logical_jobs))
                    if replacement.name != job.name:
                        raise QPhaseConfigError(
                            "a pending job revision must preserve the logical job name"
                        )
                    if replacement != job:
                        job = replacement
                        effective_jobs[job_idx] = replacement
                        compiled = self._compile_revised_jobs(workflow, effective_jobs)
                        self._compiled_workflow = compiled
                compiled_job = compiled.job(job.name)
                self._run_single(
                    job,
                    job_idx,
                    len(logical_jobs),
                    effective_jobs,
                    job_results,
                    results,
                    compiled_job=compiled_job,
                    dry_run=dry_run,
                )

            if self.manifest:
                failed = any(result.status == "failed" for result in results)
                skipped = any(
                    result.status == "skipped_dependency" for result in results
                )
                cancelled = any(result.status == "cancelled" for result in results)
                self.manifest["status"] = (
                    "failed"
                    if failed
                    else "cancelled"
                    if cancelled
                    else "partial"
                    if skipped
                    else "completed"
                )
                self._save_manifest()
            return results
        except Exception:
            if self.manifest:
                self.manifest["status"] = "failed"
                self._save_manifest()
            raise
        finally:
            self._stop_session_heartbeat()
            self._detach_session_log()

    def _resume_session(self, session_path: Path, workflow: WorkflowSpec) -> None:
        """Resume an existing session."""
        if not session_path.exists():
            raise QPhaseConfigError(f"Session directory not found: {session_path}")

        manifest_path = session_path / "session_manifest.json"
        if not manifest_path.exists():
            raise QPhaseConfigError(f"Session manifest not found in: {session_path}")

        self.manifest = cast(
            SessionManifest,
            self.state_store.load_session_manifest(session_path),
        )

        assert self.manifest is not None
        if self.manifest.get("project_id") != self.project.project_id:
            raise QPhaseConfigError(
                "Session belongs to a different project and cannot be resumed"
            )
        if self.manifest.get("workflow_id") != workflow.id:
            raise QPhaseConfigError(
                "Session workflow does not match the requested workflow"
            )
        expected_hash = self._workflow_hash(
            workflow.model_dump(mode="json", by_alias=True)
        )
        if self.manifest.get("workflow_hash") != expected_hash:
            raise QPhaseConfigError(
                "Session workflow content has changed and cannot be resumed"
            )
        self.session_id = self.manifest["session_id"]
        self.session_dir = session_path
        self._attach_session_log()
        self._start_session_heartbeat()
        log.info(f"Resuming session {self.session_id} from {self.session_dir}")

    def _handle_job_output(
        self,
        job: JobConfig,
        output_result: ResultProtocol,
        job_results: dict[str, ResultProtocol],
        job_dir: Path,
        context: ExecutionContext | None = None,
    ) -> None:
        """Handle job output based on job configuration.

        This method determines whether to:
        1. Store result for downstream jobs (if output references another job)
        2. Save result to disk (if auto_save_results is True)
        3. Both (if explicitly configured)

        If output is not specified, auto-saves using job name as filename
        (no extension).

        Parameters
        ----------
        job : JobConfig
            Job configuration
        output_result : ResultProtocol
            Result object from the job
        job_results : dict[str, ResultProtocol]
            Storage for job results that will be passed to downstream jobs
        job_dir : Path
            Session directory for this Job and its Artifacts.
        context : ExecutionContext | None
            Runtime artifact and checkpoint services for this logical job.

        Raises
        ------
        QPhaseConfigError
            If output references a non-existent downstream job

        """
        self._result_router.route_output(
            job,
            output_result,
            job_results,
            job_dir,
            context=context,
        )

    def _resolve_input(
        self,
        job: JobConfig,
        job_results: dict[str, ResultProtocol],
        source_override: str | None = None,
    ) -> ResultProtocol | None:
        """Resolve input for a job.

        Parameters
        ----------
        job : JobConfig
            Job configuration
        job_results : dict[str, ResultProtocol]
            Previously executed job results
        source_override : str | None, optional
            Compiler-normalized source name, when one is available.

        Returns
        -------
        ResultProtocol | None
            Input result object or None if no input

        """
        return self._result_router.resolve_input(
            job,
            job_results,
            source=source_override,
            manifest=self.manifest,
            session_dir=self.session_dir,
        )

    def _run_single(
        self,
        job: JobConfig,
        job_idx: int,
        job_total: int,
        logical_jobs: list[JobConfig],
        job_results: dict[str, ResultProtocol],
        results: list[JobResult],
        *,
        compiled_job: CompiledJob,
        dry_run: bool,
    ) -> None:
        """Execute one logical job and update the shared result state."""
        token = self.cancellation.token_for(job.name)
        if token.cancelled:
            self._record_cancelled_job(job, job_idx, job_total, results)
            return
        # Check if job is already completed (Resume Mode)
        if self.manifest and job.name in self.manifest["jobs"]:
            job_status = self.manifest["jobs"][job.name].get("status")
            if job_status == "completed":
                log.info(f"Skipping completed job: {job.name}")
                self._emit_snapshot(
                    ProgressSnapshot(
                        kind="job_skipped",
                        job_name=job.name,
                        job_index=job_idx,
                        total_jobs=job_total,
                        engine=job.get_engine_name(),
                        message="already completed (resume)",
                    )
                )
                return

        if dry_run:
            log.info(f"[DRY-RUN] Would execute job: {job.name}")
            log.info(f"          Engine: {job.get_engine_name()}")
            log.info(f"          Input: {job.input}")
            results.append(
                JobResult(
                    job_index=job_idx,
                    job_name=job.name,
                    job_dir=Path("dry_run"),
                    success=True,
                )
            )

            class MockResult:
                data = None
                metadata: dict[str, Any] = {}
                label: Any = None

                def save(self, path):
                    pass

            job_results[job.name] = MockResult()
            if job.output:
                job_results[job.output] = MockResult()
            return

        if not dry_run:
            self._update_job_status(job.name, "running")

        engine_name = job.get_engine_name()
        self._emit_snapshot(
            ProgressSnapshot(
                kind="job_started",
                job_name=job.name,
                job_index=job_idx,
                total_jobs=job_total,
                engine=engine_name,
                message="Starting job...",
                scan_summary=scan_summary(job),
                metadata={"plugins": configured_plugin_paths(job)},
            )
        )

        # Skip jobs whose upstream failed or was skipped earlier in this run.
        # Independent downstream jobs keep running (existing scheduler policy).
        source = compiled_job.input_source
        dependencies = list(compiled_job.depends_on)
        if source is not None and source not in dependencies:
            dependencies.append(source)
        blocked_by = [
            dependency
            for dependency in dependencies
            if self._job_statuses.get(dependency)
            in {"failed", "skipped_dependency"}
        ]
        if blocked_by:
            note = f"skipped: failed upstream dependencies {blocked_by}"
            log.info(f"Skipping job '{job.name}': {note}")
            assert self.session_dir is not None
            result = JobResult(
                job_index=job_idx,
                job_name=job.name,
                job_dir=self.session_dir / job.name,
                success=False,
                status="skipped_dependency",
                error_summary=note,
            )
            results.append(result)
            self._job_statuses[job.name] = "skipped_dependency"
            self._update_job_status(job.name, "skipped_dependency", {"note": note})
            self._emit_snapshot(
                ProgressSnapshot(
                    kind="job_skipped",
                    job_name=job.name,
                    job_index=job_idx,
                    total_jobs=job_total,
                    engine=engine_name,
                    message=note,
                )
            )
            return

        # Resolve input (input/config boundary; the engine never starts).
        try:
            input_result = self._resolve_input(job, job_results, source)
        except Exception as e:
            result = fail_job(self, job, job_idx, job_total, e, job_dir=None)
            results.append(result)
            self._job_statuses[job.name] = "failed"
            self._record_failure(result)
            return

        # Normal execution (engine/plugin boundary inside _run_job).
        raw_outcome = self._run_job(
            job,
            job_idx,
            job_total,
            input_result,
            compiled_job=compiled_job,
            display_total=len(logical_jobs),
        )
        # Keep one compatibility cycle for tests/extensions that patched the
        # former private three-tuple return contract.
        outcome = (
            _JobOutcome(*raw_outcome) if isinstance(raw_outcome, tuple) else raw_outcome
        )
        results.append(outcome.result)
        self._job_statuses[job.name] = outcome.result.status

        if not outcome.result.success:
            if outcome.result.status == "cancelled":
                self._update_job_status(job.name, "cancelled")
            else:
                self._record_failure(outcome.result)
            return

        assert outcome.output is not None and outcome.context is not None
        try:
            self._handle_job_output(
                job,
                outcome.output,
                job_results,
                outcome.result.job_dir,
                context=outcome.context,
            )
            outcome.context.checkpoints.complete()
        except Exception as e:
            failed = fail_job(
                self,
                job,
                job_idx,
                job_total,
                e,
                job_dir=outcome.result.job_dir,
            )
            results[-1] = failed
            self._job_statuses[job.name] = "failed"
            self._record_failure(failed)
            return

        assert self.session_dir is not None
        self._update_job_status(
            job.name,
            "completed",
            {
                "output_dir": str(outcome.result.job_dir.relative_to(self.session_dir)),
            },
        )

    def _record_failure(self, result: JobResult) -> None:
        """Write the failed manifest entry referencing the error report."""
        self._update_job_status(
            result.job_name,
            "failed",
            {
                "error_id": result.error_id,
                "error_code": result.error_code,
                "error": result.error_summary,
                "error_report": result.error_report_path,
            },
        )

    def _record_cancelled_job(
        self,
        job: JobConfig,
        job_idx: int,
        job_total: int,
        results: list[JobResult],
    ) -> None:
        job_dir = self.session_dir / job.name if self.session_dir else Path(".")
        results.append(
            JobResult(
                job_index=job_idx,
                job_name=job.name,
                job_dir=job_dir,
                success=False,
                status="cancelled",
                error_summary="cancelled before execution",
            )
        )
        self._job_statuses[job.name] = "cancelled"
        self._update_job_status(job.name, "cancelled")
        self._emit_snapshot(
            ProgressSnapshot(
                kind="job_skipped",
                job_name=job.name,
                job_index=job_idx,
                total_jobs=job_total,
                engine=job.get_engine_name(),
                message="Cancelled before execution",
            )
        )

    def _cancel_pending_jobs(
        self,
        jobs: list[JobConfig],
        start_index: int,
        results: list[JobResult],
    ) -> None:
        total = start_index + len(jobs)
        for offset, job in enumerate(jobs):
            self._record_cancelled_job(job, start_index + offset, total, results)

    def _emit_snapshot(self, snapshot: ProgressSnapshot) -> None:
        """Deliver a progress snapshot to the registered consumer."""
        if self.on_progress is not None:
            self.on_progress(snapshot)

    def _run_job(
        self,
        job: JobConfig,
        job_idx: int,
        job_total: int,
        input_result: ResultProtocol | None,
        *,
        compiled_job: CompiledJob,
        display_total: int | None = None,
    ) -> _JobOutcome:
        """Compatibility entry point for the module-level Job boundary."""
        return run_job(
            self,
            job,
            job_idx,
            job_total,
            input_result,
            compiled_job=compiled_job,
            display_total=display_total,
        )

    def _validate_jobs(self, workflow: WorkflowSpec) -> CompiledWorkflow:
        """Compile and validate a workflow before execution begins."""
        compiled = WorkflowCompiler(
            project=self.project,
            system_config=self.system_config,
            registry_view=self._registry.view(),
        ).compile(workflow)
        self._compiled_workflow = compiled
        return compiled

    def _compile_revised_jobs(
        self, workflow: WorkflowSpec, jobs: list[JobConfig]
    ) -> CompiledWorkflow:
        """Recompile a pending-job revision before its execution boundary."""
        revised = workflow.model_copy(update={"jobs": list(jobs)})
        return WorkflowCompiler(
            project=self.project,
            system_config=self.system_config,
            registry_view=self._registry.view(),
        ).compile(revised)

def execute_workflow(
    workflow: WorkflowSpec,
    *,
    project: ProjectContext | None = None,
    on_progress: Callable[[ProgressSnapshot], None] | None = None,
    on_job_dir: Callable[[Path], None] | None = None,
) -> list[JobResult]:
    """Execute one Workflow.

    Creates a Scheduler instance and runs all jobs in the provided workflow.

    Parameters
    ----------
    workflow : WorkflowSpec
        Workflow to execute
    project : ProjectContext | None, optional
        Explicit project boundary; discovered when omitted.
    on_progress : Callable[[ProgressSnapshot], None] | None, optional
        Progress callback function
    on_job_dir : Callable[[Path], None] | None, optional
        Callback invoked with the Session Job directory after completion.

    Returns
    -------
    list[JobResult]
        Results for each executed job

    """
    scheduler = Scheduler(
        project=project,
        on_progress=on_progress,
        on_job_dir=on_job_dir,
    )
    return scheduler.run(workflow)
