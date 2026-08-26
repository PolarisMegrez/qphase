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
import inspect
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from .artifacts import ArtifactStore
from .config import JobConfig, WorkflowSpec
from .config_loader import (
    get_config_for_job,
    merge_plugin_config_sections,
    registered_plugin_namespaces,
)
from .dataset import DatasetResultProtocol, MappedDatasetResult, iter_dataset_views
from .error_report import build_error_report, save_error_report
from .errors import (
    ErrorCode,
    QPhaseConfigError,
    QPhaseIOError,
    QPhasePluginError,
    QPhaseRuntimeError,
    attach_session_log,
    get_logger,
)
from .execution import (
    CancellationController,
    CheckpointStore,
    ExecutionContext,
    ProgressReporter,
    ResourceSnapshot,
    execution_fingerprint,
    plugin_fingerprint,
)
from .logging_context import bind_log_context, set_log_context
from .progress import ProgressEvent, ProgressSnapshot, ProgressTracker
from .project import ProjectContext
from .protocols import ResultProtocol
from .registry import registry
from .system_config import SystemConfig, load_system_config
from .utils import save_yaml

log = get_logger()


@dataclass
class JobResult:
    """Result of a single job execution."""

    job_index: int
    job_name: str
    job_dir: Path
    success: bool
    status: str = "completed"  # "completed" | "failed" | "skipped_dependency"
    error_summary: str | None = None
    error_id: str | None = None
    error_code: str | None = None
    error_report_path: str | None = None

    @property
    def error(self) -> str | None:
        """Backward-compatible alias for ``error_summary``."""
        return self.error_summary


@dataclass
class _JobOutcome:
    """Internal result of one engine invocation before artifact persistence."""

    result: JobResult
    output: ResultProtocol | None
    context: ExecutionContext | None


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
    ):
        if system_config is None:
            self.system_config = load_system_config()
        else:
            self.system_config = system_config

        self.project = project or ProjectContext.discover()

        self.on_progress = on_progress
        self.on_job_dir = on_job_dir
        self.cancellation = cancellation or CancellationController()
        self.before_job = before_job
        from .registry import registry

        self._registry = registry
        self.session_id = None
        self.session_dir = None
        self.manifest = None
        self._session_log_path: Path | None = None
        self._session_log_handler: Any | None = None
        self._job_statuses: dict[str, str] = {}
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
            manifest_path = self.session_dir / "session_manifest.json"
            try:
                temporary = manifest_path.with_suffix(".tmp")
                with open(temporary, "w", encoding="utf-8") as f:
                    json.dump(self.manifest, f, indent=2)
                temporary.replace(manifest_path)
            except Exception as e:
                log.warning(f"Failed to save session manifest: {e}")

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

        Returns
        -------
        list[JobResult]
            Results for each executed job, in order

        """
        if dry_run:
            self.session_id = None
            self.session_dir = None
            self.manifest = None
            self._validate_jobs(workflow)
            dry_results: list[JobResult] = []
            dry_job_results: dict[str, ResultProtocol] = {}
            for job_idx, original in enumerate(workflow.jobs):
                job = (
                    self.before_job(original, job_idx, len(workflow.jobs))
                    if self.before_job is not None
                    else original
                )
                self._run_single(
                    job,
                    job_idx,
                    len(workflow.jobs),
                    workflow.jobs,
                    dry_job_results,
                    dry_results,
                    dry_run=True,
                )
            return dry_results

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
        logical_jobs = workflow.jobs
        try:
            # Step 1: Validate jobs before execution
            self._validate_jobs(workflow)

            for job_idx, job in enumerate(logical_jobs):
                if self.cancellation.execution.cancelled:
                    self._cancel_pending_jobs(logical_jobs[job_idx:], job_idx, results)
                    break
                if self.before_job is not None:
                    replacement = self.before_job(job, job_idx, len(logical_jobs))
                    if replacement.name != job.name:
                        raise QPhaseConfigError(
                            "a pending job revision must preserve the logical job name"
                        )
                    job = replacement
                self._run_single(
                    job,
                    job_idx,
                    len(logical_jobs),
                    logical_jobs,
                    job_results,
                    results,
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

        try:
            with open(manifest_path, encoding="utf-8") as f:
                self.manifest = json.load(f)
        except Exception as e:
            raise QPhaseConfigError(f"Failed to load session manifest: {e}") from e

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
        # Determine the output destination (alias for downstream jobs)
        output_alias = job.output if job.output else job.name

        # Store result for downstream jobs
        # We store by job name so downstream jobs can reference it
        job_results[job.name] = output_result

        # If output is explicitly set, we might also want to store it under that name
        # (though usually output refers to filename or downstream job name)
        if job.output:
            job_results[job.output] = output_result

        # Determine if we should save to disk
        should_save = False
        save_filename = output_alias

        if job.save is not None:
            # Explicit control
            if isinstance(job.save, bool):
                should_save = job.save
            elif isinstance(job.save, str):
                should_save = True
                save_filename = job.save
        else:
            # Fallback to system default
            should_save = self.system_config.auto_save_results

        # Save to disk if enabled
        if should_save:
            # Build save path inside the Job's Session directory.
            # Note: filename should not include extension -
            # ResultProtocol.save() will add appropriate extension
            save_path = job_dir / save_filename

            try:
                if context is not None:
                    context.artifacts.save_result(output_result, save_filename)
                else:
                    output_result.save(save_path)
                log.debug(f"Job '{job.name}' result saved to {save_path}")
            except Exception as e:
                raise QPhaseIOError(
                    f"Failed to save job '{job.name}' output to '{save_path}': {e}",
                    code=ErrorCode.ARTIFACT_IO,
                    hint="Check disk space and write permissions for the Job "
                    "directory.",
                ) from e

    def _resolve_input(
        self, job: JobConfig, job_results: dict[str, ResultProtocol]
    ) -> ResultProtocol | None:
        """Resolve input for a job.

        Parameters
        ----------
        job : JobConfig
            Job configuration
        job_results : dict[str, ResultProtocol]
            Previously executed job results

        Returns
        -------
        ResultProtocol | None
            Input result object or None if no input

        """
        if job.input is None:
            return None
        source = job.input.from_

        if source in job_results:
            return job_results[source]

        # Check if input is in manifest (from a previous run in same session context)
        if self.manifest and source in self.manifest["jobs"]:
            job_entry = self.manifest["jobs"][source]
            if job_entry.get("status") == "completed" and self.session_dir:
                output_rel_path = job_entry.get("output_dir")
                if output_rel_path:
                    job_dir = self.session_dir / output_rel_path
                    try:
                        from .result_loader import load_result

                        log.info(f"Loading result for '{source}' from disk...")
                        result = load_result(source, job_dir)
                        # Cache it
                        job_results[source] = result
                        return result
                    except Exception as e:
                        log.warning(
                            f"Failed to load result for '{source}' from disk: {e}"
                        )

        # Check if input is an external directory or file
        input_path = Path(source)
        if input_path.exists():
            if input_path.is_dir():
                log.info(
                    f"Job '{job.name}' input '{source}' is a directory; "
                    "passing path to engine for resource-specific loading."
                )
                from .aggregation import DirectoryInputResult

                return DirectoryInputResult(
                    path=input_path,
                    meta={"input_kind": "directory", "path": str(input_path)},
                )
            # External file input is not supported without a loader mechanism
            # which has been removed.
            raise QPhaseConfigError(
                f"Job '{job.name}' specifies file input '{source}', "
                "but file loading is not currently supported.",
                code=ErrorCode.INPUT,
            )

        # Input not found
        raise QPhaseConfigError(
            f"Job '{job.name}' input '{source}' not found. "
            f"Expected a previous job name or a valid file path with input_loader.",
            code=ErrorCode.INPUT,
            hint="Run the upstream job first, or fix the 'input.from' reference.",
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
                scan_summary=self._scan_summary(job),
                metadata={"plugins": self._configured_plugin_paths(job)},
            )
        )

        # Skip jobs whose upstream failed or was skipped earlier in this run.
        # Independent downstream jobs keep running (existing scheduler policy).
        source = job.input.from_ if job.input else None
        upstream_status = self._job_statuses.get(source) if source else None
        if upstream_status in ("failed", "skipped_dependency"):
            note = (
                f"skipped: upstream job '{source}' {upstream_status.replace('_', ' ')}"
            )
            log.info(f"Skipping job '{job.name}': {note}")
            result = JobResult(
                job_index=job_idx,
                job_name=job.name,
                job_dir=(self.session_dir / job.name)
                if self.session_dir
                else Path("."),
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
            input_result = self._resolve_input(job, job_results)
        except Exception as e:
            result = self._fail_job(job, job_idx, job_total, e, job_dir=None)
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
                outcome.context,
            )
            outcome.context.checkpoints.complete()
        except Exception as e:
            failed = self._fail_job(
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

    @staticmethod
    def _scan_summary(job: JobConfig) -> dict[str, Any] | None:
        """Small scan descriptor for start events and error reports."""
        if job.scan is None:
            return None
        try:
            return job.scan.compile().summary()
        except Exception:
            return None

    @staticmethod
    def _configured_plugin_paths(job: JobConfig) -> list[str]:
        return [
            f"{namespace}.{name}"
            for namespace, entries in job.plugins.items()
            for name in entries
        ]

    def _fail_job(
        self,
        job: JobConfig,
        job_idx: int,
        job_total: int,
        exc: BaseException,
        *,
        job_dir: Path | None,
        stage: str | None = None,
        plugin: str | None = None,
    ) -> JobResult:
        """Build the structured error report for a failed job, exactly once.

        This is the single place that logs the exception traceback and writes
        ``error_report.json``; callers must not re-log the same failure.
        """
        report = build_error_report(
            exc,
            session_id=self.session_id,
            job_name=job.name,
            engine=job.get_engine_name(),
            stage=stage,
            plugin=plugin,
            job_dir=job_dir,
            scan_context=self._scan_summary(job),
            log_file=self._session_log_path,
        )
        if job_dir is not None:
            target_dir = job_dir
        elif self.session_dir is not None:
            target_dir = self.session_dir / job.name
        else:
            target_dir = Path(".")
        report_path = save_error_report(report, target_dir)
        log.exception(f"Job '{job.name}' failed [{report.code}]: {report.summary}")
        summary = report.summary_dto(
            report_path=str(report_path) if report_path is not None else None
        )
        self._emit_snapshot(
            ProgressSnapshot(
                kind="job_failed",
                job_name=job.name,
                job_index=job_idx,
                total_jobs=job_total,
                engine=job.get_engine_name(),
                stage=stage,
                job_dir=str(target_dir),
                error=summary,
                message=report.summary,
            )
        )
        return JobResult(
            job_index=job_idx,
            job_name=job.name,
            job_dir=target_dir,
            success=False,
            status="failed",
            error_summary=report.summary,
            error_id=report.error_id,
            error_code=report.code,
            error_report_path=(str(report_path) if report_path is not None else None),
        )

    def _get_merged_config_for_job(self, job: JobConfig) -> dict[str, Any]:
        """Merge global system config with job-specific overrides.

        Returns
        -------
        dict[str, Any]
            Merged configuration dictionary containing plugins, engine,
            params, and any top-level plugin sections defined in the job.

        """
        plugin_namespaces = registered_plugin_namespaces()

        # Merge global config with job config
        job_override: dict[str, Any] = {
            "plugins": job.plugins,
            "engine": job.engine,
            "params": job.params,
        }
        # Preserve top-level plugin sections (e.g. backend, analyser) that live in
        # JobConfig.model_extra so the merge/extraction logic sees them.
        job_extra = job.model_extra or {}
        for key in plugin_namespaces:
            if key in job_extra:
                job_override[key] = job_extra[key]

        return get_config_for_job(self.project, job_config_dict=job_override)

    def _run_job(
        self,
        job: JobConfig,
        job_idx: int,
        job_total: int,
        input_result: ResultProtocol | None,
        *,
        display_total: int | None = None,
    ) -> _JobOutcome:
        """Execute a single job and return its outcome.

        This method handles the complete job execution lifecycle:
        1. Create the Session Job directory
        2. Merge global config with job-specific config
        3. Build plugin instances from configuration
        4. Instantiate and run the engine
        5. Aggregate progress events and save snapshot

        Plugin and engine failures are converted into a failed ``_JobOutcome``
        with a structured error report; this method does not re-raise them.

        Parameters
        ----------
        job : JobConfig
            Job configuration to execute
        job_idx : int
            Index of this job in the execution group (0-based)
        job_total : int
            Total number of jobs in the execution group
        input_result : ResultProtocol | None
            Input data from upstream job, or None
        display_total : int | None, optional
            Number of jobs to display in progress reporting, so progress
            reflects the user's mental model.

        Returns
        -------
        _JobOutcome
            Job execution metadata, engine output (on success), and the
            execution context (on success).

        """
        display_total = job_total if display_total is None else display_total
        job_dir = self._create_job_dir(job)
        engine_name = job.get_engine_name()
        tracker = self._make_tracker()
        clock_start = time.monotonic()

        with bind_log_context(
            session_id=self.session_id, job=job.name, engine=engine_name
        ):
            try:
                return self._run_job_inner(
                    job,
                    job_idx,
                    display_total,
                    input_result,
                    job_dir=job_dir,
                    engine_name=engine_name,
                    tracker=tracker,
                    clock_start=clock_start,
                )
            except Exception as e:
                if getattr(e, "code", None) == ErrorCode.CANCELLATION:
                    self._emit_snapshot(
                        ProgressSnapshot(
                            kind="job_skipped",
                            job_name=job.name,
                            job_index=job_idx,
                            total_jobs=display_total,
                            engine=engine_name,
                            stage=tracker.current_stage,
                            job_dir=str(job_dir),
                            message="Cancelled",
                        )
                    )
                    return _JobOutcome(
                        result=JobResult(
                            job_index=job_idx,
                            job_name=job.name,
                            job_dir=job_dir,
                            success=False,
                            status="cancelled",
                            error_summary="cancelled by user",
                        ),
                        output=None,
                        context=None,
                    )
                result = self._fail_job(
                    job,
                    job_idx,
                    display_total,
                    e,
                    job_dir=job_dir,
                    stage=tracker.current_stage,
                )
                return _JobOutcome(result=result, output=None, context=None)

    def _run_job_inner(
        self,
        job: JobConfig,
        job_idx: int,
        display_total: int,
        input_result: ResultProtocol | None,
        *,
        job_dir: Path,
        engine_name: str,
        tracker: ProgressTracker,
        clock_start: float,
    ) -> _JobOutcome:
        """Job body executed under the error boundary of :meth:`_run_job`."""
        merged_config = self._get_merged_config_for_job(job)

        plugin_namespaces = registered_plugin_namespaces()
        plugins_cfg = merge_plugin_config_sections(merged_config)

        # Determine target Engine class to inspect Manifest
        # This helps us decide which plugins are actually needed
        engine_config_dict = merged_config.get("engine", {})

        target_engine_name = None
        if engine_name:
            target_engine_name = engine_name
        elif engine_config_dict:
            target_engine_name = list(engine_config_dict.keys())[0]

        # Inspect Engine Manifest to determine plugin requirements
        required_namespaces = set()
        optional_namespaces = set()

        if target_engine_name:
            try:
                engine_cls = registry.get_plugin_class("engine", target_engine_name)
                if hasattr(engine_cls, "manifest"):
                    manifest = engine_cls.manifest
                    # If the job consumes an upstream input and the engine declares
                    # input_plugins, use those instead of the normal required set.
                    if job.input and manifest.input_plugins:
                        required_namespaces.update(manifest.input_plugins)
                    elif manifest.required_plugins:
                        required_namespaces.update(manifest.required_plugins)
                    if manifest.optional_plugins:
                        optional_namespaces.update(manifest.optional_plugins)
            except Exception as e:
                log.debug(
                    f"Could not inspect manifest for engine '{target_engine_name}': {e}"
                )

        # Determine explicit namespaces defined in JobConfig.
        # This separates user overrides from merged defaults.
        job_extra = job.model_extra or {}
        explicit_namespaces = set(job.plugins.keys())
        for key in plugin_namespaces:
            if key in job_extra:
                explicit_namespaces.add(key)

        # Filter and configure plugins based on Job intent and Engine requirements
        final_plugins_cfg = {}

        for ns, ns_config in plugins_cfg.items():
            # 1. Explicitly configured in Job?
            # If yes, we strictly respect the Job's choice (filtering specific plugins).
            if ns in explicit_namespaces:
                allowed_plugins = set(job.plugins.get(ns, {}).keys())
                if ns in job_extra and isinstance(job_extra[ns], dict):
                    allowed_plugins.update(job_extra[ns].keys())

                final_plugins_cfg[ns] = {
                    k: v for k, v in ns_config.items() if k in allowed_plugins
                }

            # 2. Required by Engine but not in the job.
            # Fall back to global defaults for the namespace.
            elif ns in required_namespaces:
                final_plugins_cfg[ns] = ns_config

            # 3. Optional or Unknown?
            # Do NOT inherit Global defaults. This prevents side-effects from plugins
            # like 'analyser' or 'visualizer' running when not requested.
            else:
                pass

        # Build plugins (backend, integrator, state, etc.)
        plugins = self._build_plugins(final_plugins_cfg)

        # Extract engine name and config
        engine_config_dict = merged_config.get("engine", {})
        if engine_config_dict:
            # Prioritize the engine specified in the job config
            job_engine_name = job.get_engine_name()
            if job_engine_name and job_engine_name in engine_config_dict:
                engine_name = job_engine_name
            else:
                # Fallback (might be ambiguous if global config adds engines)
                engine_name = list(engine_config_dict.keys())[0]

            engine_config_raw = engine_config_dict[engine_name].copy()
            engine_config_raw["name"] = engine_name
        else:
            # Fallback to job's engine config
            engine_name = job.get_engine_name()
            engine_config_raw = job.engine.get(engine_name, {}).copy()
            engine_config_raw["name"] = engine_name

        # Inject the Job directory as output_dir for engines that support it.
        # (e.g. VizEngine). We cast to str because config expects str.
        # Engines that don't support this field should have extra="allow"
        # in their config schema.
        engine_config_raw["output_dir"] = str(job_dir)

        # Instantiate engine via registry
        try:
            engine = registry.create_plugin_instance(
                "engine", engine_config_raw, plugins=plugins
            )
        except Exception as e:
            raise QPhasePluginError(
                f"Failed to instantiate engine '{engine_name}': {e}",
                code=ErrorCode.PLUGIN_CREATION,
                hint="Check the engine configuration against its schema.",
                context={"engine": engine_name},
            ) from e

        # Also write snapshot
        self._write_snapshot(job_dir, job, merged_config, job_idx)

        # Structured progress plumbing: engines emit work events through the
        # reporter; the tracker aggregates them into snapshots. Legacy engines
        # keep working through the percent-signature adapter.
        on_progress = self.on_progress

        def _sink(event: ProgressEvent) -> None:
            observed = tracker.observe(event)
            if observed.stage:
                set_log_context(stage=observed.stage)
            if on_progress is None:
                return
            fraction, rate, remaining = tracker.estimates(observed)
            on_progress(
                ProgressSnapshot(
                    kind=(
                        "job_status" if observed.kind == "status" else "job_progress"
                    ),
                    job_name=job.name,
                    job_index=job_idx,
                    total_jobs=display_total,
                    engine=engine_name,
                    stage=observed.stage,
                    completed=observed.completed,
                    total=observed.total,
                    unit=observed.unit,
                    fraction=fraction,
                    elapsed=tracker.elapsed(observed),
                    rate=rate,
                    remaining=remaining,
                    message=observed.message,
                    importance=observed.importance,
                    metadata=observed.metadata,
                    monotonic_time=observed.monotonic_time,
                )
            )

        reporter = ProgressReporter(_sink)
        legacy_cb = reporter.legacy_callback()

        effective_system = job.merge_with_system_config(self.system_config)
        backend = plugins.get("backend")
        backend_name = None
        if backend is not None and hasattr(backend, "backend_name"):
            backend_name = str(backend.backend_name())
        backend_config = getattr(backend, "config", None)
        dtype = getattr(backend_config, "float_dtype", None)
        checkpoint_config = effective_system.scan_runtime.checkpoint
        if checkpoint_config.enabled:
            plugin_ids = {
                name: plugin_fingerprint(instance)
                for name, instance in plugins.items()
                if "." not in name
            }
            fingerprint = execution_fingerprint(
                job.model_dump(by_alias=True),
                plugins=plugin_ids,
                backend=backend_name,
                dtype=None if dtype is None else str(dtype),
            )
        else:
            fingerprint = {}
        context = ExecutionContext(
            parameter_grid=job.scan.compile() if job.scan is not None else None,
            resources=ResourceSnapshot.from_system_config(
                effective_system, backend=backend
            ),
            progress=reporter,
            cancellation=self.cancellation.token_for(job.name),
            artifacts=ArtifactStore(job_dir, effective_system.scan_runtime),
            checkpoints=CheckpointStore(
                job_dir,
                checkpoint_config,
                fingerprint,
            ),
            job_dir=job_dir,
            metadata={
                "job_name": job.name,
                "scan_summary": self._scan_summary(job),
                "configured_plugins": self._configured_plugin_paths(job),
            },
        )

        output_result: ResultProtocol
        if job.input is not None and job.input.mode == "map":
            if not isinstance(input_result, DatasetResultProtocol):
                raise QPhaseConfigError(
                    f"job {job.name!r} uses input.mode=map but its source is not "
                    "a dataset result",
                    code=ErrorCode.INPUT,
                )
            mapped: OrderedDict[str, ResultProtocol] = OrderedDict()
            views = list(
                iter_dataset_views(
                    input_result,
                    select=job.input.select,
                    group_by=tuple(job.input.group_by),
                )
            )
            total_views = len(views)
            for view_index, (label, view) in enumerate(views, start=1):
                reporter.update(
                    completed=view_index - 1,
                    total=total_views,
                    unit="view",
                    stage="map",
                    message=f"map view {view_index}/{total_views}: {label}",
                )

                def _map_child_sink(
                    event: ProgressEvent,
                    current_view_index: int = view_index,
                ) -> None:
                    detail = event.message or event.stage or "running"
                    reporter.status(
                        f"map view {current_view_index}/{total_views}: {detail}",
                        stage="map",
                        metadata={
                            "view_index": current_view_index - 1,
                            "view_total": total_views,
                            "child_stage": event.stage,
                            "child_completed": event.completed,
                            "child_total": event.total,
                            "child_unit": event.unit,
                        },
                    )

                child_reporter = ProgressReporter(_map_child_sink)
                context.progress = child_reporter
                try:
                    mapped[label] = self._invoke_engine(
                        engine,
                        view,
                        context,
                        child_reporter.legacy_callback(),
                    )
                finally:
                    context.progress = reporter
            reporter.update(
                completed=total_views,
                total=total_views,
                unit="view",
                stage="map",
                message="map views complete",
            )
            preserves_shape = not job.input.select and not job.input.group_by
            output_result = MappedDatasetResult(
                mapped,
                dict(input_result.axes)
                if preserves_shape
                else {"view": list(range(len(mapped)))},
                input_result.shape if preserves_shape else (len(mapped),),
                meta={"source": job.input.from_, "mode": "map"},
            )
        else:
            output_result = self._invoke_engine(
                engine, input_result, context, legacy_cb
            )

        # Ensure output is a ResultProtocol object
        if not isinstance(output_result, ResultProtocol):
            raise QPhaseRuntimeError(
                f"Engine '{engine_name}' did not return a "
                f"ResultProtocol object. "
                f"All engines must return a ResultProtocol instance from "
                f"their run() method."
            )

        # Report job completion
        duration = time.monotonic() - clock_start
        self._emit_snapshot(
            ProgressSnapshot(
                kind="job_completed",
                job_name=job.name,
                job_index=job_idx,
                total_jobs=display_total,
                engine=engine_name,
                message="Completed successfully",
                duration=duration,
                job_dir=str(job_dir),
            )
        )

        if self.on_job_dir is not None:
            self.on_job_dir(job_dir)

        return _JobOutcome(
            result=JobResult(
                job_index=job_idx,
                job_name=job.name,
                job_dir=job_dir,
                success=True,
                status="completed",
            ),
            output=output_result,
            context=context,
        )

    def _make_tracker(self) -> ProgressTracker:
        """Create a progress tracker parameterized by reporting config."""
        try:
            cfg = self.system_config.reporting.progress
        except AttributeError:
            cfg = None
        return ProgressTracker(
            eta_warmup_seconds=self._cfg_number(cfg, "eta_warmup_seconds", 2.0),
            eta_min_samples=int(self._cfg_number(cfg, "eta_min_samples", 3)),
            eta_smoothing=self._cfg_number(cfg, "eta_smoothing", 0.25),
        )

    @staticmethod
    def _cfg_number(cfg: Any, name: str, default: float) -> float:
        """Read a numeric config value defensively (tolerates test doubles)."""
        try:
            return float(getattr(cfg, name))
        except (TypeError, ValueError, AttributeError):
            return default

    @staticmethod
    def _invoke_engine(
        engine: Any,
        data: Any,
        context: ExecutionContext,
        progress_cb: Any | None,
    ) -> ResultProtocol:
        """Invoke an engine without masking TypeError raised inside the engine."""
        signature = inspect.signature(engine.run)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        kwargs: dict[str, Any] = {"data": data}
        if accepts_kwargs or "context" in signature.parameters:
            kwargs["context"] = context
        if accepts_kwargs or "progress_cb" in signature.parameters:
            kwargs["progress_cb"] = progress_cb
        return engine.run(**kwargs)

    def _validate_jobs(self, workflow: WorkflowSpec) -> None:
        """Validate job configurations and data flow.

        Performs two-stage validation:
        1. Check that each job has exactly one engine
        2. Validate input/output data flow

        Raises
        ------
        QPhaseConfigError
            If validation fails

        """
        log.debug("Validating job configurations...")

        # Stage 1: Check each job has exactly one engine
        self._validate_single_engine_per_job(workflow)

        # Stage 2: Validate engine dependencies
        for job in workflow.jobs:
            self._validate_job_dependencies(job)

        # Stage 3: Validate data flow
        self._validate_data_flow(workflow)

        log.debug("Job validation completed successfully")

    def _validate_job_dependencies(self, job: JobConfig) -> None:
        """Validate that the job provides all plugins required by its engine.

        The set of "provided" plugins is computed consistently with
        :meth:`_run_job`: it includes explicit ``plugins:`` entries, top-level
        plugin sections (``backend``, ``integrator``, ``model``, ``analyser``,
        etc.), and plugin namespaces provided by the merged global configuration.
        """
        engine_name = job.get_engine_name()
        try:
            engine_cls = registry.get_plugin_class("engine", engine_name)
        except Exception as e:
            # If we can't find the engine class, we can't validate dependencies.
            log.warning(
                f"Could not validate dependencies for engine '{engine_name}': {e}"
            )
            return

        if not hasattr(engine_cls, "manifest"):
            # Engine does not declare dependencies
            return

        manifest = engine_cls.manifest
        provided_plugins = self._effective_plugin_namespaces(job)

        # When an upstream input is provided and the engine declares input_plugins,
        # validate against those instead of the normal required plugins. This allows
        # engines to run in analysis/aggregation mode without their simulation
        # dependencies.
        required_namespaces = (
            manifest.input_plugins
            if job.input and manifest.input_plugins
            else manifest.required_plugins
        )

        # Check required plugins
        missing = required_namespaces - provided_plugins
        if missing:
            mode_hint = (
                " for input/analyze mode"
                if job.input and manifest.input_plugins
                else ""
            )
            raise QPhaseConfigError(
                f"Job '{job.name}' uses engine '{engine_name}' but is missing "
                f"required plugins{mode_hint}: {missing}"
            )

    def _effective_plugin_namespaces(self, job: JobConfig) -> set[str]:
        """Return all plugin namespaces available to a job.

        This mirrors the resolution logic used in :meth:`_run_job` so that
        validation and execution agree on which plugins are available.
        """
        namespaces: set[str] = set(job.plugins.keys())

        # Top-level plugin sections are stored as model extras by JobConfig
        job_extra = job.model_extra or {}
        for key in registered_plugin_namespaces():
            if key in job_extra:
                namespaces.add(key)

        # Merge project defaults so inherited required plugins are recognized.
        # are not reported as missing.
        try:
            job_override = {
                "plugins": job.plugins,
                "engine": job.engine,
                "params": job.params,
            }
            merged = get_config_for_job(self.project, job_config_dict=job_override)
            merged_plugins = merge_plugin_config_sections(merged)
            namespaces.update(merged_plugins.keys())
        except Exception as e:
            log.debug(
                f"Could not merge global config for plugin validation of "
                f"'{job.name}': {e}"
            )

        return namespaces

    def _validate_single_engine_per_job(self, workflow: WorkflowSpec) -> None:
        """Verify each job has exactly one engine."""
        for job in workflow.jobs:
            if not job.get_engine_name():
                raise QPhaseConfigError(
                    f"Job '{job.name}' is missing required 'engine' field"
                )

    def _validate_data_flow(self, workflow: WorkflowSpec) -> None:
        """Validate input/output data flow.

        Checks:
        - Input references are valid (job name or engine name with no ambiguity)
        - Output references are valid (optional - can point to multiple jobs)
        """
        jobs_by_name = {job.name: job for job in workflow.jobs}
        jobs_by_engine: dict[str, list[JobConfig]] = {}

        # Group jobs by engine name
        for job in workflow.jobs:
            engine_name = job.get_engine_name()
            if engine_name not in jobs_by_engine:
                jobs_by_engine[engine_name] = []
            jobs_by_engine[engine_name].append(job)

        # Validate input references
        for job in workflow.jobs:
            if not job.input:
                continue
            source = job.input.from_

            # Check if input matches a job name
            if source in jobs_by_name:
                # Valid job reference
                continue

            # Check if input matches an engine name
            upstream_jobs = jobs_by_engine.get(source, [])
            if not upstream_jobs:
                # Not a job name or engine name - could be a file path
                # This is valid (external input)
                log.debug(f"Job '{job.name}' input '{source}' appears to be external")
                continue

            # It's an engine name - check for ambiguity
            if len(upstream_jobs) > 1:
                job_names = ", ".join([j.name for j in upstream_jobs])
                raise QPhaseConfigError(
                    f"Job '{job.name}' input '{source}' is ambiguous. "
                    f"Multiple jobs use this engine: {job_names}. "
                    "Specify the exact job name instead."
                )

        # Validate output references (optional - just check for existence)
        for job in workflow.jobs:
            if not job.output:
                continue

            # Output can be a job name or engine name
            # We don't validate ambiguity for output since one job can feed
            # multiple downstream jobs
            if job.output in jobs_by_name or job.output in jobs_by_engine:
                log.debug(f"Job '{job.name}' output '{job.output}' is valid")
            else:
                # Could be a file path
                log.debug(
                    f"Job '{job.name}' output '{job.output}' appears to be external"
                )

    def _build_plugins(self, plugins_config: dict[str, Any]) -> dict[str, Any]:
        """Instantiate plugins based on configuration.

        Supports nested format: {plugin_type: {plugin_name: config}}
        or flat format: {plugin_type: {name: "...", params: {...}}}
        """
        plugins: dict[str, Any] = {}

        for plugin_type, config_data in plugins_config.items():
            if not config_data:
                continue

            # Check if this is nested format (plugin_name -> config)
            # or flat format (with 'name' field)
            if isinstance(config_data, dict) and "name" in config_data:
                # Flat format: {name: "...", params: {...}}
                try:
                    instance = registry.create_plugin_instance(plugin_type, config_data)
                    plugins[plugin_type] = instance
                except Exception as e:
                    raise QPhasePluginError(
                        f"Failed to create plugin '{plugin_type}': {e}"
                    ) from e
            elif isinstance(config_data, dict):
                # Nested format: {plugin_name: config, ...}
                # Create instances for each plugin
                type_instances = {}
                for plugin_name, plugin_config in config_data.items():
                    if not isinstance(plugin_config, dict):
                        continue

                    # Convert to flat format with name
                    flat_config = dict(plugin_config)
                    flat_config["name"] = plugin_name

                    try:
                        instance = registry.create_plugin_instance(
                            plugin_type, flat_config
                        )
                        # Store by specific name (e.g. "analyser.psd")
                        plugins[f"{plugin_type}.{plugin_name}"] = instance
                        type_instances[plugin_name] = instance
                    except Exception as e:
                        raise QPhasePluginError(
                            f"Failed to create plugin "
                            f"'{plugin_type}.{plugin_name}': {e}"
                        ) from e

                # Store single instance directly, multiple instances as dict
                if len(type_instances) == 1:
                    plugins[plugin_type] = list(type_instances.values())[0]
                else:
                    plugins[plugin_type] = type_instances

            else:
                raise QPhasePluginError(f"Invalid plugin config for '{plugin_type}'")

        return plugins

    def _create_job_dir(self, job: JobConfig) -> Path:
        """Create and return this logical Job's Artifact directory."""
        if self.session_dir:
            job_dir = self.session_dir / job.name
            job_dir.mkdir(parents=True, exist_ok=True)
            return job_dir

        # Defensive fallback; normal Workflow execution always owns a Session.
        output_root = self.project.session_root
        job_dir = output_root / job.name
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def _write_snapshot(
        self,
        job_dir: Path,
        job: JobConfig,
        config: dict[str, Any],
        job_idx: int,
    ) -> None:
        """Write configuration snapshot for reproducibility.

        Parameters
        ----------
        job_dir : Path
            Session directory for this Job.
        job : JobConfig
            Job configuration
        config : dict[str, Any]
            Merged configuration (global + job)
        job_idx : int
            Job index

        """
        try:
            from .snapshot import SnapshotManager

            # Extract validated plugins from job
            validated_plugins = job.get_all_plugin_configs()

            # Create and save snapshot
            snapshot_manager = SnapshotManager(self.project.session_root)

            # Create snapshot
            snapshot = snapshot_manager.create_snapshot(
                job=job,
                job_index=job_idx,
                system_config=self.system_config,
                validated_plugins=validated_plugins,
                engine_config=config.get("engine", {}),
                session_id=self.session_id,
                job_dir=job_dir,
                input_job=job.input.from_ if job.input is not None else None,
                output_job=job.output,
                metadata={
                    "scheduler_version": "2.0",
                    "snapshot_created_by": "scheduler",
                },
            )

            # Save snapshot
            snapshot_path = snapshot_manager.save_snapshot(snapshot, job_dir)
            log.debug(f"Snapshot saved to {snapshot_path}")

        except Exception as e:
            log.warning(f"Failed to write snapshot: {e}")

        # Don't raise - snapshot failure shouldn't stop job execution


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
