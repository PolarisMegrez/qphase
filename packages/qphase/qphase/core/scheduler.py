"""qphase: Job Scheduler
---------------------------------------------------------
Orchestrates the execution of simulation jobs, managing the complete lifecycle from
dependency resolution to result persistence. The Scheduler handles serial execution
of ``JobList`` items, passes logical scans to resource engines, manages run
directory creation, aggregates structured progress events into snapshots, and
builds structured error reports for failed jobs.

Public API
----------
`Scheduler` : Main class for job execution and lifecycle management.
`JobResult` : Dataclass containing job execution results and metadata.
`run_jobs` : Convenience function to execute a JobList.
"""

from __future__ import annotations

import inspect
import json
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from .artifacts import ArtifactStore
from .config import JobConfig, JobList
from .config_loader import get_config_for_job
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
    CancellationToken,
    CheckpointStore,
    ExecutionContext,
    ProgressReporter,
    ResourceSnapshot,
    execution_fingerprint,
    plugin_fingerprint,
)
from .logging_context import bind_log_context, set_log_context
from .progress import ProgressEvent, ProgressSnapshot, ProgressTracker
from .protocols import ResultProtocol
from .registry import registry
from .system_config import SystemConfig, load_system_config

log = get_logger()


@dataclass
class JobResult:
    """Result of a single job execution."""

    job_index: int
    job_name: str
    run_dir: Path
    run_id: str
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

    session_id: str
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
    default_output_dir : str | None, optional
        Override default output directory from system config.
    on_progress : Callable[[ProgressSnapshot], None] | None, optional
        Callback for progress snapshots during job execution.
    on_run_dir : Callable[[Path], None] | None, optional
        Callback invoked with run directory after each job completes.

    """

    system_config: SystemConfig
    default_output_dir: str
    session_id: str | None
    session_dir: Path | None
    manifest: SessionManifest | None

    def __init__(
        self,
        system_config: SystemConfig | None = None,
        default_output_dir: str | None = None,
        on_progress: Callable[[ProgressSnapshot], None] | None = None,
        on_run_dir: Callable[[Path], None] | None = None,
    ):
        if system_config is None:
            self.system_config = load_system_config()
        else:
            self.system_config = system_config

        if default_output_dir is None:
            self.default_output_dir = self.system_config.paths.output_dir
        else:
            self.default_output_dir = default_output_dir

        self.on_progress = on_progress
        self.on_run_dir = on_run_dir
        from .registry import registry

        self._registry = registry
        self.session_id = None
        self.session_dir = None
        self.manifest = None
        self._session_log_path: Path | None = None
        self._session_log_handler: Any | None = None
        self._run_statuses: dict[str, str] = {}

    def _initialize_session(self) -> None:
        """Initialize a new execution session."""
        # Generate session ID
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        short_uuid = uuid.uuid4().hex[:6]
        self.session_id = f"{ts}_{short_uuid}"

        # Create session directory
        output_root = Path(self.default_output_dir).resolve()
        self.session_dir = output_root / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Attach the per-session log file (full DEBUG content). A failure here
        # surfaces one explicit warning and never blocks the run.
        self._attach_session_log()

        # Initialize manifest
        self.manifest = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "jobs": {},
        }
        self._save_manifest()
        log.debug(f"Initialized session {self.session_id} at {self.session_dir}")

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
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(self.manifest, f, indent=2)
            except Exception as e:
                log.warning(f"Failed to save session manifest: {e}")

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
        job_list: JobList,
        dry_run: bool = False,
        resume_from: Path | None = None,
    ) -> list[JobResult]:
        """Execute all jobs in the job list serially.

        Parameters
        ----------
        job_list : JobList
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
        # Step 0: Initialize Session
        if resume_from:
            self._resume_session(resume_from)
        else:
            self._initialize_session()

        # Seed per-run job statuses from the manifest so that jobs depending
        # on a previously failed upstream are marked skipped_dependency.
        self._run_statuses = {}
        if self.manifest:
            for name, entry in self.manifest["jobs"].items():
                status = entry.get("status")
                if status in ("failed", "skipped_dependency"):
                    self._run_statuses[name] = status

        results: list[JobResult] = []
        job_results: dict[str, ResultProtocol] = {}
        logical_jobs = job_list.jobs
        try:
            if dry_run:
                log.info("Starting DRY RUN execution plan...")

            # Step 1: Validate jobs before execution
            self._validate_jobs(job_list)

            for job_idx, job in enumerate(logical_jobs):
                self._run_single(
                    job,
                    job_idx,
                    len(logical_jobs),
                    logical_jobs,
                    job_results,
                    results,
                    dry_run=dry_run,
                )

            if self.manifest and not dry_run:
                failed = any(result.status == "failed" for result in results)
                skipped = any(
                    result.status == "skipped_dependency" for result in results
                )
                self.manifest["status"] = (
                    "failed" if failed else "partial" if skipped else "completed"
                )
                self._save_manifest()
            return results
        except Exception:
            if self.manifest and not dry_run:
                self.manifest["status"] = "failed"
                self._save_manifest()
            raise
        finally:
            self._detach_session_log()

    def _resume_session(self, session_path: Path) -> None:
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
        self.session_id = self.manifest["session_id"]
        self.session_dir = session_path
        self._attach_session_log()
        log.info(f"Resuming session {self.session_id} from {self.session_dir}")

    def _handle_job_output(
        self,
        job: JobConfig,
        output_result: ResultProtocol,
        job_results: dict[str, ResultProtocol],
        run_dir: Path,
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
        run_dir : Path
            Run directory for this job (where results should be saved)
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
            # Build save path: run_dir / output_filename
            # Note: filename should not include extension -
            # ResultProtocol.save() will add appropriate extension
            save_path = run_dir / save_filename

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
                    hint="Check disk space and write permissions for the run "
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
                    run_dir=Path("dry_run"),
                    run_id="dry_run",
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
            self._update_job_status(job.name, "pending")

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
            )
        )

        # Skip jobs whose upstream failed or was skipped earlier in this run.
        # Independent downstream jobs keep running (existing scheduler policy).
        source = job.input.from_ if job.input else None
        upstream_status = self._run_statuses.get(source) if source else None
        if upstream_status in ("failed", "skipped_dependency"):
            note = (
                f"skipped: upstream job '{source}' "
                f"{upstream_status.replace('_', ' ')}"
            )
            log.info(f"Skipping job '{job.name}': {note}")
            result = JobResult(
                job_index=job_idx,
                job_name=job.name,
                run_dir=(self.session_dir / job.name)
                if self.session_dir
                else Path("."),
                run_id="",
                success=False,
                status="skipped_dependency",
                error_summary=note,
            )
            results.append(result)
            self._run_statuses[job.name] = "skipped_dependency"
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
            result = self._fail_job(job, job_idx, job_total, e, run_dir=None)
            results.append(result)
            self._run_statuses[job.name] = "failed"
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
            _JobOutcome(*raw_outcome)
            if isinstance(raw_outcome, tuple)
            else raw_outcome
        )
        results.append(outcome.result)
        self._run_statuses[job.name] = outcome.result.status

        if not outcome.result.success:
            self._record_failure(outcome.result)
            return

        assert outcome.output is not None and outcome.context is not None
        try:
            self._handle_job_output(
                job,
                outcome.output,
                job_results,
                outcome.result.run_dir,
                outcome.context,
            )
            outcome.context.checkpoints.complete()
        except Exception as e:
            failed = self._fail_job(
                job,
                job_idx,
                job_total,
                e,
                run_dir=outcome.result.run_dir,
                run_id=outcome.result.run_id,
            )
            results[-1] = failed
            self._run_statuses[job.name] = "failed"
            self._record_failure(failed)
            return

        assert self.session_dir is not None
        self._update_job_status(
            job.name,
            "completed",
            {
                "run_id": outcome.result.run_id,
                "output_dir": str(
                    outcome.result.run_dir.relative_to(self.session_dir)
                ),
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

    def _fail_job(
        self,
        job: JobConfig,
        job_idx: int,
        job_total: int,
        exc: BaseException,
        *,
        run_dir: Path | None,
        run_id: str = "",
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
            run_dir=run_dir,
            scan_context=self._scan_summary(job),
            log_file=self._session_log_path,
        )
        if run_dir is not None:
            target_dir = run_dir
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
                run_dir=str(target_dir),
                error=summary,
                message=report.summary,
            )
        )
        return JobResult(
            job_index=job_idx,
            job_name=job.name,
            run_dir=target_dir,
            run_id=run_id,
            success=False,
            status="failed",
            error_summary=report.summary,
            error_id=report.error_id,
            error_code=report.code,
            error_report_path=(
                str(report_path) if report_path is not None else None
            ),
        )

    def _get_merged_config_for_job(self, job: JobConfig) -> dict[str, Any]:
        """Merge global system config with job-specific overrides.

        Returns
        -------
        dict[str, Any]
            Merged configuration dictionary containing plugins, engine,
            params, and any top-level plugin sections defined in the job.

        """
        system_cfg = job.merge_with_system_config(self.system_config)

        # Plugin namespaces that may appear as top-level keys in a job file.
        plugin_keys = [
            "backend",
            "integrator",
            "model",
            "analyser",
            "visualizer",
            "analyzer",
        ]

        # Merge global config with job config
        job_override: dict[str, Any] = {
            "plugins": job.plugins,
            "engine": job.engine,
            "params": job.params,
        }
        # Preserve top-level plugin sections (e.g. backend, analyser) that live in
        # JobConfig.model_extra so the merge/extraction logic sees them.
        job_extra = job.model_extra or {}
        for key in plugin_keys:
            if key in job_extra:
                job_override[key] = job_extra[key]

        return get_config_for_job(
            system_cfg, job_name=job.name, job_config_dict=job_override
        )

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
        1. Create run directory and generate run ID
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
        run_id = self._generate_run_id()
        run_dir = self._create_run_dir(job, run_id)
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
                    run_id=run_id,
                    run_dir=run_dir,
                    engine_name=engine_name,
                    tracker=tracker,
                    clock_start=clock_start,
                )
            except Exception as e:
                result = self._fail_job(
                    job,
                    job_idx,
                    display_total,
                    e,
                    run_dir=run_dir,
                    run_id=run_id,
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
        run_id: str,
        run_dir: Path,
        engine_name: str,
        tracker: ProgressTracker,
        clock_start: float,
    ) -> _JobOutcome:
        """Job body executed under the error boundary of :meth:`_run_job`."""
        merged_config = self._get_merged_config_for_job(job)

        # Plugin namespaces that may appear as top-level keys in a job file.
        plugin_keys = [
            "backend",
            "integrator",
            "model",
            "analyser",
            "visualizer",
            "analyzer",
        ]

        # Normalize config: move top-level plugin keys to 'plugins' if not present
        # This supports the simplified config format where plugins are at the root
        plugins_cfg = merged_config.get("plugins", {}).copy()
        for key in plugin_keys:
            if key in merged_config and key not in plugins_cfg:
                plugins_cfg[key] = merged_config[key]

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
        for key in plugin_keys:
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

        # Inject run_dir as output_dir for engines that support it
        # (e.g. VizEngine). We cast to str because config expects str.
        # Engines that don't support this field should have extra="allow"
        # in their config schema.
        engine_config_raw["output_dir"] = str(run_dir)

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
        self._write_snapshot(run_dir, job, merged_config, job_idx)

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
        context = ExecutionContext(
            parameter_grid=job.scan.compile() if job.scan is not None else None,
            resources=ResourceSnapshot.from_system_config(
                effective_system, backend=backend
            ),
            progress=reporter,
            cancellation=CancellationToken(),
            artifacts=ArtifactStore(run_dir, effective_system.scan_runtime),
            checkpoints=CheckpointStore(
                run_dir,
                effective_system.scan_runtime.checkpoint,
                fingerprint,
            ),
            run_dir=run_dir,
            metadata={
                "job_name": job.name,
                "scan_summary": self._scan_summary(job),
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
                run_dir=str(run_dir),
            )
        )

        if self.on_run_dir is not None:
            self.on_run_dir(run_dir)

        return _JobOutcome(
            result=JobResult(
                job_index=job_idx,
                job_name=job.name,
                run_dir=run_dir,
                run_id=run_id,
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

    def _validate_jobs(self, job_list: JobList) -> None:
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
        self._validate_single_engine_per_job(job_list)

        # Stage 2: Validate engine dependencies
        for job in job_list.jobs:
            self._validate_job_dependencies(job)

        # Stage 3: Validate data flow
        self._validate_data_flow(job_list)

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
        plugin_keys = [
            "backend",
            "integrator",
            "model",
            "analyser",
            "visualizer",
            "analyzer",
        ]

        namespaces: set[str] = set(job.plugins.keys())

        # Top-level plugin sections are stored as model extras by JobConfig
        job_extra = job.model_extra or {}
        for key in plugin_keys:
            if key in job_extra:
                namespaces.add(key)

        # Merge in global defaults so required plugins supplied by global.yaml
        # are not reported as missing.
        system_cfg = job.system if job.system is not None else self.system_config
        try:
            job_override = {
                "plugins": job.plugins,
                "engine": job.engine,
                "params": job.params,
            }
            merged = get_config_for_job(
                system_cfg, job_name=job.name, job_config_dict=job_override
            )
            merged_plugins = dict(merged.get("plugins", {}))
            for key in plugin_keys:
                if key in merged and key not in merged_plugins:
                    merged_plugins[key] = merged[key]
            namespaces.update(merged_plugins.keys())
        except Exception as e:
            log.debug(
                f"Could not merge global config for plugin validation of "
                f"'{job.name}': {e}"
            )

        return namespaces

    def _validate_single_engine_per_job(self, job_list: JobList) -> None:
        """Verify each job has exactly one engine."""
        for job in job_list.jobs:
            if not job.get_engine_name():
                raise QPhaseConfigError(
                    f"Job '{job.name}' is missing required 'engine' field"
                )

    def _validate_data_flow(self, job_list: JobList) -> None:
        """Validate input/output data flow.

        Checks:
        - Input references are valid (job name or engine name with no ambiguity)
        - Output references are valid (optional - can point to multiple jobs)
        """
        jobs_by_name = {job.name: job for job in job_list.jobs}
        jobs_by_engine: dict[str, list[JobConfig]] = {}

        # Group jobs by engine name
        for job in job_list.jobs:
            engine_name = job.get_engine_name()
            if engine_name not in jobs_by_engine:
                jobs_by_engine[engine_name] = []
            jobs_by_engine[engine_name].append(job)

        # Validate input references
        for job in job_list.jobs:
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
        for job in job_list.jobs:
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

    def _generate_run_id(self) -> str:
        """Generate a unique run ID with timestamp and UUID suffix."""
        # In session mode, run_id can be simpler or just a UUID,
        # but we keep the timestamp for consistency.
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        return f"{ts}_{uuid.uuid4().hex[:8]}"

    def _create_run_dir(self, job: JobConfig, run_id: str) -> Path:
        """Create and return the run directory for a job."""
        # If session is active, create directory inside session dir
        if self.session_dir:
            run_dir = self.session_dir / job.name
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir

        # Fallback for non-session execution (should not happen in normal flow
        # Get the effective system config (job.system overrides global)
        effective_system = job.system if job.system is not None else self.system_config
        output_dir = effective_system.paths.output_dir

        output_root = Path(output_dir).resolve()
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_snapshot(
        self,
        run_dir: Path,
        job: JobConfig,
        config: dict[str, Any],
        job_idx: int,
    ) -> None:
        """Write configuration snapshot for reproducibility.

        Parameters
        ----------
        run_dir : Path
            Run directory for this job
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
            snapshot_manager = SnapshotManager(
                Path(self.system_config.paths.output_dir)
            )

            # Get run_id from run_dir if available
            run_id = run_dir.name if run_dir.name else None

            # Create snapshot
            snapshot = snapshot_manager.create_snapshot(
                job=job,
                job_index=job_idx,
                system_config=self.system_config,
                validated_plugins=validated_plugins,
                engine_config=config.get("engine", {}),
                run_id=run_id,
                run_dir=run_dir,
                input_job=job.input.from_ if job.input is not None else None,
                output_job=job.output,
                metadata={
                    "scheduler_version": "2.0",
                    "snapshot_created_by": "scheduler",
                },
            )

            # Save snapshot
            snapshot_path = snapshot_manager.save_snapshot(snapshot, run_dir)
            log.debug(f"Snapshot saved to {snapshot_path}")

        except Exception as e:
            log.warning(f"Failed to write snapshot: {e}")

        # Don't raise - snapshot failure shouldn't stop job execution


def run_jobs(
    job_list: JobList,
    *,
    default_output_dir: str | None = None,
    on_progress: Callable[[ProgressSnapshot], None] | None = None,
    on_run_dir: Callable[[Path], None] | None = None,
) -> list[JobResult]:
    """Execute a list of jobs.

    Creates a Scheduler instance and runs all jobs in the provided job list.

    Parameters
    ----------
    job_list : JobList
        List of jobs to execute
    default_output_dir : str | None, optional
        Override default output directory
    on_progress : Callable[[ProgressSnapshot], None] | None, optional
        Progress callback function
    on_run_dir : Callable[[Path], None] | None, optional
        Callback invoked with run directory after each job completes

    Returns
    -------
    list[JobResult]
        Results for each executed job

    """
    scheduler = Scheduler(
        default_output_dir=default_output_dir,
        on_progress=on_progress,
        on_run_dir=on_run_dir,
    )
    return scheduler.run(job_list)
