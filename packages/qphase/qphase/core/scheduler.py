"""qphase: Job Scheduler
---------------------------------------------------------
Orchestrates the execution of simulation jobs, managing the complete lifecycle from
dependency resolution to result persistence. The Scheduler handles serial execution
of ``JobList`` items, expands parameter scans into multiple tasks, manages run
directory creation, and provides hooks for progress reporting and snapshot generation.

Public API
----------
`Scheduler` : Main class for job execution and lifecycle management.
`JobResult` : Dataclass containing job execution results and metadata.
`JobProgressUpdate` : Dataclass for progress callback information.
`run_jobs` : Convenience function to execute a JobList.
"""

from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from .aggregation import AggregateResult
from .batch_negotiator import BatchJob, BatchNegotiator, SingleJob
from .config import JobConfig, JobList
from .config_loader import get_config_for_job
from .errors import (
    QPhaseConfigError,
    QPhasePluginError,
    QPhaseRuntimeError,
    get_logger,
)
from .job_expansion import JobExpander
from .protocols import ResultProtocol
from .registry import registry
from .system_config import SystemConfig, load_system_config

log = get_logger()


@dataclass
class JobProgressUpdate:
    """Progress update for a single job."""

    job_name: str
    job_index: int
    total_jobs: int
    message: str
    percent: float | None = None
    job_eta: float | None = None
    global_eta: float | None = None
    stage: str | None = None


@dataclass
class JobResult:
    """Result of a single job execution."""

    job_index: int
    job_name: str
    run_dir: Path
    run_id: str
    success: bool
    error: str | None = None


class SessionManifest(TypedDict):
    """Type definition for session manifest."""

    session_id: str
    start_time: str
    status: str
    jobs: dict[str, dict[str, Any]]


class Scheduler:
    """Scheduler for executing simulation jobs.

    Manages serial job execution with dependency resolution, parameter scanning,
    configuration merging, and progress reporting.

    Parameters
    ----------
    system_config : SystemConfig | None, optional
        System configuration. If None, loads from system.yaml.
    default_output_dir : str | None, optional
        Override default output directory from system config.
    on_progress : Callable[[JobProgressUpdate], None] | None, optional
        Callback for progress updates during job execution.
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
        on_progress: Callable[[JobProgressUpdate], None] | None = None,
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

        # Initialize manifest
        self.manifest = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "jobs": {},
        }
        self._save_manifest()
        log.info(f"Initialized session {self.session_id} at {self.session_dir}")

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

        if dry_run:
            log.info("Starting DRY RUN execution plan...")

        # Step 1: Validate jobs before execution
        self._validate_jobs(job_list)

        # Step 2: Expand parameter scan jobs
        expanded_jobs = self._expand_parameter_scans(job_list)

        results: list[JobResult] = []
        job_results: dict[str, ResultProtocol] = {}

        # Group expanded jobs into single or batched execution units.
        # Resource packs advertise BatchPlanners that decide how scan jobs can be
        # fused; the scheduler only orchestrates and preserves per-job identity.
        negotiator = BatchNegotiator(registry)
        job_groups = negotiator.group_jobs(expanded_jobs)

        # Map original job name to its group index for stable ordering.
        group_count = len(job_groups)

        for group_idx, group in enumerate(job_groups):
            if isinstance(group, SingleJob):
                self._run_single(
                    group.job,
                    group_idx,
                    group_count,
                    expanded_jobs,
                    job_results,
                    results,
                    dry_run=dry_run,
                )
            elif isinstance(group, BatchJob):
                self._run_batch(
                    group.plan,
                    group.original_jobs,
                    group_idx,
                    group_count,
                    job_results,
                    results,
                    dry_run=dry_run,
                )

        # Finalize session
        if self.manifest and not dry_run:
            self.manifest["status"] = (
                "completed" if all(r.success for r in results) else "failed"
            )
            self._save_manifest()

        return results

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
        log.info(f"Resuming session {self.session_id} from {self.session_dir}")

    def _handle_job_output(
        self,
        job: JobConfig,
        output_result: ResultProtocol,
        job_results: dict[str, ResultProtocol],
        run_dir: Path,
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
                output_result.save(save_path)
                log.debug(f"Job '{job.name}' result saved to {save_path}")
            except Exception as e:
                raise QPhaseRuntimeError(
                    f"Failed to save job '{job.name}' output to '{save_path}': {e}"
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
        if not job.input:
            return None

        # Check if input is from a previous job in current session
        if job.input in job_results:
            return job_results[job.input]

        # Check for parameter scan aggregation (N-to-1)
        # If job.input matches the base name of a set of expanded jobs
        # e.g. input="sim", but we have "sim[p=1]", "sim[p=2]"
        # Note: Expansion uses "_" separator for numbering, e.g. "sim_001"
        scan_prefix = f"{job.input}_"
        aggregated_results = {
            k: v
            for k, v in job_results.items()
            if k.startswith(scan_prefix) or k.startswith(f"{job.input}[")
        }

        # Apply implicit filtering if present (from aggregate_input expansion)
        input_filter = job.params.get("_input_filter")
        if input_filter and aggregated_results:
            filtered_results = {}
            for k, result in aggregated_results.items():
                match = True
                # Access result metadata for parameters
                # We assume standard metadata structure from Engine execution
                if hasattr(result, "metadata"):
                    res_params = result.metadata.get("params", {})
                    # Also check top-level if not in params, just in case
                    if not res_params and hasattr(result, "params"):
                        res_params = getattr(result, "params", {})
                else:
                    res_params = {}

                for filter_k, filter_v in input_filter.items():
                    # filter_k matches parameter name
                    val = res_params.get(filter_k)

                    if isinstance(val, float) and isinstance(filter_v, float):
                        if abs(val - filter_v) > 1e-7:
                            match = False
                            break
                    elif val != filter_v:
                        match = False
                        break

                if match:
                    filtered_results[k] = result

            aggregated_results = filtered_results
            log.info(
                f"Filtered input for job '{job.name}': matched "
                f"{len(aggregated_results)} results using filter {input_filter}"
            )

        if aggregated_results:
            # We found multiple results matching the input pattern.
            return AggregateResult(
                aggregated_results,
                meta={"aggregated": True, "count": len(aggregated_results)},
            )

        # Check if input is in manifest (from a previous run in same session context)
        if self.manifest and job.input in self.manifest["jobs"]:
            job_entry = self.manifest["jobs"][job.input]
            if job_entry.get("status") == "completed" and self.session_dir:
                output_rel_path = job_entry.get("output_dir")
                if output_rel_path:
                    job_dir = self.session_dir / output_rel_path
                    try:
                        from .result_loader import load_result

                        log.info(f"Loading result for '{job.input}' from disk...")
                        result = load_result(job.input, job_dir)
                        # Cache it
                        job_results[job.input] = result
                        return result
                    except Exception as e:
                        log.warning(
                            f"Failed to load result for '{job.input}' from disk: {e}"
                        )

        # Check if input is an external directory or file
        input_path = Path(job.input)
        if input_path.exists():
            if input_path.is_dir():
                log.info(
                    f"Job '{job.name}' input '{job.input}' is a directory; "
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
                f"Job '{job.name}' specifies file input '{job.input}', "
                "but file loading is not currently supported."
            )

        # Input not found
        raise QPhaseConfigError(
            f"Job '{job.name}' input '{job.input}' not found. "
            f"Expected a previous job name or a valid file path with input_loader."
        )

    def _run_single(
        self,
        job: JobConfig,
        job_idx: int,
        job_total: int,
        expanded_jobs: list[JobConfig],
        job_results: dict[str, ResultProtocol],
        results: list[JobResult],
        *,
        dry_run: bool,
    ) -> None:
        """Execute one non-batched job and update the shared result state."""
        # Check if job is already completed (Resume Mode)
        if self.manifest and job.name in self.manifest["jobs"]:
            job_status = self.manifest["jobs"][job.name].get("status")
            if job_status == "completed":
                log.info(f"Skipping completed job: {job.name}")
                return

        # Register job in manifest
        if not dry_run:
            self._update_job_status(job.name, "pending")

        try:
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

            # Normal Execution
            input_result = self._resolve_input(job, job_results)
            job_result, output_result = self._run_job(
                job,
                job_idx,
                job_total,
                input_result,
                display_total=len(expanded_jobs),
            )
            results.append(job_result)

            if job_result.success:
                self._handle_job_output(
                    job, output_result, job_results, job_result.run_dir
                )
                assert self.session_dir is not None
                self._update_job_status(
                    job.name,
                    "completed",
                    {
                        "run_id": job_result.run_id,
                        "output_dir": str(
                            job_result.run_dir.relative_to(self.session_dir)
                        ),
                    },
                )
            else:
                self._update_job_status(job.name, "failed")

        except Exception as e:
            log.error(f"Job {job.name} failed: {e}")
            if not dry_run:
                self._update_job_status(job.name, "failed", {"error": str(e)})
            results.append(
                JobResult(
                    job_index=job_idx,
                    job_name=job.name,
                    run_dir=Path("."),
                    run_id="",
                    success=False,
                    error=str(e),
                )
            )

    def _get_merged_config_for_job(self, job: JobConfig) -> dict[str, Any]:
        """Merge global system config with job-specific overrides.

        Returns
        -------
        dict[str, Any]
            Merged configuration dictionary containing plugins, engine,
            params, and any top-level plugin sections defined in the job.
        """
        system_cfg = job.system if job.system is not None else self.system_config

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

    def _run_batch(
        self,
        plan: Any,
        original_jobs: list[JobConfig],
        group_idx: int,
        group_total: int,
        job_results: dict[str, ResultProtocol],
        results: list[JobResult],
        *,
        dry_run: bool,
    ) -> None:
        """Execute a batch of jobs and split the result back per original job."""
        # Resume check: skip only if all original jobs are completed.
        if self.manifest:
            all_completed = all(
                self.manifest["jobs"].get(j.name, {}).get("status") == "completed"
                for j in original_jobs
            )
            if all_completed:
                log.info(f"Skipping completed batch: {', '.join(j.name for j in original_jobs)}")
                return

        batch_name = f"__batch_{group_idx}"
        if not dry_run:
            self._update_job_status(batch_name, "pending")

        try:
            if dry_run:
                for j in original_jobs:
                    log.info(f"[DRY-RUN] Would execute job (batched): {j.name}")
                    results.append(
                        JobResult(
                            job_index=group_idx,
                            job_name=j.name,
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
                    job_results[j.name] = MockResult()
                return

            # Batch jobs currently only support simulations without upstream input.
            input_result = None
            if any(j.input for j in original_jobs):
                raise QPhaseConfigError(
                    "Batch execution does not yet support jobs with upstream input"
                )

            # Build and run the merged job via the resource pack's BatchPlanner.
            # Use a temporary job name for the combined run directory; the real
            # per-job directories are created after splitting.
            batch_job = copy.deepcopy(plan.batch_job)
            batch_job.name = batch_name
            run_id = self._generate_run_id()
            run_dir = self._create_run_dir(batch_job, run_id)

            _, output_result = self._run_job(
                batch_job,
                group_idx,
                group_total,
                input_result,
                display_total=len(original_jobs),
            )

            # Split the combined result back into per-job results.
            splitter = registry.get_result_splitter(plan.result_splitter)
            split_results = splitter.split(output_result, original_jobs)

            for j in original_jobs:
                single_result = split_results.get(j.name)
                if single_result is None:
                    raise QPhaseRuntimeError(
                        f"Batch result splitter did not produce result for '{j.name}'"
                    )

                # Each original job gets its own run directory and manifest entry.
                single_run_id = self._generate_run_id()
                single_run_dir = self._create_run_dir(j, single_run_id)

                # Write per-job snapshot so downstream tools can inspect the
                # original (non-batched) configuration.
                single_merged_config = self._get_merged_config_for_job(j)
                self._write_snapshot(
                    single_run_dir, j, single_merged_config, group_idx
                )

                self._handle_job_output(j, single_result, job_results, single_run_dir)

                assert self.session_dir is not None
                self._update_job_status(
                    j.name,
                    "completed",
                    {
                        "run_id": single_run_id,
                        "output_dir": str(single_run_dir.relative_to(self.session_dir)),
                        "batched": True,
                        "batch_group": group_idx,
                    },
                )

                results.append(
                    JobResult(
                        job_index=group_idx,
                        job_name=j.name,
                        run_dir=single_run_dir,
                        run_id=single_run_id,
                        success=True,
                    )
                )

            self._update_job_status(batch_name, "completed")

        except Exception as e:
            log.error(f"Batch group {group_idx} failed: {e}")
            if not dry_run:
                self._update_job_status(batch_name, "failed", {"error": str(e)})
            for j in original_jobs:
                if not dry_run:
                    self._update_job_status(j.name, "failed", {"error": str(e)})
                results.append(
                    JobResult(
                        job_index=group_idx,
                        job_name=j.name,
                        run_dir=Path("."),
                        run_id="",
                        success=False,
                        error=str(e),
                    )
                )

    def _run_job(
        self,
        job: JobConfig,
        job_idx: int,
        job_total: int,
        input_result: ResultProtocol | None,
        *,
        display_total: int | None = None,
    ) -> tuple[JobResult, ResultProtocol]:
        """Execute a single job and return its result.

        This method handles the complete job execution lifecycle:
        1. Create run directory and generate run ID
        2. Merge global config with job-specific config
        3. Build plugin instances from configuration
        4. Instantiate and run the engine
        5. Report progress and save snapshot

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
            Number of jobs to display in progress reporting. When a job is
            batched, this is the number of original (pre-batch) jobs so that
            ETA and progress percentages reflect the user's mental model.

        Returns
        -------
        tuple[JobResult, ResultProtocol]
            Tuple of (job execution metadata, engine output result)

        Raises
        ------
        QPhasePluginError
            If engine instantiation fails
        QPhaseRuntimeError
            If engine execution fails

        """
        display_total = job_total if display_total is None else display_total
        run_id = self._generate_run_id()
        run_dir = self._create_run_dir(job, run_id)

        merged_config = self._get_merged_config_for_job(job)

        try:
            from .registry import registry
        except ImportError:
            # Should be available
            pass

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
        job_engine_name = job.get_engine_name()

        target_engine_name = None
        if job_engine_name:
            target_engine_name = job_engine_name
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
                f"Failed to instantiate engine '{job.get_engine_name()}': {e}"
            ) from e

        # Also write snapshot
        self._write_snapshot(run_dir, job, merged_config, job_idx)

        # Report job start
        if self.on_progress is not None:
            self.on_progress(
                JobProgressUpdate(
                    job_name=job.name,
                    job_index=job_idx,
                    total_jobs=display_total,
                    message="Starting job...",
                )
            )

        try:
            # Prepare progress callback
            progress_cb = None
            if self.on_progress is not None:
                import time

                last_update_time = 0.0
                min_interval = self.system_config.progress_update_interval

                def _on_engine_progress(
                    percent: float | None,
                    total_duration_estimate: float | None,
                    message: str,
                    stage: str | None,
                ) -> None:
                    nonlocal last_update_time
                    now = time.monotonic()

                    # Rate limiting: only update if interval passed or job finished
                    # (percent=1.0)
                    if (
                        now - last_update_time < min_interval
                        and percent is not None
                        and percent < 1.0
                    ):
                        return

                    last_update_time = now

                    # Calculate Job ETA
                    job_eta = None
                    if (
                        percent is not None
                        and total_duration_estimate is not None
                        and percent > 0
                    ):
                        job_eta = total_duration_estimate * (1.0 - percent)

                    # Calculate Global ETA (only for expanded jobs)
                    # Heuristic: if display_total > 1, assume homogeneous expansion
                    global_eta = None
                    if (
                        display_total > 1
                        and job_eta is not None
                        and total_duration_estimate is not None
                    ):
                        # Simple extrapolation: remaining jobs * current job total
                        # duration. Use display_total so batching is accounted for.
                        remaining_jobs = display_total - job_idx
                        global_eta = job_eta + (
                            remaining_jobs * total_duration_estimate
                        )

                    if self.on_progress is not None:
                        self.on_progress(
                            JobProgressUpdate(
                                job_name=job.name,
                                job_index=job_idx,
                                total_jobs=display_total,
                                message=message,
                                percent=percent,
                                job_eta=job_eta,
                                global_eta=global_eta,
                                stage=stage,
                            )
                        )

                progress_cb = _on_engine_progress

            # Execute engine
            # Pass input result's data to engine.
            # For analyze mode, preserve the full result object (e.g. AggregateResult
            # or SDEResult) so that cross-job analyzers can access metadata.
            if engine_config_raw.get("mode") == "analyze":
                input_data = input_result
            else:
                input_data = input_result.data if input_result else None

            # Check if engine accepts progress_cb (Duck Typing / Inspection)
            # Since EngineBase protocol defines it as optional kwarg, we try passing it.
            # However, some legacy engines might not accept **kwargs or progress_cb.
            # Ideally, all engines should accept **kwargs.
            try:
                output_result = engine.run(data=input_data, progress_cb=progress_cb)
            except TypeError:
                # Fallback for engines that don't accept progress_cb
                log.warning(
                    f"Engine '{job.get_engine_name()}' does not accept progress_cb. "
                    "Progress reporting disabled."
                )
                output_result = engine.run(data=input_data)

            # Ensure output is a ResultProtocol object
            if not isinstance(output_result, ResultProtocol):
                raise QPhaseRuntimeError(
                    f"Engine '{job.get_engine_name()}' did not return a "
                    f"ResultProtocol object. "
                    f"All engines must return a ResultProtocol instance from "
                    f"their run() method."
                )

            # Report job completion
            if self.on_progress is not None:
                self.on_progress(
                    JobProgressUpdate(
                        job_name=job.name,
                        job_index=job_idx,
                        total_jobs=job_total,
                        message="Completed successfully",
                        percent=1.0,
                    )
                )

            if self.on_run_dir is not None:
                self.on_run_dir(run_dir)

            return (
                JobResult(
                    job_index=job_idx,
                    job_name=job.name,
                    run_dir=run_dir,
                    run_id=run_id,
                    success=True,
                ),
                output_result,
            )

        except Exception as e:
            # Report job failure
            if self.on_progress is not None:
                self.on_progress(
                    JobProgressUpdate(
                        job_name=job.name,
                        job_index=job_idx,
                        total_jobs=job_total,
                        message=f"Failed: {e}",
                        percent=0.0,
                    )
                )

            log.error(f"Job execution failed: {e}")
            raise QPhaseRuntimeError(
                f"Job '{job.name}' execution failed in engine "
                f"'{job.get_engine_name()}': {e}"
            ) from e

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
        log.info("Validating job configurations...")

        # Stage 1: Check each job has exactly one engine
        self._validate_single_engine_per_job(job_list)

        # Stage 2: Validate engine dependencies
        for job in job_list.jobs:
            self._validate_job_dependencies(job)

        # Stage 3: Validate data flow
        self._validate_data_flow(job_list)

        log.info("Job validation completed successfully")

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

    def _expand_parameter_scans(self, job_list: JobList) -> list[JobConfig]:
        """Expand jobs with scanable list parameters into multiple jobs.

        This method scans job configurations for parameters marked as 'scanable'
        and expands them into multiple jobs based on the configured expansion method.
        It also handles dependency expansion: if a job depends on an upstream job
        that was expanded, the downstream job is also expanded to match.

        Parameters
        ----------
        job_list : JobList
            List of jobs to expand

        Returns
        -------
        list[JobConfig]
            Expanded list of jobs with scanable parameters expanded into individual jobs

        """
        # Check if parameter scan is enabled
        if not self.system_config.parameter_scan.get("enabled", True):
            log.debug("Parameter scan is disabled, returning original jobs")
            return job_list.jobs

        expanded_jobs = []
        numbering_enabled = self.system_config.parameter_scan.get(
            "numbered_outputs", True
        )
        method = self.system_config.parameter_scan.get("method", "cartesian")

        expander = JobExpander(self._registry)

        # Map original job name to list of expanded job names
        # e.g. "vdp_sde" -> ["vdp_sde_001", "vdp_sde_002"]
        expansion_map: dict[str, list[str]] = {}
        # Map expanded job name to JobConfig object for parameter inspection
        job_registry: dict[str, JobConfig] = {}

        for job in job_list.jobs:
            # 1. Check if this job needs to be expanded due to upstream dependency
            upstream_expansion = []
            if job.input and job.input in expansion_map:
                upstream_expansion = expansion_map[job.input]

            # 2. Check if this job has its own parameter scan
            scan_expansion = expander.expand(job, method=method)

            final_expansion = []

            # Check for aggregation request
            aggregation_cfg = job.aggregate_input

            if aggregation_cfg and upstream_expansion:
                # Aggregation logic
                target_path = aggregation_cfg.get("on")
                if not target_path:
                    log.warning(f"Job {job.name} has aggregate_input but no 'on' field")
                    continue

                # We need to group upstream jobs by "everything else"
                # First, identify all varying parameters in upstream jobs
                upstream_jobs = [job_registry[name] for name in upstream_expansion]

                # Helper to get value by dotted path
                def get_val(obj, path):
                    parts = path.split(".")
                    curr = obj
                    for p in parts:
                        if isinstance(curr, dict):
                            curr = curr.get(p)
                        elif hasattr(curr, p):
                            curr = getattr(curr, p)
                        else:
                            return None
                    return curr

                # Find varying parameters with a deep search.
                # Scan the full upstream job config for varying values.

                varying_keys = set()

                # Helper to flatten dict with dot notation
                def flatten_dict(d: dict, prefix: str = "") -> dict[str, Any]:
                    items = {}
                    for k, v in d.items():
                        new_key = f"{prefix}.{k}" if prefix else k
                        if isinstance(v, dict):
                            items.update(flatten_dict(v, new_key))
                        else:
                            items[new_key] = v
                    return items

                # Helper to flatten relevant job parts
                def flatten_job(j: JobConfig) -> dict[str, Any]:
                    flat = {}
                    # Inspect params, engine, plugins
                    if j.params:
                        flat.update(flatten_dict(j.params, "params"))
                    if j.engine:
                        flat.update(flatten_dict(j.engine, "engine"))
                    if j.plugins:
                        flat.update(flatten_dict(j.plugins, "plugins"))
                    return flat

                if upstream_jobs:
                    # Scan keys from the first job
                    # Note: We assume all jobs share the same schema structure for keys
                    first_flat = flatten_job(upstream_jobs[0])
                    candidate_keys = list(first_flat.keys())

                    # Check variance for each key
                    for key in candidate_keys:
                        values = set()
                        for uj in upstream_jobs:
                            val = get_val(uj, key)
                            # Handle unhashable types (list, dict) by stringifying
                            try:
                                values.add(val)
                            except TypeError:
                                values.add(str(val))

                        if len(values) > 1:
                            varying_keys.add(key)

                # Remove the aggregation target from varying keys.
                # Also support shorthand targets like "epsilon" or "params.epsilon"
                # that match nested keys such as "plugins.model.kerr_3pa.epsilon".
                if target_path in varying_keys:
                    varying_keys.remove(target_path)
                else:
                    target_tail = target_path.split(".")[-1]
                    matching = [
                        k
                        for k in varying_keys
                        if k.endswith(f".{target_tail}") or k == target_tail
                    ]
                    if matching:
                        varying_keys.discard(matching[0])

                # Group jobs by the remaining varying keys
                groups: dict[tuple[Any, ...], list[Any]] = {}

                for uj in upstream_jobs:
                    # signature is a tuple of values for varying keys
                    sig_list = []
                    for k in sorted(varying_keys):
                        sig_list.append(get_val(uj, k))
                    sig = tuple(sig_list)

                    # sig = tuple(get_val(uj, k) for k in sorted(varying_keys))
                    if sig not in groups:
                        groups[sig] = []
                    groups[sig].append(uj)

                log.info(
                    f"Aggregating {job.name} on {target_path}. Found {len(groups)} "
                    f"groups from {len(upstream_expansion)} upstream jobs."
                )

                # Create a job for each group
                for idx, (sig, _group_jobs) in enumerate(groups.items()):
                    new_job = copy.deepcopy(job)

                    # Construct input filter logic
                    # We pass the filter to _resolve_input via metadata in params
                    # The filter defines the fixed parameters for this group
                    input_filter = {}
                    group_name_suffix_parts = []

                    for i, key in enumerate(sorted(varying_keys)):
                        # key is "params.k"
                        param_name = key.split(".")[-1]
                        val = sig[i]
                        input_filter[param_name] = val
                        group_name_suffix_parts.append(f"{param_name}={val}")

                    if not group_name_suffix_parts:
                        group_name_suffix = "all"
                    else:
                        group_name_suffix = "_".join(group_name_suffix_parts)

                    # Store filter in params (private key)
                    new_job.params["_input_filter"] = input_filter
                    new_job.params["_aggregated_on"] = target_path

                    # Update name.
                    # Append a suffix when multiple groups are present.
                    if len(groups) > 1:
                        # Keep filenames clean and predictable.
                        if numbering_enabled:
                            new_job = self._apply_job_numbering(
                                new_job, job.name, idx + 1, len(groups)
                            )
                        else:
                            new_job.name = f"{job.name}_{group_name_suffix}"

                    final_expansion.append(new_job)

            elif upstream_expansion:
                # Case A: Follow upstream expansion
                if len(scan_expansion) > 1:
                    # If both are present, we might need a more complex strategy.
                    # For now, warn and prioritize upstream to maintain flow.
                    log.warning(
                        f"Job '{job.name}' has both upstream expansion "
                        f"(from '{job.input}') and internal parameter scan. "
                        "Prioritizing upstream expansion structure."
                    )

                # Create N copies of the job, each pointing to one upstream output
                for idx, upstream_job_name in enumerate(upstream_expansion):
                    new_job = copy.deepcopy(job)
                    # Update input to point to specific upstream job
                    new_job.input = upstream_job_name

                    # Apply numbering to match upstream index
                    # We use the same index (1-based)
                    # Only apply numbering if upstream was actually expanded (count > 1)
                    if numbering_enabled and len(upstream_expansion) > 1:
                        # Determine padding based on total count
                        total_count = len(upstream_expansion)
                        new_job = self._apply_job_numbering(
                            new_job, job.name, idx + 1, total_count
                        )

                    final_expansion.append(new_job)

            elif len(scan_expansion) > 1:
                # Case B: Standard scan expansion
                log.debug(
                    f"Job '{job.name}' expanded to {len(scan_expansion)} jobs "
                    f"using {method} expansion"
                )

                if numbering_enabled:
                    total_count = len(scan_expansion)
                    for idx, new_job in enumerate(scan_expansion, start=1):
                        new_job = self._apply_job_numbering(
                            new_job, job.name, idx, total_count
                        )
                        final_expansion.append(new_job)
                else:
                    final_expansion.extend(scan_expansion)
            else:
                # No expansion
                final_expansion.append(scan_expansion[0])

            # Register expansion in map and registry
            expansion_map[job.name] = [j.name for j in final_expansion]
            for j in final_expansion:
                job_registry[j.name] = j
            expanded_jobs.extend(final_expansion)

        return expanded_jobs

    def _apply_job_numbering(
        self, job: JobConfig, base_name: str, index: int, total: int
    ) -> JobConfig:
        """Apply numbering to a job name and output.

        Parameters
        ----------
        job : JobConfig
            Job configuration to modify
        base_name : str
            Base name to which numbering will be appended
        index : int
            Index number (1-based)
        total : int
            Total number of jobs in this expansion group (used for padding)

        Returns
        -------
        JobConfig
            Modified job configuration with numbered name

        """
        # Create a copy to avoid modifying the original
        job = copy.deepcopy(job)

        # Determine padding width
        # e.g. total=10 -> width=2, total=100 -> width=3
        width = len(str(total))
        # Ensure minimum width of 3 for consistency with previous behavior if desired,
        # or just use dynamic width. User asked for >100 support.
        width = max(3, width)

        suffix = f"{index:0{width}d}"

        # Update job name
        job.name = f"{base_name}_{suffix}"

        # Update output name if specified
        if job.output:
            job.output = f"{job.output}_{suffix}"

        return job

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

            # Check if input matches a job name
            if job.input in jobs_by_name:
                # Valid job reference
                continue

            # Check if input matches an engine name
            upstream_jobs = jobs_by_engine.get(job.input, [])
            if not upstream_jobs:
                # Not a job name or engine name - could be a file path
                # This is valid (external input)
                log.debug(
                    f"Job '{job.name}' input '{job.input}' appears to be external"
                )
                continue

            # It's an engine name - check for ambiguity
            if len(upstream_jobs) > 1:
                job_names = ", ".join([j.name for j in upstream_jobs])
                raise QPhaseConfigError(
                    f"Job '{job.name}' input '{job.input}' is ambiguous. "
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
                input_job=job.input,
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
    on_progress: Callable[[JobProgressUpdate], None] | None = None,
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
    on_progress : Callable[[JobProgressUpdate], None] | None, optional
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
