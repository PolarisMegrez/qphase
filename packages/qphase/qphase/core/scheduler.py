"""qphase: Job Scheduler
---------------------------------------------------------
Orchestrates the execution of simulation jobs, managing the complete lifecycle from
dependency resolution to result persistence. The Scheduler handles serial execution
of ``JobList`` items, expands parameter scans into multiple tasks, manages run
directory creation, and provides hooks for progress reporting and snapshot generation.

Public API
----------
``Scheduler`` : Main class for job execution and lifecycle management
``JobResult`` : Dataclass containing job execution results and metadata
``JobProgressUpdate`` : Dataclass for progress callback information
``run_jobs`` : Convenience function to execute a JobList

Notes
-----
- Jobs execute serially in definition order with automatic dependency resolution
- Parameter scanning expands jobs with scanable fields into multiple executions
- Each job run creates a timestamped directory with configuration snapshot

"""

from __future__ import annotations

import copy
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

    def run(self, job_list: JobList) -> list[JobResult]:
        """Execute all jobs in the job list serially.

        Parameters
        ----------
        job_list : JobList
            List of jobs to execute

        Returns
        -------
        list[JobResult]
            Results for each executed job, in order

        """
        # Step 1: Validate jobs before execution
        self._validate_jobs(job_list)

        # Step 2: Expand parameter scan jobs
        expanded_jobs = self._expand_parameter_scans(job_list)

        results: list[JobResult] = []
        job_results: dict[str, ResultProtocol] = {}

        for job_idx, job in enumerate(expanded_jobs):
            try:
                input_result = self._resolve_input(job, job_results)
                job_result, output_result = self._run_job(
                    job, job_idx, len(expanded_jobs), input_result
                )
                results.append(job_result)

                # Handle output based on job config
                if job_result.success:
                    self._handle_job_output(
                        job, output_result, job_results, job_result.run_dir
                    )

            except Exception as e:
                log.error(f"Job {job.name} failed: {e}")
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

        return results

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
        output_result : ResultBase
            Result object from the job
        job_results : dict[str, ResultBase]
            Storage for job results that will be passed to downstream jobs
        run_dir : Path
            Run directory for this job (where results should be saved)

        Raises
        ------
        QPhaseConfigError
            If output references a non-existent downstream job

        """
        # Determine the output destination
        output_dest = job.output if job.output else job.name

        # Store result for downstream jobs
        # We store by job name so downstream jobs can reference it
        job_results[job.name] = output_result

        # If output is explicitly set, we might also want to store it under that name
        # (though usually output refers to filename or downstream job name)
        if job.output:
            job_results[job.output] = output_result

        # Save to disk if auto_save_results is enabled
        if self.system_config.auto_save_results:
            # Build save path: run_dir / output_filename
            # Note: filename should not include extension -
            # ResultBase.save() will add appropriate extension
            save_path = run_dir / output_dest

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

        # Check if input is from a previous job
        if job.input in job_results:
            return job_results[job.input]

        # Check if input is an external file
        input_path = Path(job.input)
        if input_path.exists():
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

    def _run_job(
        self,
        job: JobConfig,
        job_idx: int,
        job_total: int,
        input_result: ResultProtocol | None,
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
            Index of this job in the job list (0-based)
        job_total : int
            Total number of jobs being executed
        input_result : ResultProtocol | None
            Input data from upstream job, or None

        Returns
        -------
        tuple[JobResult, ResultBase]
            Tuple of (job execution metadata, engine output result)

        Raises
        ------
        QPhasePluginError
            If engine instantiation fails
        QPhaseRuntimeError
            If engine execution fails

        """
        run_id = self._generate_run_id()
        run_dir = self._create_run_dir(job, run_id)

        system_cfg = job.system if job.system is not None else self.system_config

        # Merge global config with job config
        job_override = {
            "plugins": job.plugins,
            "engine": job.engine,
            "params": job.params,
        }

        merged_config = get_config_for_job(
            system_cfg, job_name=job.name, job_config_dict=job_override
        )

        # Build plugins (backend, integrator, state, etc.)
        plugins = self._build_plugins(merged_config.get("plugins", {}))

        # Extract engine name and config
        engine_config_dict = merged_config.get("engine", {})
        if engine_config_dict:
            engine_name = list(engine_config_dict.keys())[0]
            engine_config_raw = engine_config_dict[engine_name].copy()
            engine_config_raw["name"] = engine_name
        else:
            # Fallback to job's engine config
            engine_name = job.get_engine_name()
            engine_config_raw = job.engine.get(engine_name, {}).copy()
            engine_config_raw["name"] = engine_name

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
                    total_jobs=job_total,
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
                    # Heuristic: if total_jobs > 1, assume homogeneous expansion
                    global_eta = None
                    if (
                        job_total > 1
                        and job_eta is not None
                        and total_duration_estimate is not None
                    ):
                        # Simple extrapolation: remaining jobs * current job total
                        # duration
                        remaining_jobs = job_total - job_idx
                        global_eta = job_eta + (
                            remaining_jobs * total_duration_estimate
                        )

                    if self.on_progress is not None:
                        self.on_progress(
                            JobProgressUpdate(
                                job_name=job.name,
                                job_index=job_idx,
                                total_jobs=job_total,
                                message=message,
                                percent=percent,
                                job_eta=job_eta,
                                global_eta=global_eta,
                                stage=stage,
                            )
                        )

                progress_cb = _on_engine_progress

            # Execute engine
            # Pass input result's data to engine
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

            # Ensure output is a ResultBase object
            if not isinstance(output_result, ResultProtocol):
                raise QPhaseRuntimeError(
                    f"Engine '{job.get_engine_name()}' did not return a "
                    f"ResultBase object. "
                    f"All engines must return a ResultBase instance from "
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

        # Stage 2: Validate data flow
        self._validate_data_flow(job_list)

        log.info("Job validation completed successfully")

    def _expand_parameter_scans(self, job_list: JobList) -> list[JobConfig]:
        """Expand jobs with scanable list parameters into multiple jobs.

        This method scans job configurations for parameters marked as 'scanable'
        and expands them into multiple jobs based on the configured expansion method.

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

        for job in job_list.jobs:
            new_jobs = expander.expand(job, method=method)

            if len(new_jobs) > 1:
                log.debug(
                    f"Job '{job.name}' expanded to {len(new_jobs)} jobs "
                    f"using {method} expansion"
                )

                # Apply numbering to outputs if enabled
                if numbering_enabled:
                    for idx, new_job in enumerate(new_jobs, start=1):
                        new_job = self._apply_job_numbering(new_job, job.name, idx)
                        expanded_jobs.append(new_job)
                else:
                    expanded_jobs.extend(new_jobs)
            else:
                expanded_jobs.extend(new_jobs)

        return expanded_jobs

    def _apply_job_numbering(
        self, job: JobConfig, base_name: str, index: int
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

        Returns
        -------
        JobConfig
            Modified job configuration with numbered name

        """
        # Create a copy to avoid modifying the original
        job = copy.deepcopy(job)

        # Update job name
        job.name = f"{base_name}_{index:03d}"

        # Update output name if specified
        if job.output:
            job.output = f"{job.output}_{index:03d}"

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
                        plugins[f"{plugin_type}.{plugin_name}"] = instance
                        plugins[plugin_type] = instance
                    except Exception as e:
                        raise QPhasePluginError(
                            f"Failed to create plugin "
                            f"'{plugin_type}.{plugin_name}': {e}"
                        ) from e
            else:
                raise QPhasePluginError(f"Invalid plugin config for '{plugin_type}'")

        return plugins

    def _generate_run_id(self) -> str:
        """Generate a unique run ID with timestamp and UUID suffix."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        return f"{ts}_{uuid.uuid4().hex[:8]}"

    def _create_run_dir(self, job: JobConfig, run_id: str) -> Path:
        """Create and return the run directory for a job."""
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
