"""Single logical-Job execution boundary for the core scheduler."""

from __future__ import annotations

import inspect
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifacts import ArtifactStore
from .compiler import CompiledJob
from .config import JobConfig
from .dataset import DatasetResultProtocol, MappedDatasetResult, iter_dataset_views
from .error_report import build_error_report, save_error_report
from .errors import (
    ErrorCode,
    QPhaseConfigError,
    QPhasePluginError,
    QPhaseRuntimeError,
    QPhaseSchedulerError,
)
from .execution import (
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
from .registry import RegistryCenter

if TYPE_CHECKING:
    from .scheduler import Scheduler

log = logging.getLogger("qphase")

__all__ = ["JobResult", "_JobOutcome", "run_job"]


@dataclass
class JobResult:
    """Result of one logical Job execution."""

    job_index: int
    job_name: str
    job_dir: Path
    success: bool
    status: str = "completed"
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
    """Internal result before the scheduler routes the output."""

    result: JobResult
    output: ResultProtocol | None
    context: ExecutionContext | None


def scan_summary(job: JobConfig) -> dict[str, Any] | None:
    """Return the compact scan descriptor used by progress and errors."""
    if job.scan is None:
        return None
    return job.scan.compile().summary()


def configured_plugin_paths(job: JobConfig) -> list[str]:
    """Return configured plugin paths for job metadata."""
    return [
        f"{namespace}.{name}"
        for namespace, entries in job.plugins.items()
        for name in entries
    ]


def create_job_dir(scheduler: Scheduler, job: JobConfig) -> Path:
    """Create the artifact directory of one logical Job."""
    if scheduler.session_dir is None:
        raise QPhaseSchedulerError(
            f"cannot create Job directory for '{job.name}' without a Session"
        )
    job_dir = scheduler.session_dir / job.name
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def make_tracker(system_config: Any) -> ProgressTracker:
    """Create a progress tracker from the current reporting policy."""
    cfg = system_config.reporting.progress
    return ProgressTracker(
        eta_warmup_seconds=float(cfg.eta_warmup_seconds),
        eta_min_samples=int(cfg.eta_min_samples),
        eta_smoothing=float(cfg.eta_smoothing),
    )


def build_plugins(
    registry: RegistryCenter, config: dict[str, Any]
) -> dict[str, Any]:
    """Instantiate the resolved plugin selection from one registry source."""
    plugins: dict[str, Any] = {}
    for plugin_type, config_data in config.items():
        if not config_data:
            continue
        if isinstance(config_data, dict) and "name" in config_data:
            try:
                plugins[plugin_type] = registry.create_plugin_instance(
                    plugin_type, config_data
                )
            except Exception as exc:
                raise QPhasePluginError(
                    f"Failed to create plugin '{plugin_type}': {exc}"
                ) from exc
            continue
        if not isinstance(config_data, dict):
            raise QPhasePluginError(f"Invalid plugin config for '{plugin_type}'")
        type_instances: dict[str, Any] = {}
        for plugin_name, plugin_config in config_data.items():
            if not isinstance(plugin_config, dict):
                continue
            flat_config = dict(plugin_config)
            flat_config["name"] = plugin_name
            try:
                instance = registry.create_plugin_instance(
                    plugin_type, flat_config
                )
            except Exception as exc:
                raise QPhasePluginError(
                    f"Failed to create plugin '{plugin_type}.{plugin_name}': {exc}"
                ) from exc
            plugins[f"{plugin_type}.{plugin_name}"] = instance
            type_instances[plugin_name] = instance
        plugins[plugin_type] = (
            next(iter(type_instances.values()))
            if len(type_instances) == 1
            else type_instances
        )
    return plugins


def invoke_engine(
    engine: Any,
    data: Any,
    context: ExecutionContext,
    progress_cb: Any | None,
) -> ResultProtocol:
    """Invoke an engine using only the arguments it declares."""
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


def write_snapshot(
    scheduler: Scheduler,
    job_dir: Path,
    job: JobConfig,
    compiled_job: CompiledJob,
    job_idx: int,
) -> None:
    """Persist the resolved Job snapshot; failures remain fatal."""
    from .snapshot import SnapshotManager

    manager = SnapshotManager(scheduler.project.session_root)
    snapshot = manager.create_snapshot(
        job=job,
        job_index=job_idx,
        system_config=scheduler.system_config,
        validated_plugins=dict(compiled_job.plugin_config),
        engine_config=dict(compiled_job.engine_config),
        session_id=scheduler.session_id,
        job_dir=job_dir,
        input_job=compiled_job.input_source,
        output_job=compiled_job.output,
        metadata={
            "scheduler_version": "2.0",
            "snapshot_created_by": "scheduler",
        },
    )
    path = manager.save_snapshot(snapshot, job_dir)
    log.debug("Snapshot saved to %s", path)


def fail_job(
    scheduler: Scheduler,
    job: JobConfig,
    job_idx: int,
    job_total: int,
    exc: BaseException,
    *,
    job_dir: Path | None,
    stage: str | None = None,
    plugin: str | None = None,
) -> JobResult:
    """Build and emit one typed Job failure report."""
    report = build_error_report(
        exc,
        session_id=scheduler.session_id,
        job_name=job.name,
        engine=job.get_engine_name(),
        stage=stage,
        plugin=plugin,
        job_dir=job_dir,
        scan_context=scan_summary(job),
        log_file=scheduler._session_log_path,
    )
    target_dir = (
        job_dir
        if job_dir is not None
        else scheduler.session_dir / job.name
        if scheduler.session_dir is not None
        else Path(".")
    )
    report_path = save_error_report(report, target_dir)
    log.exception("Job '%s' failed [%s]: %s", job.name, report.code, report.summary)
    summary = report.summary_dto(
        report_path=str(report_path) if report_path is not None else None
    )
    scheduler._emit_snapshot(
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


def run_job(
    scheduler: Scheduler,
    job: JobConfig,
    job_idx: int,
    job_total: int,
    input_result: ResultProtocol | None,
    *,
    compiled_job: CompiledJob,
    display_total: int | None = None,
) -> _JobOutcome:
    """Execute one compiled Job and return its output/context boundary."""
    display_total = job_total if display_total is None else display_total
    job_dir = create_job_dir(scheduler, job)
    engine_name = job.get_engine_name()
    tracker = make_tracker(scheduler.system_config)
    clock_start = time.monotonic()

    with bind_log_context(
        session_id=scheduler.session_id, job=job.name, engine=engine_name
    ):
        try:
            return _run_job_inner(
                scheduler,
                job,
                job_idx,
                display_total,
                input_result,
                compiled_job=compiled_job,
                job_dir=job_dir,
                engine_name=engine_name,
                tracker=tracker,
                clock_start=clock_start,
            )
        except Exception as exc:
            if getattr(exc, "code", None) == ErrorCode.CANCELLATION:
                scheduler._emit_snapshot(
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
            result = fail_job(
                scheduler,
                job,
                job_idx,
                display_total,
                exc,
                job_dir=job_dir,
                stage=tracker.current_stage,
            )
            return _JobOutcome(result=result, output=None, context=None)


def _run_job_inner(
    scheduler: Scheduler,
    job: JobConfig,
    job_idx: int,
    display_total: int,
    input_result: ResultProtocol | None,
    *,
    compiled_job: CompiledJob,
    job_dir: Path,
    engine_name: str,
    tracker: ProgressTracker,
    clock_start: float,
) -> _JobOutcome:
    """Construct plugins/context, invoke the engine, and collect its output."""
    merged_config = dict(compiled_job.merged_config)
    final_plugins_cfg = dict(compiled_job.plugin_config)
    engine_name = compiled_job.engine_name

    plugins = build_plugins(scheduler._registry, final_plugins_cfg)
    engine_config_dict = merged_config.get("engine", {})
    if (
        not isinstance(engine_config_dict, dict)
        or engine_name not in engine_config_dict
    ):
        raise QPhaseConfigError(
            f"merged config has no engine config for {engine_name!r}",
            code=ErrorCode.CONFIG,
        )
    engine_config_raw = dict(engine_config_dict[engine_name])
    engine_config_raw["name"] = engine_name
    engine_config_raw["output_dir"] = str(job_dir)

    try:
        engine = scheduler._registry.create_plugin_instance(
            "engine", engine_config_raw, plugins=plugins
        )
    except Exception as exc:
        raise QPhasePluginError(
            f"Failed to instantiate engine '{engine_name}': {exc}",
            code=ErrorCode.PLUGIN_CREATION,
            hint="Check the engine configuration against its schema.",
            context={"engine": engine_name},
        ) from exc

    write_snapshot(scheduler, job_dir, job, compiled_job, job_idx)
    on_progress = scheduler.on_progress

    def _sink(event: ProgressEvent) -> None:
        observed = tracker.observe(event)
        if observed.stage:
            set_log_context(stage=observed.stage)
        if on_progress is None:
            return
        fraction, rate, remaining = tracker.estimates(observed)
        on_progress(
            ProgressSnapshot(
                kind="job_status" if observed.kind == "status" else "job_progress",
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
    effective_system = job.merge_with_system_config(scheduler.system_config)
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
    from qphase.data.resolver import ProjectArtifactResolver

    context = ExecutionContext(
        parameter_grid=compiled_job.parameter_grid,
        resources=ResourceSnapshot.from_system_config(
            effective_system, backend=backend
        ),
        progress=reporter,
        cancellation=scheduler.cancellation.token_for(job.name),
        artifacts=ArtifactStore(job_dir, effective_system.scan_runtime),
        checkpoints=CheckpointStore(job_dir, checkpoint_config, fingerprint),
        job_dir=job_dir,
        artifact_resolver=ProjectArtifactResolver(scheduler.project),
        metadata={
            "job_name": job.name,
            "scan_summary": scan_summary(job),
            "configured_plugins": configured_plugin_paths(job),
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
                mapped[label] = invoke_engine(
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
        output_result = invoke_engine(
            engine, input_result, context, legacy_cb
        )

    if not isinstance(output_result, ResultProtocol):
        raise QPhaseRuntimeError(
            f"Engine '{engine_name}' did not return a ResultProtocol object. "
            "All engines must return a ResultProtocol instance from their run() method."
        )

    duration = time.monotonic() - clock_start
    scheduler._emit_snapshot(
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
    if scheduler.on_job_dir is not None:
        scheduler.on_job_dir(job_dir)
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
