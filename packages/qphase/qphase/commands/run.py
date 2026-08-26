"""Execute one project workflow."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from qphase.core.errors import QPhaseError, configure_logging, get_logger
from qphase.core.project import ProjectContext
from qphase.core.registry import discovery
from qphase.core.system_config import load_system_config
from qphase.service import SchedulerService
from qphase.service.models import ExecutionPlan

from .progress import CliProgressRenderer, ProgressLogHandler

log = get_logger()
_RESUME_FROM_OPTION = typer.Option(None, "--resume-from")
_TAG_OPTION = typer.Option(
    [],
    "--tag",
    help="Submission tag recorded in the session manifest (repeatable)",
)


def run_command(
    workflow_reference: str = typer.Argument(
        ..., help="Stable workflow ID or path relative to configs/workflows"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    log_file: str | None = typer.Option(None, help="Write full logs to this file"),
    log_json: bool = typer.Option(False, help="Write file logs as JSON"),
    suppress_warnings: bool = typer.Option(False),
    dry_run: bool = typer.Option(False, "--dry-run"),
    show_plan: bool = typer.Option(False, "--plan"),
    resume_from: Path | None = _RESUME_FROM_OPTION,
    tag: list[str] = _TAG_OPTION,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Plan or execute one versioned workflow in the current project."""
    try:
        project = ProjectContext.discover()
        system = load_system_config()
        progress = system.reporting.progress
        logging_config = system.reporting.logging
        renderer = (
            None
            if json_output
            else CliProgressRenderer(
                verbose=verbose,
                refresh_interval=progress.refresh_interval,
                milestone_percent=progress.non_tty_milestone_percent,
            )
        )
        configure_logging(
            verbose=verbose,
            log_file=log_file,
            as_json=log_json or logging_config.format == "json",
            suppress_warnings=suppress_warnings or not logging_config.capture_warnings,
            console_level=logging_config.console_level,
            file_level=logging_config.file_level,
            console_handler=logging.NullHandler()
            if renderer is None
            else ProgressLogHandler(renderer),
        )
        discovery.discover_plugins()
        discovery.discover_local_plugins(project)
        service = SchedulerService(system, project=project)
        workflow = service.load_workflow(workflow_reference)
        if show_plan or dry_run:
            plan = service.build_plan(workflow)
            typer.echo(
                json.dumps(plan.model_dump(mode="json"), indent=2)
                if json_output
                else _format_execution_plan(workflow.id, plan)
            )
            return
        results = service.run(
            workflow,
            progress_callback=None if renderer is None else renderer.handle,
            resume_from=resume_from,
            submission_tags=tag or None,
        )
        success = sum(result.success for result in results)
        failed = sum(result.status == "failed" for result in results)
        skipped = sum(result.status == "skipped_dependency" for result in results)
        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "workflow_id": workflow.id,
                        "session_id": service.last_session_handle.session_id
                        if service.last_session_handle
                        else None,
                        "session_dir": str(service.last_session_handle.session_dir)
                        if service.last_session_handle
                        and service.last_session_handle.session_dir
                        else None,
                        "completed": success,
                        "failed": failed,
                        "skipped": skipped,
                    },
                    indent=2,
                )
            )
        else:
            typer.echo(
                f"\nWorkflow {workflow.id}: {success} completed, "
                f"{failed} failed, {skipped} skipped"
            )
            if service.last_session_handle:
                typer.echo(f"Session: {service.last_session_handle.session_dir}")
        if failed or skipped:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except QPhaseError as exc:
        typer.echo(
            json.dumps({"status": "failed", "error": str(exc)})
            if json_output
            else f"Error: {exc}"
        )
        raise typer.Exit(code=1) from exc


def _format_execution_plan(workflow_id: str, plan: ExecutionPlan) -> str:
    lines = [f"Workflow: {workflow_id}", f"Logical jobs: {len(plan.jobs)}", "", "Jobs:"]
    for job in plan.jobs:
        lines.append(
            f"  - {job.name} engine={job.engine} "
            f"output={job.expected_output_name or '<not saved>'}"
        )
        if job.scan_summary:
            lines.append(
                f"    scan: shape={tuple(job.scan_summary['shape'])}, "
                f"points={job.scan_summary['size']}"
            )
    if plan.edges:
        lines.extend(["", "Edges:"])
        lines.extend(
            f"  - {edge.source} -> {edge.target} ({edge.kind})" for edge in plan.edges
        )
    if plan.validation_issues:
        lines.extend(["", "Validation issues:"])
        lines.extend(
            f"  - {item.path}: {item.message}" for item in plan.validation_issues
        )
    return "\n".join(lines)
