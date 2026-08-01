"""qphase: Job Execution CLI Commands
---------------------------------------------------------
Implements the ``qphase run`` command group, serving as the primary execution entry
point. It includes the ``jobs`` command for running simulations defined in YAML/JSON
files, handling path resolution and scheduler invocation, and the ``list`` command
for displaying available engine packages that can be used in job configurations.

Public API
----------
`list` : List available engine packages with descriptions.
`jobs` : Execute job configurations from YAML/JSON files.
"""

import json
import logging
import sys
from pathlib import Path

import typer

from qphase.core.config_loader import (
    _find_job_config,
    load_jobs_from_files,
)
from qphase.core.errors import (
    QPhaseError,
    configure_logging,
    get_logger,
)
from qphase.core.registry import discovery, registry
from qphase.core.system_config import load_system_config
from qphase.service import SchedulerService
from qphase.service.models import ExecutionPlan

from .progress import CliProgressRenderer, ProgressLogHandler

app = typer.Typer()
log = get_logger()

# Module-level singleton for typer.Argument to avoid function call in default (B008)
JOB_NAMES_ARG = typer.Argument(
    default_factory=list,
    help="Name(s) of the job(s) to run (searched in configs/jobs/ directory)",
)

_RESUME_FROM_OPT = typer.Option(
    None, "--resume-from", help="Resume from a previous session directory"
)


def _list_engines():
    """List available engine packages."""
    # Ensure plugins are discovered
    discovery.discover_plugins()
    discovery.discover_local_plugins()

    # Get all engine plugins
    engines = registry.list(namespace="engine")

    if not engines:
        typer.echo("No engine packages found.")
        return

    typer.echo("Available Engines:")
    for engine_name in sorted(engines.keys()):
        typer.echo(f"  - {engine_name}")

    typer.echo(f"\nTotal: {len(engines)} engine package(s)")


@app.command()
def jobs(
    job_names: list[str] = JOB_NAMES_ARG,
    list_jobs: bool = typer.Option(
        False, "--list", help="List available jobs and exit"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed progress information"
    ),
    log_file: str | None = typer.Option(None, help="Write logs to file path"),
    log_json: bool = typer.Option(False, help="Write file logs in JSON format"),
    suppress_warnings: bool = typer.Option(False, help="Suppress warnings output"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Build the execution plan without running jobs"
    ),
    show_plan: bool = typer.Option(
        False, "--plan", help="Print the execution plan and exit"
    ),
    resume_from: Path | None = _RESUME_FROM_OPT,
    json_output: bool = typer.Option(
        False, "--json", help="Print machine-readable JSON output"
    ),
):
    """Run SDE simulation jobs by name from configs/jobs/ directory.

    JOB_NAMES should be the name(s) of job configuration file(s) (without extension)
    located in the configs/jobs/ directory. The command will automatically search
    for .yaml or .yml files with that name.

    Job file format (in configs/jobs):
        name: job_name
        engine:
          sde:
            t0: 0.0
            t1: 10.0
            dt: 0.01
            n_traj: 16
            seed: 42
            ic:
              - ["1.0+0.0j"]
        backend:
          numpy:
            float_dtype: float64
        integrator:
          euler_maruyama: {}
        model:
          kerr_2mode:
            omega_a: 1.0
            omega_b: 1.0
            chi: 0.01
            gamma_a: 0.1
            gamma_b: 0.1
            g: 0.1

    Examples
    --------
        qphase run my_simulation
        qphase run job1 job2
        qphase run --list
        qphase run --verbose my_job
        qphase run my_job --plan
        qphase run my_job --plan --json
        qphase run my_job --dry-run

    """
    # Handle "list" argument as a command to list engines
    if "list" in job_names:
        _list_engines()
        return

    try:
        system_cfg = load_system_config()
        progress_cfg = system_cfg.reporting.progress
        logging_cfg = system_cfg.reporting.logging
        renderer = None
        console_handler: logging.Handler
        if json_output:
            console_handler = logging.NullHandler()
        else:
            renderer = CliProgressRenderer(
                verbose=verbose,
                refresh_interval=progress_cfg.refresh_interval,
                milestone_percent=progress_cfg.non_tty_milestone_percent,
            )
            console_handler = ProgressLogHandler(renderer)
        configure_logging(
            verbose=verbose,
            log_file=log_file,
            as_json=log_json or logging_cfg.format == "json",
            suppress_warnings=suppress_warnings or not logging_cfg.capture_warnings,
            console_level=logging_cfg.console_level,
            file_level=logging_cfg.file_level,
            console_handler=console_handler,
        )

        if not list_jobs and not job_names:
            raise QPhaseError("No job names provided. Use --list to list jobs.")

        # Ensure plugins are discovered
        discovery.discover_plugins()
        discovery.discover_local_plugins()

        scheduler_service = SchedulerService(system_cfg)

        # Handle --list option
        if list_jobs:
            available_jobs = scheduler_service.list_jobs()
            if not available_jobs:
                typer.echo("No jobs found in configs/jobs/ directory.")
            else:
                typer.echo("\nAvailable jobs:")
                for job in available_jobs:
                    typer.echo(f"  - {job}")
                typer.echo(f"\nTotal: {len(available_jobs)} job(s)")
            return

        # Find job configuration files
        cfg_paths = []
        for job_name in job_names:
            cfg_path = _find_job_config(system_cfg.paths.config_dirs, job_name)

            if cfg_path is None or not cfg_path.exists():
                log.error(f"Job '{job_name}' not found in configs/jobs/ directories")
                log.error(f"Searched in: {system_cfg.paths.config_dirs}")
                available_jobs = scheduler_service.list_jobs()
                if available_jobs:
                    log.error(f"Available jobs: {', '.join(available_jobs)}")
                raise typer.Exit(code=1)

            log.debug(f"Found job configuration: {cfg_path}")
            cfg_paths.append(cfg_path)

        # Add config directories to Python path for model imports
        added_paths = set()
        for config_path in cfg_paths:
            for cand in (config_path.parent, config_path.parent.parent):
                if cand.exists():
                    pstr = str(cand)
                    if pstr not in sys.path and pstr not in added_paths:
                        sys.path.insert(0, pstr)
                        added_paths.add(pstr)

        # Load JobList from YAML files
        log.debug(f"Loading {len(cfg_paths)} configuration file(s)")
        job_list = load_jobs_from_files(cfg_paths)

        log.debug(f"Loaded {len(job_list.jobs)} jobs")

        if show_plan or dry_run:
            plan_obj = scheduler_service.build_plan(job_list)
            if json_output:
                typer.echo(json.dumps(plan_obj.model_dump(mode="json"), indent=2))
            else:
                typer.echo(_format_execution_plan(plan_obj))
            return

        progress_callback = None if renderer is None else renderer.handle

        # Execute jobs
        log.debug("Starting job execution")
        results = scheduler_service.run(
            job_list,
            progress_callback=progress_callback,
            resume_from=resume_from,
        )

        # Report results
        success_count = sum(1 for r in results if r.success)
        skipped_count = sum(1 for r in results if r.status == "skipped_dependency")
        failed_count = sum(1 for r in results if r.status == "failed")
        total_count = len(results)

        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "success_count": success_count,
                        "failed_count": failed_count,
                        "skipped_count": skipped_count,
                        "total_count": total_count,
                        "session_dir": (
                            str(scheduler_service.last_run_handle.session_dir)
                            if scheduler_service.last_run_handle
                            and scheduler_service.last_run_handle.session_dir
                            else None
                        ),
                        "results": [
                            {
                                "job_index": result.job_index,
                                "job_name": result.job_name,
                                "run_dir": str(result.run_dir),
                                "run_id": result.run_id,
                                "success": result.success,
                                "status": result.status,
                                "error_summary": result.error_summary,
                                "error_id": result.error_id,
                                "error_code": result.error_code,
                                "error_report_path": result.error_report_path,
                            }
                            for result in results
                        ],
                    },
                    indent=2,
                )
            )
            if failed_count or skipped_count:
                raise typer.Exit(code=1)
            return

        typer.echo(
            f"\nSummary: {success_count} completed, {failed_count} failed, "
            f"{skipped_count} skipped"
        )
        if scheduler_service.last_run_handle is not None:
            typer.echo(f"Session: {scheduler_service.last_run_handle.session_dir}")
        if failed_count or skipped_count:
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except QPhaseError as e:
        if json_output:
            typer.echo(json.dumps({"status": "failed", "error": str(e)}))
        else:
            log.error(str(e))
        raise typer.Exit(code=1) from e
    except Exception as e:
        if json_output:
            typer.echo(
                json.dumps({"status": "failed", "error": f"Unexpected error: {e}"})
            )
        else:
            log.error(f"Unexpected error: {e}")
        raise typer.Exit(code=1) from e


def _format_execution_plan(plan: ExecutionPlan) -> str:
    """Format an execution plan for terminal output."""
    lines = [
        "Execution plan:",
        f"  Logical jobs: {len(plan.jobs)}",
    ]

    if plan.validation_issues:
        lines.append("\nValidation issues:")
        for issue in plan.validation_issues:
            location = f" ({issue.path})" if issue.path else ""
            lines.append(f"  - {issue.level}{location}: {issue.message}")

    lines.append("\nJobs:")
    for job in plan.jobs:
        output = job.expected_output_name or "<not saved>"
        lines.append(f"  - {job.name} engine={job.engine} output={output}")
        if job.scan_summary:
            lines.append(
                f"    scan: shape={tuple(job.scan_summary['shape'])}, "
                f"points={job.scan_summary['size']}"
            )
        if job.required_plugins:
            lines.append(f"    required: {', '.join(job.required_plugins)}")
        if job.optional_plugins_enabled:
            lines.append(f"    optional: {', '.join(job.optional_plugins_enabled)}")

    if plan.edges:
        lines.append("\nEdges:")
        for edge in plan.edges:
            lines.append(f"  - {edge.source} -> {edge.target} ({edge.kind})")

    return "\n".join(lines)


@app.command(name="list")
def list_engines():
    """List available engine packages that can be used in job configurations."""
    _list_engines()


run_command = jobs
