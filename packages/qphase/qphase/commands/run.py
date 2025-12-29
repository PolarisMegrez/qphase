"""qphase: Job Execution CLI Commands
---------------------------------------------------------
Implements the ``qps run`` command group, serving as the primary execution entry
point. It includes the ``jobs`` command for running simulations defined in YAML/JSON
files, handling path resolution and scheduler invocation, and the ``list`` command
for displaying available engine packages that can be used in job configurations.

Public API
----------
``list`` : List available engine packages with descriptions
``jobs`` : Execute job configurations from YAML/JSON files
"""

import sys
from pathlib import Path
from typing import cast

import typer

from qphase.core import JobProgressUpdate, Scheduler
from qphase.core.config_loader import (
    _find_job_config,
    list_available_jobs,
    load_jobs_from_files,
)
from qphase.core.errors import (
    QPhaseError,
    configure_logging,
    get_logger,
)
from qphase.core.registry import discovery, registry
from qphase.core.system_config import load_system_config

app = typer.Typer()


@app.command()
def jobs(
    job_name: str = typer.Argument(
        ...,
        help="Name of the job to run (searched in configs/jobs/ directory)",
    ),
    list_jobs: bool = typer.Option(
        False, "--list", help="List available jobs and exit"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    log_file: str | None = typer.Option(None, help="Write logs to file path"),
    log_json: bool = typer.Option(False, help="Log in JSON format"),
    suppress_warnings: bool = typer.Option(False, help="Suppress warnings output"),
):
    """Run SDE simulation jobs by name from configs/jobs/ directory.

    JOBS_NAME should be the name of a job configuration file (without extension)
    located in the configs/jobs/ directory. The command will automatically search
    for .yaml or .yml files with that name.

    Job file format (in configs/jobs/):
        name: job_name
        engine: sde
        plugins:
          backend:
            numpy:
              float_dtype: float64
          model:
            vdp_two_mode:
              D: 1.0
        engine:
          sde:
            t_end: 10.0

    Examples
    --------
        qps run jobs my_simulation
        qps run jobs --list
        qps run jobs --verbose my_job

    """
    # Configure logging
    configure_logging(
        verbose=verbose,
        log_file=log_file,
        as_json=log_json,
        suppress_warnings=suppress_warnings,
    )
    log = get_logger()

    try:
        # Ensure plugins are discovered
        discovery.discover_plugins()
        discovery.discover_local_plugins()

        # Load system configuration to get config directories
        system_cfg = load_system_config()

        # Handle --list option
        if list_jobs:
            available_jobs = list_available_jobs(system_cfg)
            if not available_jobs:
                typer.echo("No jobs found in configs/jobs/ directory.")
            else:
                typer.echo("\nAvailable jobs:")
                for job in available_jobs:
                    typer.echo(f"  - {job}")
                typer.echo(f"\nTotal: {len(available_jobs)} job(s)")
            return

        # Find job configuration file
        cfg_path = _find_job_config(system_cfg.paths.config_dirs, job_name)

        if cfg_path is None or not cfg_path.exists():
            log.error(f"Job '{job_name}' not found in configs/jobs/ directories")
            log.error(f"Searched in: {system_cfg.paths.config_dirs}")
            available_jobs = list_available_jobs(system_cfg)
            if available_jobs:
                log.error(f"Available jobs: {', '.join(available_jobs)}")
            raise typer.Exit(code=1)

        log.info(f"Found job configuration: {cfg_path}")

        # Add config directories to Python path for model imports
        for config_path in [cfg_path]:
            for cand in (config_path.parent, config_path.parent.parent):
                if cand.exists():
                    pstr = str(cand)
                    if pstr not in sys.path:
                        sys.path.insert(0, pstr)

        # Load JobList from YAML files
        log.info("Loading 1 configuration file(s)")
        job_list = load_jobs_from_files([cfg_path])

        log.info(f"Loaded {len(job_list.jobs)} jobs")

        # Load system configuration
        system_cfg = load_system_config()

        # Create scheduler
        scheduler = Scheduler(
            system_config=system_cfg,
            on_progress=_make_progress_callback(),
            on_run_dir=_make_run_dir_callback(),
        )

        # Execute jobs
        log.info("Starting job execution")
        results = scheduler.run(job_list)

        # Report results
        success_count = sum(1 for r in results if r.success)
        total_count = len(results)

        if success_count == total_count:
            log.info(f"All {total_count} jobs completed successfully")
        else:
            failed = total_count - success_count
            log.warning(
                f"{success_count}/{total_count} jobs succeeded ({failed} failed)"
            )

        # Print run directories
        typer.echo("\nRun directories:")
        for result in results:
            if result.success:
                typer.echo(f"  [{result.job_name}] {result.run_dir}")
            else:
                typer.echo(f"  [{result.job_name}] FAILED: {result.error}")

    except QPhaseError as e:
        log.error(str(e))
        raise typer.Exit(code=1) from e
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        raise typer.Exit(code=1) from e


def _make_progress_callback():
    """Create a progress callback for the scheduler."""

    def _on_progress(update: JobProgressUpdate):
        # Format total duration estimate (total estimated time including elapsed)
        total_est = update.global_eta
        est_ok = total_est is not None and total_est == total_est and total_est >= 0.0
        mm = int(cast(float, total_est) // 60) if est_ok else 0
        ss = int(cast(float, total_est) % 60) if est_ok else 0
        est_str = f"~{mm:02d}:{ss:02d}" if est_ok else "--:--"

        # Build progress message
        has_progress = update.percent is not None
        if has_progress:
            msg = f"[{update.job_name}] {update.percent:5.1f}% {est_str}"
        else:
            msg = f"[{update.job_name}] {update.message}"

        typer.echo(msg, nl=False)
        print("\r", end="")

    return _on_progress


def _make_run_dir_callback():
    """Create a run directory callback for the scheduler."""

    def _on_run_dir(run_dir: Path):
        pass

    return _on_run_dir


@app.command()
def list():
    """List available engine packages that can be used in job configurations."""
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
