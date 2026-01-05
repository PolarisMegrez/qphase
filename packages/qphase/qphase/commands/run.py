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

# Module-level singleton for typer.Argument to avoid function call in default (B008)
JOB_NAMES_ARG = typer.Argument(
    ...,
    help="Name(s) of the job(s) to run (searched in configs/jobs/ directory)",
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
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    log_file: str | None = typer.Option(None, help="Write logs to file path"),
    log_json: bool = typer.Option(False, help="Log in JSON format"),
    suppress_warnings: bool = typer.Option(False, help="Suppress warnings output"),
):
    """Run SDE simulation jobs by name from configs/jobs/ directory.

    JOB_NAMES should be the name(s) of job configuration file(s) (without extension)
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
        qphase run jobs my_simulation
        qphase run jobs job1 job2
        qphase run jobs --list
        qphase run jobs --verbose my_job

    """
    # Handle "list" argument as a command to list engines
    if "list" in job_names:
        _list_engines()
        return

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

        # Find job configuration files
        cfg_paths = []
        for job_name in job_names:
            cfg_path = _find_job_config(system_cfg.paths.config_dirs, job_name)

            if cfg_path is None or not cfg_path.exists():
                log.error(f"Job '{job_name}' not found in configs/jobs/ directories")
                log.error(f"Searched in: {system_cfg.paths.config_dirs}")
                available_jobs = list_available_jobs(system_cfg)
                if available_jobs:
                    log.error(f"Available jobs: {', '.join(available_jobs)}")
                raise typer.Exit(code=1)

            log.info(f"Found job configuration: {cfg_path}")
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
        log.info(f"Loading {len(cfg_paths)} configuration file(s)")
        job_list = load_jobs_from_files(cfg_paths)

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

        # Job counter (1-based)
        job_str = f"[{update.job_index + 1}/{update.total_jobs}]"

        # Build progress message
        has_progress = update.percent is not None
        if has_progress:
            # Ensure percent is float
            p_val = float(update.percent) if update.percent is not None else 0.0
            percent = p_val * 100.0

            # Visual bar
            bar_len = 20
            filled = int(p_val * bar_len)
            # Clamp filled to bar_len
            filled = max(0, min(filled, bar_len))
            bar = "=" * filled + "-" * (bar_len - filled)

            msg = (
                f"{job_str} [{update.job_name}] {percent:5.1f}% [{bar}] ETA: {est_str}"
            )
        else:
            msg = f"{job_str} [{update.job_name}] {update.message}"

        # Clear line and print
        # Use ANSI escape code to clear line if supported, or just padding
        # \033[K clears from cursor to end of line
        sys.stdout.write(f"\r{msg:<80}")
        sys.stdout.flush()

        if has_progress and percent >= 100.0:
            sys.stdout.write("\n")
            sys.stdout.flush()

    return _on_progress


def _make_run_dir_callback():
    """Create a run directory callback for the scheduler."""

    def _on_run_dir(run_dir: Path):
        pass

    return _on_run_dir


@app.command(name="list")
def list_engines():
    """List available engine packages that can be used in job configurations."""
    _list_engines()


run_command = jobs
