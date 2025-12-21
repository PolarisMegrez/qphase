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

import builtins
import sys
from pathlib import Path
from typing import cast

import typer

from qphase.core.system_config import load_system_config
from qphase.core import JobProgressUpdate, Scheduler
from qphase.core.config_loader import load_jobs_from_files
from qphase.core.errors import (
    QPhaseError,
    configure_logging,
    get_logger,
)
from qphase.core.registry import discovery, registry

app = typer.Typer()


@app.command()
def list(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed engine information"
    ),
):
    """List available engine packages."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Ensure plugins are discovered
    discovery.discover_plugins()
    discovery.discover_local_plugins()

    # Get engine plugins from registry
    engine_plugins = registry.list(namespace="engine")

    if not engine_plugins:
        console.print("[yellow]No engine packages found[/yellow]")
        return

    # Display results
    console.print("\nAvailable Engine Packages", style="blue")
    console.print("=" * 60, style="blue")

    table = Table(title="Engine Packages")
    table.add_column("Name", style="cyan")
    table.add_column("Module Path", style="dim")
    table.add_column("Description", style="yellow")

    for engine_name in sorted(engine_plugins.keys()):
        plugin_info = engine_plugins.get(engine_name, {})
        module_path = plugin_info.get("module_path", "")

        # Try to get description from plugin class
        description = ""
        try:
            target = plugin_info.get("target") or plugin_info.get("module_path")
            if target:
                cls = registry._import_target(target)
                if hasattr(cls, "description"):
                    desc = cls.description
                    if isinstance(desc, str) and desc.strip():
                        description = desc
        except Exception:
            pass

        if verbose:
            table.add_row(
                engine_name,
                module_path if module_path else "N/A",
                description if description else "No description available",
            )
        else:
            table.add_row(
                engine_name,
                module_path if module_path else "",
                description if description else "",
            )

    console.print(table)
    console.print("\n" + "=" * 60, style="blue")
    console.print(
        f"Total: {len(engine_plugins)} engine package"
        f"{'s' if len(engine_plugins) != 1 else ''}",
        style="green",
    )


@app.command()
def jobs(
    # noqa: B008 - typer.Argument is a decorator, not a function call in default value
    configs: builtins.list[str] = typer.Argument(  # noqa: B008
        ...,
        help="Path to one or more job YAML/JSON configuration files",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    log_file: str | None = typer.Option(None, help="Write logs to file path"),
    log_json: bool = typer.Option(False, help="Log in JSON format"),
    suppress_warnings: bool = typer.Option(False, help="Suppress warnings output"),
):
    """Run SDE simulation jobs from one or more configuration files.

    Each CONFIG argument should be a path to a YAML or JSON file containing
    a single job configuration. Each file must contain:
    - A single JobConfig with one engine per job

    Job file format:
        name: job_name
        engine: engine_name
        backend:
          numpy:
            float_dtype: float64
        model:
          vdp_two_mode:
            D: 1.0
        engine:
          sde:
            t_end: 10.0

    Examples:
        qps run jobs job1.yaml job2.yaml
        qps run jobs simulation.yaml visualization.yaml

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

        # Resolve all config file paths
        cfg_paths = []
        for config in configs:
            cfg_path = Path(config).resolve()

            # Check if file exists
            if not cfg_path.exists():
                log.error(f"Configuration file not found: {cfg_path}")
                raise typer.Exit(code=1)

            cfg_paths.append(cfg_path)

        # Add config directories to Python path for model imports
        for cfg_path in cfg_paths:
            for cand in (cfg_path.parent, cfg_path.parent.parent):
                if cand.exists():
                    pstr = str(cand)
                    if pstr not in sys.path:
                        sys.path.insert(0, pstr)

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
        total_est = update.total_duration_estimate
        est_ok = total_est is not None and total_est == total_est and total_est >= 0.0
        mm = int(cast(float, total_est) // 60) if est_ok else 0
        ss = int(cast(float, total_est) % 60) if est_ok else 0
        est_str = f"~{mm:02d}:{ss:02d}" if est_ok else "--:--"

        # Build progress message
        if update.has_progress:
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
