"""Project lifecycle CLI commands."""

from pathlib import Path

import typer

from qphase.core.config_loader import construct_plugins_config, save_project_defaults
from qphase.core.errors import QPhaseError
from qphase.core.project import ProjectContext
from qphase.core.registry import discovery, registry
from qphase.service import CatalogService

app = typer.Typer(help="Initialize and inspect QPhase projects")
_PATH_ARGUMENT = typer.Argument(Path("."))


@app.command("init")
def initialize(
    path: Path = _PATH_ARGUMENT,
    name: str | None = typer.Option(None, "--name"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Create a portable QPhase project manifest and standard directories."""
    project = ProjectContext.create(path, name=name, force=force)
    discovery.discover_plugins()
    discovery.discover_local_plugins(project)
    save_project_defaults(construct_plugins_config(registry), project.defaults_path)
    typer.echo(f"Initialized project {project.project_id} at {project.root}")


@app.command("show")
def show() -> None:
    """Show the resolved current project and its storage roots."""
    project = ProjectContext.discover()
    typer.echo(f"Project: {project.manifest.name} ({project.project_id})")
    typer.echo(f"Root: {project.root}")
    typer.echo(f"Workflows: {project.workflow_root}")
    typer.echo(f"Sessions: {project.session_root}")
    typer.echo(f"Defaults: {project.defaults_path}")


@app.command("reindex")
def reindex() -> None:
    """Rebuild the project object catalog read model from disk truth."""
    try:
        stats = CatalogService(ProjectContext.discover()).reindex()
    except QPhaseError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Reindexed catalog: {stats.workflows} workflows, "
        f"{stats.executions} executions, {stats.sessions} sessions, "
        f"{stats.artifacts} artifacts, {stats.occurrences} occurrences, "
        f"{stats.effective_tags} effective tags "
        f"in {stats.duration_seconds:.2f}s"
    )
