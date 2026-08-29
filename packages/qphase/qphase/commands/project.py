"""Project lifecycle CLI commands."""

from pathlib import Path

import typer

from qphase.core.config_loader import construct_plugins_config, save_project_defaults
from qphase.core.errors import QPhaseError
from qphase.core.project import ProjectContext
from qphase.core.registry import discovery, registry
from qphase.service import CatalogService

from ._annotations import (
    ADD_OPTION,
    CLEAR_OPTION,
    PRIVATE_OPTION,
    REMOVE_OPTION,
    catalog_service,
    fail,
)

app = typer.Typer(help="Initialize and inspect QPhase projects")
_PATH_ARGUMENT = typer.Argument(Path("."))

#: Maximum number of preview entries printed per migration report list.


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


@app.command("tag")
def tag_project(
    add: list[str] = ADD_OPTION,
    remove: list[str] = REMOVE_OPTION,
    private: bool = PRIVATE_OPTION,
) -> None:
    """Add or remove annotation tags on the project itself."""
    try:
        tags = catalog_service().tag_project(add=add, remove=remove, private=private)
    except (QPhaseError, RuntimeError, ValueError) as exc:
        fail(exc)
    typer.echo(f"project tags=[{', '.join(item.tag for item in tags)}]")


@app.command("alias")
def project_alias(
    value: str | None = typer.Argument(None, help="Project alias"),
    clear: bool = CLEAR_OPTION,
) -> None:
    """Set or clear the project alias."""
    if not clear and value is None:
        typer.echo("Error: pass an alias value or use --clear")
        raise typer.Exit(code=1)
    try:
        catalog_service().set_project_alias(None if clear else value)
    except (QPhaseError, RuntimeError) as exc:
        fail(exc)
    typer.echo(f"project alias={None if clear else value}")


@app.command("note")
def project_note(
    value: str | None = typer.Argument(None, help="Project note"),
    clear: bool = CLEAR_OPTION,
) -> None:
    """Set or clear the project note."""
    if not clear and value is None:
        typer.echo("Error: pass a note value or use --clear")
        raise typer.Exit(code=1)
    try:
        catalog_service().set_project_note(None if clear else value)
    except (QPhaseError, RuntimeError) as exc:
        fail(exc)
    typer.echo(f"project note={None if clear else value}")


@app.command("reindex")
def reindex() -> None:
    """Rebuild the project object catalog read model from disk truth."""
    try:
        service = CatalogService(ProjectContext.discover())
        stats = service.reindex()
    except QPhaseError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Reindexed catalog: {stats.projects} project, "
        f"{stats.workflows} workflow revisions, {stats.jobs} jobs, "
        f"{stats.executions} executions, {stats.sessions} sessions, "
        f"{stats.artifacts} artifacts, {stats.occurrences} occurrences, "
        f"{stats.effective_tags} effective tags "
        f"in {stats.duration_seconds:.2f}s"
    )
    issues = service.location_issues()
    if issues:
        typer.echo(f"Location issues ({len(issues)}):")
        for issue in issues:
            typer.echo(f"  [{issue['kind']}] {issue['path']}: {issue['message']}")
