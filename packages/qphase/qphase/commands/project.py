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


@app.command("migrate")
def migrate(
    dry_run: bool = typer.Option(False, "--dry-run", help="Report only"),
) -> None:
    """Preview the history migration (the real migration lands in Phase 4)."""
    if not dry_run:
        typer.echo(
            "Error: the formal history migration will be provided in Phase 4; "
            "use --dry-run to preview what it will do"
        )
        raise typer.Exit(code=1)
    try:
        report = CatalogService(ProjectContext.discover()).migration_dry_run()
    except QPhaseError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(f"Migration dry-run: {report.sessions_total} sessions scanned")
    imports = report.legacy_metadata_imports
    typer.echo(f"Legacy metadata imports ({len(imports)}):")
    for item in imports:
        typer.echo(f"  {item.session_id}  alias={item.alias} note={item.note}")
    typer.echo(
        f"Sessions without annotations or legacy metadata: "
        f"{report.untouched_sessions} (no action)"
    )
    invalid = report.invalid_snapshot_tags
    affected = sorted({item.session_id for item in invalid})
    typer.echo(f"Invalid snapshot tags ({len(invalid)} in {len(affected)} sessions):")
    for entry in invalid:
        typer.echo(
            f"  {entry.session_id}  {entry.source} tag {entry.tag!r}: {entry.error}"
        )
