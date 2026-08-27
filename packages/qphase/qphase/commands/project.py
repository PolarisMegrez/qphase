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
_MIGRATE_LIST_LIMIT = 10


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
    typer.echo(
        f"Rebuildable: {report.rebuildable_workflow_revisions} workflow revisions, "
        f"{report.rebuildable_jobs} jobs"
    )
    convertible = report.convertible_occurrence_keys
    ambiguous = report.ambiguous_occurrence_keys
    typer.echo(
        f"Legacy occurrence keys: {len(convertible)} convertible, "
        f"{len(ambiguous)} ambiguous"
    )
    for conv in convertible[:_MIGRATE_LIST_LIMIT]:
        typer.echo(f"  {conv.session_id}  {conv.old_key} -> {conv.new_key}")
    if len(convertible) > _MIGRATE_LIST_LIMIT:
        typer.echo(f"  ... and {len(convertible) - _MIGRATE_LIST_LIMIT} more")
    for amb in ambiguous[:_MIGRATE_LIST_LIMIT]:
        typer.echo(
            f"  {amb.session_id}  {amb.old_key} ambiguous"
            f" (locations: {', '.join(amb.locations) or 'none'})"
        )
    if len(ambiguous) > _MIGRATE_LIST_LIMIT:
        typer.echo(f"  ... and {len(ambiguous) - _MIGRATE_LIST_LIMIT} more")
    duplicates = report.duplicate_artifacts
    typer.echo(f"Duplicate artifact identities ({len(duplicates)}):")
    for dup in duplicates[:_MIGRATE_LIST_LIMIT]:
        flag = " [conflict]" if dup.conflict else ""
        typer.echo(f"  {dup.artifact_id}{flag}: {', '.join(dup.locations)}")
    if len(duplicates) > _MIGRATE_LIST_LIMIT:
        typer.echo(f"  ... and {len(duplicates) - _MIGRATE_LIST_LIMIT} more")
    provenance = report.assignments_without_policy_revision
    if provenance:
        summary = ", ".join(f"{scope}={count}" for scope, count in provenance.items())
        typer.echo(f"Assignments without policy provenance: {summary}")
    else:
        typer.echo("Assignments without policy provenance: none")
    if report.catalog_drift is None:
        typer.echo("Catalog reindex parity: absent (no catalog yet)")
    elif report.catalog_drift:
        typer.echo("Catalog reindex parity: drift (run `qphase project reindex`)")
    else:
        typer.echo("Catalog reindex parity: in sync")
    counts = ", ".join(
        f"{kind}={count}" for kind, count in report.object_counts.items()
    )
    typer.echo(f"Object counts: {counts}")
    issues = report.location_issues_by_kind
    if issues:
        summary = ", ".join(f"{kind}={count}" for kind, count in issues.items())
        typer.echo(f"Location issues: {summary}")
    else:
        typer.echo("Location issues: none")
    typer.echo(
        f"Private store: {report.private_tag_count} tags, "
        f"{report.saved_view_count} saved views, "
        f"{report.private_annotation_count} private annotations"
    )
