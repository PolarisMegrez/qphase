"""Shared helpers for the catalog annotation CLI commands."""

from __future__ import annotations

from typing import NoReturn, cast

import typer

from qphase.core.annotations import Lifecycle, RetentionPolicy
from qphase.core.project import ProjectContext
from qphase.service import CatalogService
from qphase.service.models import CatalogObject

LIFECYCLES = ("active", "reference", "superseded", "archived")
RETENTIONS = ("transient", "preserve", "evidence", "pinned")

TAG_OPTION = typer.Option([], "--tag", help="Required effective tag (repeatable)")
LIFECYCLE_OPTION = typer.Option(None, "--lifecycle", help="Filter by lifecycle")
RETENTION_OPTION = typer.Option(None, "--retention", help="Filter by retention")
LIMIT_OPTION = typer.Option(100, "--limit", help="Maximum number of objects")
ADD_OPTION = typer.Option([], "--add", help="Tag to add (repeatable)")
REMOVE_OPTION = typer.Option([], "--remove", help="Tag to remove (repeatable)")
CLEAR_OPTION = typer.Option(False, "--clear", help="Clear the value")


def catalog_service() -> CatalogService:
    """Construct the catalog service for the discovered project."""
    return CatalogService(ProjectContext.discover())


def fail(exc: Exception) -> NoReturn:
    """Print a brief error and exit with a non-zero status."""
    typer.echo(f"Error: {exc}")
    raise typer.Exit(code=1) from exc


def resolve_lifecycle(value: str | None, clear: bool) -> Lifecycle | None:
    """Resolve the lifecycle argument/--clear pair to a typed value."""
    if clear:
        return None
    if value in LIFECYCLES:
        return cast("Lifecycle", value)
    typer.echo(
        f"Error: lifecycle must be one of {', '.join(LIFECYCLES)} or use --clear"
    )
    raise typer.Exit(code=1)


def resolve_retention(value: str | None, clear: bool) -> RetentionPolicy | None:
    """Resolve the retention argument/--clear pair to a typed value."""
    if clear:
        return None
    if value in RETENTIONS:
        return cast("RetentionPolicy", value)
    typer.echo(
        f"Error: retention must be one of {', '.join(RETENTIONS)} or use --clear"
    )
    raise typer.Exit(code=1)


def format_object(kind: str, item: CatalogObject) -> str:
    """Render one catalog object as a single line: id + key facets + tags."""
    facets = item.facets
    tags = ", ".join(tag.tag for tag in item.effective_tags)
    if kind == "session":
        summary = (
            f"status={facets.get('status')} lifecycle={facets.get('lifecycle')} "
            f"retention={facets.get('retention')}"
        )
    elif kind == "artifact":
        summary = (
            f"health={facets.get('health')} lifecycle={facets.get('lifecycle')} "
            f"retention={facets.get('retention')}"
        )
    else:
        summary = f"retention={facets.get('effective_retention')}"
    return f"{item.id}  {summary} tags=[{tags}]"
