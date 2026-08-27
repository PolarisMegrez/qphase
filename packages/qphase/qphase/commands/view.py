"""Saved catalog view CLI commands (user-private)."""

from __future__ import annotations

import typer

from qphase.core.catalog import CatalogQuery
from qphase.core.errors import QPhaseError

from ._annotations import (
    LIFECYCLE_OPTION,
    LIMIT_OPTION,
    RETENTION_OPTION,
    TAG_OPTION,
    catalog_service,
    fail,
)

app = typer.Typer(help="Manage user-private saved catalog views")

_KIND_OPTION = typer.Option(
    ...,
    "--kind",
    help="Object kind: project|workflow|job|execution|session|artifact|occurrence",
)


@app.command("save")
def save_view(
    name: str = typer.Argument(..., help="View name"),
    kind: str = _KIND_OPTION,
    tag: list[str] = TAG_OPTION,
    lifecycle: str | None = LIFECYCLE_OPTION,
    retention: str | None = RETENTION_OPTION,
    limit: int = LIMIT_OPTION,
) -> None:
    """Save the current filter set as a named view in the private store."""
    try:
        query = CatalogQuery(
            object_kind=kind,
            tags_all=tuple(tag),
            lifecycle=lifecycle,
            retention=retention,
            limit=limit,
        )
        catalog_service().save_view(name, query)
    except (QPhaseError, ValueError) as exc:
        fail(exc)
    typer.echo(f"saved view {name!r} (kind={query.object_kind})")


@app.command("list")
def list_views() -> None:
    """List saved views."""
    try:
        views = catalog_service().list_views()
    except QPhaseError as exc:
        fail(exc)
    for name, query in views:
        typer.echo(
            f"{name}  kind={query.object_kind} tags={list(query.tags_all)} "
            f"lifecycle={query.lifecycle} retention={query.retention} "
            f"limit={query.limit}"
        )


@app.command("delete")
def delete_view(name: str = typer.Argument(..., help="View name")) -> None:
    """Delete a saved view."""
    try:
        catalog_service().delete_view(name)
    except QPhaseError as exc:
        fail(exc)
    typer.echo(f"deleted view {name!r}")
