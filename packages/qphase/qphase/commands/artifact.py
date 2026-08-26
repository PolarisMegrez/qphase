"""Artifact listing and annotation CLI commands."""

from __future__ import annotations

import typer

from qphase.core.catalog import CatalogQuery
from qphase.core.errors import QPhaseError

from ._annotations import (
    ADD_OPTION,
    CLEAR_OPTION,
    LIFECYCLE_OPTION,
    LIMIT_OPTION,
    REMOVE_OPTION,
    RETENTION_OPTION,
    TAG_OPTION,
    catalog_service,
    fail,
    format_object,
    resolve_lifecycle,
    resolve_retention,
)

app = typer.Typer(help="List and annotate artifacts")


@app.command("list")
def list_artifacts(
    tag: list[str] = TAG_OPTION,
    lifecycle: str | None = LIFECYCLE_OPTION,
    retention: str | None = RETENTION_OPTION,
    limit: int = LIMIT_OPTION,
) -> None:
    """List artifacts, optionally filtered by tag, lifecycle or retention."""
    try:
        objects = catalog_service().query(
            CatalogQuery(
                object_kind="artifact",
                tags_all=tuple(tag),
                lifecycle=lifecycle,
                retention=retention,
                limit=limit,
            )
        )
    except (QPhaseError, RuntimeError) as exc:
        fail(exc)
    for item in objects:
        typer.echo(format_object("artifact", item))


@app.command("tag")
def tag_artifact(
    artifact_id: str = typer.Argument(..., help="Artifact id"),
    add: list[str] = ADD_OPTION,
    remove: list[str] = REMOVE_OPTION,
) -> None:
    """Add or remove annotation tags on an artifact."""
    try:
        tags = catalog_service().tag_artifact(artifact_id, add=add, remove=remove)
    except (QPhaseError, RuntimeError, FileNotFoundError) as exc:
        fail(exc)
    typer.echo(f"artifact {artifact_id} tags=[{', '.join(item.tag for item in tags)}]")


@app.command("lifecycle")
def artifact_lifecycle(
    artifact_id: str = typer.Argument(..., help="Artifact id"),
    value: str | None = typer.Argument(
        None, help="active|reference|superseded|archived"
    ),
    clear: bool = CLEAR_OPTION,
) -> None:
    """Set or clear the lifecycle of an artifact."""
    lifecycle = resolve_lifecycle(value, clear)
    try:
        catalog_service().set_artifact_lifecycle(artifact_id, lifecycle)
    except (QPhaseError, RuntimeError, FileNotFoundError) as exc:
        fail(exc)
    typer.echo(f"artifact {artifact_id} lifecycle={lifecycle}")


@app.command("retention")
def artifact_retention(
    artifact_id: str = typer.Argument(..., help="Artifact id"),
    value: str | None = typer.Argument(None, help="transient|preserve|evidence|pinned"),
    clear: bool = CLEAR_OPTION,
) -> None:
    """Set or clear the retention policy of an artifact."""
    retention = resolve_retention(value, clear)
    try:
        catalog_service().set_artifact_retention(artifact_id, retention)
    except (QPhaseError, RuntimeError, FileNotFoundError) as exc:
        fail(exc)
    typer.echo(f"artifact {artifact_id} retention={retention}")
