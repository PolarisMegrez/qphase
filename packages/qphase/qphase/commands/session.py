"""Session listing and annotation CLI commands."""

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

app = typer.Typer(help="List and annotate sessions")


@app.command("list")
def list_sessions(
    tag: list[str] = TAG_OPTION,
    lifecycle: str | None = LIFECYCLE_OPTION,
    retention: str | None = RETENTION_OPTION,
    limit: int = LIMIT_OPTION,
) -> None:
    """List sessions, optionally filtered by tag, lifecycle or retention."""
    try:
        objects = catalog_service().query(
            CatalogQuery(
                object_kind="session",
                tags_all=tuple(tag),
                lifecycle=lifecycle,
                retention=retention,
                limit=limit,
            )
        )
    except (QPhaseError, RuntimeError) as exc:
        fail(exc)
    for item in objects:
        typer.echo(format_object("session", item))


@app.command("tag")
def tag_session(
    session_id: str = typer.Argument(..., help="Session id"),
    add: list[str] = ADD_OPTION,
    remove: list[str] = REMOVE_OPTION,
) -> None:
    """Add or remove annotation tags on a session."""
    try:
        summary = catalog_service().tag_session(session_id, add=add, remove=remove)
    except (QPhaseError, RuntimeError, FileNotFoundError) as exc:
        fail(exc)
    typer.echo(f"session {summary.session_id} updated")


@app.command("lifecycle")
def session_lifecycle(
    session_id: str = typer.Argument(..., help="Session id"),
    value: str | None = typer.Argument(
        None, help="active|reference|superseded|archived"
    ),
    clear: bool = CLEAR_OPTION,
) -> None:
    """Set or clear the lifecycle of a session."""
    lifecycle = resolve_lifecycle(value, clear)
    try:
        catalog_service().set_session_lifecycle(session_id, lifecycle)
    except (QPhaseError, RuntimeError, FileNotFoundError) as exc:
        fail(exc)
    typer.echo(f"session {session_id} lifecycle={lifecycle}")


@app.command("retention")
def session_retention(
    session_id: str = typer.Argument(..., help="Session id"),
    value: str | None = typer.Argument(None, help="transient|preserve|evidence|pinned"),
    clear: bool = CLEAR_OPTION,
) -> None:
    """Set or clear the retention policy of a session."""
    retention = resolve_retention(value, clear)
    try:
        catalog_service().set_session_retention(session_id, retention)
    except (QPhaseError, RuntimeError, FileNotFoundError) as exc:
        fail(exc)
    typer.echo(f"session {session_id} retention={retention}")
