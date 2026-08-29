"""Session listing and annotation CLI commands."""

from __future__ import annotations

import typer

from qphase.core.errors import QPhaseError

from ._annotations import (
    ADD_OPTION,
    CLEAR_OPTION,
    DIRECT_OPTION,
    ENGINE_OPTION,
    FACET_OPTION,
    HAS_MODEL_OPTION,
    LIFECYCLE_OPTION,
    LIMIT_OPTION,
    MODEL_OPTION,
    OFFSET_OPTION,
    PRIVATE_OPTION,
    RANGE_OPTION,
    REMOVE_OPTION,
    RETENTION_OPTION,
    TAG_ANY_OPTION,
    TAG_DESCENDANT_OPTION,
    TAG_NAMESPACE_OPTION,
    TAG_OPTION,
    TAG_WITHOUT_OPTION,
    catalog_service,
    fail,
    format_object,
    make_query,
    resolve_lifecycle,
    resolve_retention,
)

app = typer.Typer(help="List and annotate sessions")


@app.command("list")
def list_sessions(
    tag: list[str] = TAG_OPTION,
    tag_any: list[str] = TAG_ANY_OPTION,
    tag_without: list[str] = TAG_WITHOUT_OPTION,
    tag_descendant: str | None = TAG_DESCENDANT_OPTION,
    tag_namespace: str | None = TAG_NAMESPACE_OPTION,
    facet: list[str] = FACET_OPTION,
    range_: list[str] = RANGE_OPTION,
    lifecycle: str | None = LIFECYCLE_OPTION,
    retention: str | None = RETENTION_OPTION,
    model: str | None = MODEL_OPTION,
    engine: str | None = ENGINE_OPTION,
    has_model: bool = HAS_MODEL_OPTION,
    direct: bool = DIRECT_OPTION,
    limit: int = LIMIT_OPTION,
    offset: int = OFFSET_OPTION,
) -> None:
    """List sessions with the full catalog query filter set."""
    try:
        objects = catalog_service().query(
            make_query(
                "session",
                tag=tag,
                tag_any=tag_any,
                tag_without=tag_without,
                tag_descendant=tag_descendant,
                tag_namespace=tag_namespace,
                facet=facet,
                range_=range_,
                lifecycle=lifecycle,
                retention=retention,
                model=model,
                engine=engine,
                has_model=has_model,
                direct=direct,
                limit=limit,
                offset=offset,
            )
        )
    except (QPhaseError, RuntimeError, ValueError) as exc:
        fail(exc)
    for item in objects:
        typer.echo(format_object("session", item))


@app.command("tag")
def tag_session(
    session_id: str = typer.Argument(..., help="Session id"),
    add: list[str] = ADD_OPTION,
    remove: list[str] = REMOVE_OPTION,
    private: bool = PRIVATE_OPTION,
) -> None:
    """Add or remove annotation tags on a session."""
    try:
        summary = catalog_service().tag_session(
            session_id, add=add, remove=remove, private=private
        )
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
