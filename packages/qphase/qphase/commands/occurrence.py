"""Artifact occurrence annotation CLI commands."""

from __future__ import annotations

import typer

from qphase.core.errors import QPhaseError

from ._annotations import (
    ADD_OPTION,
    CLEAR_OPTION,
    REMOVE_OPTION,
    catalog_service,
    fail,
    resolve_retention,
)

app = typer.Typer(help="Annotate artifact occurrences (artifact within a session)")


@app.command("tag")
def tag_occurrence(
    session_id: str = typer.Argument(..., help="Session id"),
    artifact_id: str = typer.Argument(..., help="Artifact id"),
    add: list[str] = ADD_OPTION,
    remove: list[str] = REMOVE_OPTION,
) -> None:
    """Add or remove annotation tags on one producing occurrence."""
    try:
        tags = catalog_service().tag_occurrence(
            session_id, artifact_id, add=add, remove=remove
        )
    except (QPhaseError, RuntimeError, FileNotFoundError) as exc:
        fail(exc)
    typer.echo(
        f"occurrence {artifact_id} in {session_id} tags="
        f"[{', '.join(item.tag for item in tags)}]"
    )


@app.command("retention")
def occurrence_retention(
    session_id: str = typer.Argument(..., help="Session id"),
    artifact_id: str = typer.Argument(..., help="Artifact id"),
    value: str | None = typer.Argument(None, help="transient|preserve|evidence|pinned"),
    clear: bool = CLEAR_OPTION,
) -> None:
    """Set or clear the retention policy of one producing occurrence."""
    retention = resolve_retention(value, clear)
    try:
        catalog_service().set_occurrence_retention(session_id, artifact_id, retention)
    except (QPhaseError, RuntimeError, FileNotFoundError) as exc:
        fail(exc)
    typer.echo(f"occurrence {artifact_id} in {session_id} retention={retention}")
