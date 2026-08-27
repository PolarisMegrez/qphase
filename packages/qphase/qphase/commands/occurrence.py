"""Artifact occurrence listing and annotation CLI commands."""

from __future__ import annotations

import typer

from qphase.core.errors import QPhaseError

from ._annotations import (
    ADD_OPTION,
    CLEAR_OPTION,
    DIRECT_OPTION,
    FACET_OPTION,
    LIFECYCLE_OPTION,
    LIMIT_OPTION,
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
    resolve_retention,
)

app = typer.Typer(
    help="List and annotate artifact occurrences (artifact within a session)"
)

_JOB_OPTION = typer.Option(
    None,
    "--job",
    help="Job name; required when the artifact occurs in several jobs",
)
_SESSION_OPTION = typer.Option(None, "--session", help="Filter by session id")
_ARTIFACT_OPTION = typer.Option(None, "--artifact", help="Filter by artifact id")


@app.command("list")
def list_occurrences(
    session: str | None = _SESSION_OPTION,
    artifact: str | None = _ARTIFACT_OPTION,
    tag: list[str] = TAG_OPTION,
    tag_any: list[str] = TAG_ANY_OPTION,
    tag_without: list[str] = TAG_WITHOUT_OPTION,
    tag_descendant: str | None = TAG_DESCENDANT_OPTION,
    tag_namespace: str | None = TAG_NAMESPACE_OPTION,
    facet: list[str] = FACET_OPTION,
    range_: list[str] = RANGE_OPTION,
    lifecycle: str | None = LIFECYCLE_OPTION,
    retention: str | None = RETENTION_OPTION,
    direct: bool = DIRECT_OPTION,
    limit: int = LIMIT_OPTION,
    offset: int = OFFSET_OPTION,
) -> None:
    """List occurrences with the full catalog query filter set."""
    if session is not None:
        facet = [*facet, f"session_id={session}"]
    if artifact is not None:
        facet = [*facet, f"artifact_id={artifact}"]
    try:
        objects = catalog_service().query(
            make_query(
                "occurrence",
                tag=tag,
                tag_any=tag_any,
                tag_without=tag_without,
                tag_descendant=tag_descendant,
                tag_namespace=tag_namespace,
                facet=facet,
                range_=range_,
                lifecycle=lifecycle,
                retention=retention,
                direct=direct,
                limit=limit,
                offset=offset,
            )
        )
    except (QPhaseError, RuntimeError, ValueError) as exc:
        fail(exc)
    for item in objects:
        typer.echo(format_object("occurrence", item))


@app.command("tag")
def tag_occurrence(
    session_id: str = typer.Argument(..., help="Session id"),
    artifact_id: str = typer.Argument(..., help="Artifact id"),
    add: list[str] = ADD_OPTION,
    remove: list[str] = REMOVE_OPTION,
    job: str | None = _JOB_OPTION,
    private: bool = PRIVATE_OPTION,
) -> None:
    """Add or remove annotation tags on one producing occurrence."""
    try:
        tags = catalog_service().tag_occurrence(
            session_id, artifact_id, job_name=job, add=add, remove=remove,
            private=private,
        )
    except (QPhaseError, RuntimeError, ValueError, FileNotFoundError) as exc:
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
    job: str | None = _JOB_OPTION,
) -> None:
    """Set or clear the retention policy of one producing occurrence."""
    retention = resolve_retention(value, clear)
    try:
        catalog_service().set_occurrence_retention(
            session_id, artifact_id, retention, job_name=job
        )
    except (QPhaseError, RuntimeError, ValueError, FileNotFoundError) as exc:
        fail(exc)
    typer.echo(f"occurrence {artifact_id} in {session_id} retention={retention}")
