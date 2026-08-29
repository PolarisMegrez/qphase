"""Artifact listing and annotation CLI commands."""

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
    QUANTITY_OPTION,
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

app = typer.Typer(help="List and annotate artifacts")

@app.command("list")
def list_artifacts(
    tag: list[str] = TAG_OPTION,
    tag_any: list[str] = TAG_ANY_OPTION,
    tag_without: list[str] = TAG_WITHOUT_OPTION,
    tag_descendant: str | None = TAG_DESCENDANT_OPTION,
    tag_namespace: str | None = TAG_NAMESPACE_OPTION,
    facet: list[str] = FACET_OPTION,
    range_: list[str] = RANGE_OPTION,
    lifecycle: str | None = LIFECYCLE_OPTION,
    retention: str | None = RETENTION_OPTION,
    quantity: str | None = QUANTITY_OPTION,
    direct: bool = DIRECT_OPTION,
    limit: int = LIMIT_OPTION,
    offset: int = OFFSET_OPTION,
) -> None:
    """List artifacts with the full catalog query filter set."""
    try:
        objects = catalog_service().query(
            make_query(
                "artifact",
                tag=tag,
                tag_any=tag_any,
                tag_without=tag_without,
                tag_descendant=tag_descendant,
                tag_namespace=tag_namespace,
                facet=facet,
                range_=range_,
                lifecycle=lifecycle,
                retention=retention,
                quantity=quantity,
                direct=direct,
                limit=limit,
                offset=offset,
            )
        )
    except (QPhaseError, RuntimeError, ValueError) as exc:
        fail(exc)
    for item in objects:
        typer.echo(format_object("artifact", item))


@app.command("tag")
def tag_artifact(
    artifact_id: str = typer.Argument(..., help="Artifact id"),
    add: list[str] = ADD_OPTION,
    remove: list[str] = REMOVE_OPTION,
    private: bool = PRIVATE_OPTION,
) -> None:
    """Add or remove annotation tags on an artifact."""
    try:
        tags = catalog_service().tag_artifact(
            artifact_id, add=add, remove=remove, private=private
        )
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
