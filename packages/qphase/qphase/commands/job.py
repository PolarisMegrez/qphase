"""Job listing and annotation CLI commands."""

from __future__ import annotations

import typer

from qphase.core.errors import QPhaseError

from ._annotations import (
    ADD_OPTION,
    DIRECT_OPTION,
    FACET_OPTION,
    LIMIT_OPTION,
    OFFSET_OPTION,
    PRIVATE_OPTION,
    RANGE_OPTION,
    REMOVE_OPTION,
    TAG_ANY_OPTION,
    TAG_DESCENDANT_OPTION,
    TAG_NAMESPACE_OPTION,
    TAG_OPTION,
    TAG_WITHOUT_OPTION,
    catalog_service,
    fail,
    format_object,
    make_query,
)

app = typer.Typer(help="List and annotate jobs (logical jobs of one workflow revision)")


@app.command("list")
def list_jobs(
    tag: list[str] = TAG_OPTION,
    tag_any: list[str] = TAG_ANY_OPTION,
    tag_without: list[str] = TAG_WITHOUT_OPTION,
    tag_descendant: str | None = TAG_DESCENDANT_OPTION,
    tag_namespace: str | None = TAG_NAMESPACE_OPTION,
    facet: list[str] = FACET_OPTION,
    range_: list[str] = RANGE_OPTION,
    direct: bool = DIRECT_OPTION,
    limit: int = LIMIT_OPTION,
    offset: int = OFFSET_OPTION,
) -> None:
    """List jobs with the full catalog query filter set."""
    try:
        objects = catalog_service().query(
            make_query(
                "job",
                tag=tag,
                tag_any=tag_any,
                tag_without=tag_without,
                tag_descendant=tag_descendant,
                tag_namespace=tag_namespace,
                facet=facet,
                range_=range_,
                direct=direct,
                limit=limit,
                offset=offset,
            )
        )
    except (QPhaseError, RuntimeError, ValueError) as exc:
        fail(exc)
    for item in objects:
        typer.echo(format_object("job", item))


@app.command("tag")
def tag_job(
    job_id: str = typer.Argument(..., help="Job id (workflow_id@revision:job_name)"),
    add: list[str] = ADD_OPTION,
    remove: list[str] = REMOVE_OPTION,
    private: bool = PRIVATE_OPTION,
) -> None:
    """Add or remove annotation tags on one job of one workflow revision."""
    try:
        tags = catalog_service().tag_job(
            job_id, add=add, remove=remove, private=private
        )
    except (QPhaseError, RuntimeError, ValueError) as exc:
        fail(exc)
    typer.echo(f"job {job_id} tags=[{', '.join(item.tag for item in tags)}]")
