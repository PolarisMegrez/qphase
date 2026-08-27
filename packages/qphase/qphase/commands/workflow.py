"""Workflow catalog CLI commands."""

from __future__ import annotations

import json

import typer

from qphase.core.errors import QPhaseError
from qphase.core.project import ProjectContext
from qphase.core.workflow import WorkflowCatalog

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
    make_query,
)

app = typer.Typer(help="Inspect and annotate versioned workflows")


@app.command("list")
def list_workflows(
    collection: str | None = typer.Option(None, "--collection", "-c"),
    query: str | None = typer.Option(None, "--query", "-q"),
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
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List workflow revisions from the object catalog.

    Each revision of a workflow id is one row (``workflow_id@revision``);
    ``--query`` filters the listed id/title client-side.
    """
    if collection is not None:
        facet = [*facet, f"collection={collection}"]
    try:
        objects = catalog_service().query(
            make_query(
                "workflow",
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
    if query:
        needle = query.lower()
        objects = [
            item
            for item in objects
            if needle in item.id.lower()
            or needle in str(item.facets.get("title", "")).lower()
        ]
    if json_output:
        typer.echo(
            json.dumps(
                [
                    {
                        "id": item.id,
                        "title": item.facets.get("title"),
                        "collection": item.facets.get("collection"),
                        "tags": [tag.tag for tag in item.effective_tags],
                        "path": item.facets.get("relative_path"),
                        "job_count": item.facets.get("job_count"),
                    }
                    for item in objects
                ],
                indent=2,
            )
        )
        return
    for item in objects:
        group = (
            f" [{item.facets['collection']}]" if item.facets.get("collection") else ""
        )
        typer.echo(
            f"{item.id}{group}: {item.facets.get('title')} "
            f"({item.facets.get('job_count')} jobs)"
        )


@app.command("tag")
def tag_workflow(
    revision_id: str = typer.Argument(
        ..., help="Workflow revision (workflow_id@revision)"
    ),
    add: list[str] = ADD_OPTION,
    remove: list[str] = REMOVE_OPTION,
    private: bool = PRIVATE_OPTION,
) -> None:
    """Add or remove annotation tags on one workflow revision."""
    try:
        tags = catalog_service().tag_workflow(
            revision_id, add=add, remove=remove, private=private
        )
    except (QPhaseError, RuntimeError, ValueError) as exc:
        fail(exc)
    typer.echo(
        f"workflow {revision_id} tags=[{', '.join(item.tag for item in tags)}]"
    )


@app.command("show")
def show_workflow(reference: str) -> None:
    """Print one resolved workflow as JSON."""
    catalog = WorkflowCatalog(ProjectContext.discover())
    workflow = catalog.load(reference)
    typer.echo(json.dumps(workflow.model_dump(mode="json", by_alias=True), indent=2))


@app.command("path")
def workflow_path(reference: str) -> None:
    """Print the current path for a stable workflow ID."""
    typer.echo(WorkflowCatalog(ProjectContext.discover()).resolve(reference).path)
