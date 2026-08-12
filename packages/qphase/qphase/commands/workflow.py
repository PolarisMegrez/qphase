"""Workflow catalog CLI commands."""

import json

import typer

from qphase.core.project import ProjectContext
from qphase.core.workflow import WorkflowCatalog

app = typer.Typer(help="Inspect versioned workflows")


@app.command("list")
def list_workflows(
    collection: str | None = typer.Option(None, "--collection", "-c"),
    tag: str | None = typer.Option(None, "--tag", "-t"),
    query: str | None = typer.Option(None, "--query", "-q"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List workflows recursively by stable ID."""
    items = WorkflowCatalog(ProjectContext.discover()).search(
        collection=collection, tag=tag, query=query
    )
    if json_output:
        typer.echo(
            json.dumps(
                [
                    {
                        "id": item.id,
                        "title": item.title,
                        "collection": item.collection,
                        "tags": item.tags,
                        "path": item.relative_path,
                        "job_count": item.job_count,
                    }
                    for item in items
                ],
                indent=2,
            )
        )
        return
    for item in items:
        group = f" [{item.collection}]" if item.collection else ""
        typer.echo(f"{item.id}{group}: {item.title} ({item.job_count} jobs)")


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
