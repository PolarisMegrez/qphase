"""Shared helpers for the catalog annotation CLI commands."""

from __future__ import annotations

from typing import NoReturn, cast

import typer

from qphase.core.annotations import Lifecycle, RetentionPolicy
from qphase.core.catalog import CatalogQuery
from qphase.core.project import ProjectContext
from qphase.service import CatalogService
from qphase.service.catalog import parse_facet_filters, parse_range_filters
from qphase.service.models import CatalogObject

LIFECYCLES = ("active", "reference", "superseded", "archived")
RETENTIONS = ("transient", "preserve", "evidence", "pinned")

TAG_OPTION = typer.Option([], "--tag", help="Required effective tag (repeatable)")
TAG_ANY_OPTION = typer.Option([], "--tag-any", help="Any-of effective tag (repeatable)")
TAG_WITHOUT_OPTION = typer.Option(
    [], "--tag-without", help="Excluded effective tag (repeatable)"
)
TAG_DESCENDANT_OPTION = typer.Option(
    None, "--tag-descendant", help="Match the tag or one of its descendants"
)
TAG_NAMESPACE_OPTION = typer.Option(None, "--tag-namespace", help="Tag namespace")
FACET_OPTION = typer.Option([], "--facet", help="Facet equality k=v (repeatable)")
RANGE_OPTION = typer.Option([], "--range", help="Facet range k=low..high (repeatable)")
DIRECT_OPTION = typer.Option(
    False, "--direct", help="Match only tags assigned directly on the object"
)
LIFECYCLE_OPTION = typer.Option(None, "--lifecycle", help="Filter by lifecycle")
RETENTION_OPTION = typer.Option(None, "--retention", help="Filter by retention")
PLUGIN_OPTION = typer.Option(
    None, "--plugin", help="Filter jobs by configured plugin (namespace:name)"
)
QUANTITY_OPTION = typer.Option(
    None, "--quantity", help="Filter artifacts by product quantity"
)
MODEL_OPTION = typer.Option(None, "--model", help="Filter by model plugin")
ENGINE_OPTION = typer.Option(None, "--engine", help="Filter by engine plugin")
HAS_MODEL_OPTION = typer.Option(
    False, "--has-model", help="Keep only objects with a model plugin"
)
LIMIT_OPTION = typer.Option(100, "--limit", help="Maximum number of objects")
OFFSET_OPTION = typer.Option(0, "--offset", help="Pagination offset")
ADD_OPTION = typer.Option([], "--add", help="Tag to add (repeatable)")
REMOVE_OPTION = typer.Option([], "--remove", help="Tag to remove (repeatable)")
CLEAR_OPTION = typer.Option(False, "--clear", help="Clear the value")
PRIVATE_OPTION = typer.Option(False, "--private", help="Edit user-private tags")


def catalog_service() -> CatalogService:
    """Construct the catalog service for the discovered project."""
    return CatalogService(ProjectContext.discover())


def fail(exc: Exception) -> NoReturn:
    """Print a brief error and exit with a non-zero status."""
    typer.echo(f"Error: {exc}")
    raise typer.Exit(code=1) from exc


def resolve_lifecycle(value: str | None, clear: bool) -> Lifecycle | None:
    """Resolve the lifecycle argument/--clear pair to a typed value."""
    if clear:
        return None
    if value in LIFECYCLES:
        return cast("Lifecycle", value)
    typer.echo(
        f"Error: lifecycle must be one of {', '.join(LIFECYCLES)} or use --clear"
    )
    raise typer.Exit(code=1)


def resolve_retention(value: str | None, clear: bool) -> RetentionPolicy | None:
    """Resolve the retention argument/--clear pair to a typed value."""
    if clear:
        return None
    if value in RETENTIONS:
        return cast("RetentionPolicy", value)
    typer.echo(
        f"Error: retention must be one of {', '.join(RETENTIONS)} or use --clear"
    )
    raise typer.Exit(code=1)


def make_query(
    object_kind: str,
    *,
    tag: list[str] | tuple[str, ...] = (),
    tag_any: list[str] | tuple[str, ...] = (),
    tag_without: list[str] | tuple[str, ...] = (),
    tag_descendant: str | None = None,
    tag_namespace: str | None = None,
    facet: list[str] | tuple[str, ...] = (),
    range_: list[str] | tuple[str, ...] = (),
    lifecycle: str | None = None,
    retention: str | None = None,
    plugin: str | None = None,
    quantity: str | None = None,
    model: str | None = None,
    engine: str | None = None,
    has_model: bool = False,
    direct: bool = False,
    limit: int = 100,
    offset: int = 0,
) -> CatalogQuery:
    """Build a catalog query from the shared list-command option set."""
    return CatalogQuery(
        object_kind=object_kind,
        facets=parse_facet_filters(facet),
        ranges=parse_range_filters(range_),
        tags_all=tuple(tag),
        tags_any=tuple(tag_any),
        tags_without=tuple(tag_without),
        tag_descendant_of=tag_descendant,
        tag_namespace=tag_namespace,
        lifecycle=lifecycle,
        retention=retention,
        plugin=plugin,
        quantity=quantity,
        model=model,
        engine=engine,
        has_model=has_model,
        effective=not direct,
        limit=limit,
        offset=offset,
    )


def format_object(kind: str, item: CatalogObject) -> str:
    """Render one catalog object as a single line: id + key facets + tags."""
    facets = item.facets
    tags = ", ".join(tag.tag for tag in item.effective_tags)
    if kind == "session":
        summary = (
            f"status={facets.get('status')} lifecycle={facets.get('lifecycle')} "
            f"retention={facets.get('retention')}"
        )
    elif kind == "artifact":
        summary = (
            f"bundle={facets.get('bundle_type')} lifecycle={facets.get('lifecycle')} "
            f"retention={facets.get('retention')}"
        )
    elif kind == "occurrence":
        summary = f"retention={facets.get('effective_retention')}"
    elif kind == "project":
        summary = f"name={facets.get('name')}"
    elif kind == "workflow":
        summary = (
            f"title={facets.get('title')!r} collection={facets.get('collection')} "
            f"jobs={facets.get('job_count')}"
        )
    elif kind == "job":
        summary = (
            f"workflow={facets.get('workflow_id')} engine={facets.get('engine')} "
            f"model={facets.get('model')}"
        )
    elif kind == "execution":
        summary = (
            f"state={facets.get('state')} submitted_at={facets.get('submitted_at')}"
        )
    else:
        summary = ""
    return f"{item.id}  {summary} tags=[{tags}]"
