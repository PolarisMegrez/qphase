"""Tag policy CLI commands."""

from __future__ import annotations

import typer

from qphase.core.errors import QPhaseError

from ._annotations import catalog_service, fail

app = typer.Typer(help="Inspect the project tag policy")
policy_app = typer.Typer(help="Inspect and validate the project tag policy")
app.add_typer(policy_app, name="policy")


@policy_app.command("show")
def policy_show() -> None:
    """Show the project tag policy location, revision and namespaces."""
    try:
        info = catalog_service().tag_policy()
    except QPhaseError as exc:
        fail(exc)
    if info.path is None:
        typer.echo("no tag policy (configs/tags.yaml absent)")
        return
    typer.echo(f"Path: {info.path}")
    typer.echo(f"Revision: {info.revision}")
    typer.echo("Namespaces:")
    for name, rule in info.namespaces.items():
        typer.echo(f"  - {name}: {rule}")


@policy_app.command("validate")
def policy_validate() -> None:
    """Validate configs/tags.yaml; exits non-zero on errors."""
    try:
        info = catalog_service().tag_policy()
    except QPhaseError as exc:
        fail(exc)
    if info.path is None:
        typer.echo("no tag policy (configs/tags.yaml absent)")
        return
    typer.echo(f"valid: {info.path} (revision {info.revision})")


@app.command("promote")
def promote_tag(
    kind: str = typer.Argument(
        ...,
        help="Object kind: project|workflow|job|execution|session|artifact|occurrence",
    ),
    object_id: str = typer.Argument(..., help="Catalog object id"),
    tag: str = typer.Argument(..., help="Private tag to promote to the shared layer"),
) -> None:
    """Move one private tag into the shared annotation document."""
    try:
        tags = catalog_service().promote_tag(kind, object_id, tag)
    except (QPhaseError, RuntimeError, ValueError) as exc:
        fail(exc)
    typer.echo(f"{kind} {object_id} tags=[{', '.join(item.tag for item in tags)}]")
