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
