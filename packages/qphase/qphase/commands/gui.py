"""CLI command for launching the local GUI backend."""

from __future__ import annotations

import typer


def gui_command(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind"),
    port: int = typer.Option(8000, "--port", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Enable Uvicorn reload"),
) -> None:
    """Launch the local QPhase GUI API and web console."""
    try:
        import uvicorn
    except ImportError as exc:
        typer.echo("Install GUI dependencies with: pip install 'qphase[gui]'")
        raise typer.Exit(code=1) from exc

    typer.echo(f"Starting QPhase GUI at http://{host}:{port}")
    uvicorn.run(
        "qphase.gui.api:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )
