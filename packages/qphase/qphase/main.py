"""qphase: CLI Entry Point
---------------------------------------------------------
The central entry point for the QPhase Command Line Interface (CLI). This module
initializes the main Typer application and aggregates all sub-commands (init,
run, config, plugin management) into a unified command structure. It serves as
the execution root for the ``qphase`` console script.

Public API
----------
``app`` : The main Typer application instance orchestrating all CLI commands.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

from .commands import config as config_cmd
from .commands import project as project_cmd
from .commands import run as run_cmd
from .commands import workflow as workflow_cmd
from .commands.gui import gui_command
from .commands.plugin import list_command, show_command, template_command

app = typer.Typer(help="QPhase CLI")
_PROJECT_OPTION = typer.Option(None, "--project", help="Project root or qphase.toml")


@app.callback()
def main(
    project: Path | None = _PROJECT_OPTION,
):
    """QPhase command line interface."""
    if project is not None:
        os.environ["QPHASE_PROJECT"] = str(project)


# Register commands
app.command("list")(list_command)
app.command("show")(show_command)
app.command("template")(template_command)
app.command("run")(run_cmd.run_command)
app.command("gui")(gui_command)

# Config command group
app.add_typer(config_cmd.app, name="config", help="Manage system configuration")
app.add_typer(project_cmd.app, name="project")
app.add_typer(workflow_cmd.app, name="workflow")
