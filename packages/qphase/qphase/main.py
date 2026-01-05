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

import typer

from .commands import config as config_cmd
from .commands import run as run_cmd
from .commands.init import init_command
from .commands.plugin import list_command, show_command, template_command

app = typer.Typer(help="QPhase CLI")


@app.callback()
def main():
    """QPhase command line interface."""
    pass


# Register commands
app.command("init")(init_command)
app.command("list")(list_command)
app.command("show")(show_command)
app.command("template")(template_command)
app.command("run")(run_cmd.run_command)

# Config command group
app.add_typer(config_cmd.app, name="config", help="Manage system configuration")
