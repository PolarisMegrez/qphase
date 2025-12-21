"""qphase: CLI Entry Point
---------------------------------------------------------
The central entry point for the QPhase Command Line Interface (CLI). This module
initializes the main Typer application and aggregates all sub-commands (init,
run, config, plugin management) into a unified command structure. It serves as
the execution root for the ``qps`` console script.

Public API
----------
``app`` : The main Typer application instance orchestrating all CLI commands.
"""

from __future__ import annotations

import typer

from .commands import config as config_cmd
from .commands import run as run_cmd
from .commands.init import init_command
from .commands.plugin import plugin_app

app = typer.Typer(help="QPhase CLI")


@app.callback()
def main():
    """QPhase command line interface."""
    pass


# Register commands
app.command("init")(init_command)

# Command groups
app.add_typer(plugin_app, name="plugin", help="Manage plugins")
app.add_typer(run_cmd.app, name="run", help="Run SDE simulations")

# Config command group
app.add_typer(config_cmd.app, name="config", help="Manage system configuration")
