"""qphase: Project Initialization CLI Command
---------------------------------------------------------
Implements the ``qps init`` command, which bootstraps new QPhase projects by creating
the standard directory structure (configs, plugins, runs) and generating initial
configuration files. It automatically discovers available plugins to populate the
default ``global.yaml`` with sensible defaults.

Public API
----------
``init_command`` : Initialize new project with directory structure and configs
"""

from pathlib import Path

import typer
from rich.console import Console

from qphase.core.system_config import load_system_config
from qphase.core.config_loader import (
    construct_plugins_config,
    save_global_config,
)
from qphase.core.registry import discovery, registry


def init_command(
    force: bool = typer.Option(
        False, "--force", "-f", help="Force initialization without confirmation"
    ),
):
    """Initialize a new qphase project.

    Creates the default folder structure and configuration files.

    WARNING: This command will:
    1. Reset the internal plugin registry and re-discover plugins.
    2. Overwrite 'global.yaml' with a fresh default configuration.
    3. Update the new 'global.yaml' with paths from 'system.yaml' (if it exists).

    It does NOT modify 'system.yaml'.
    """
    console = Console()

    if not force:
        if not typer.confirm(
            "This will overwrite global.yaml and reset the registry. Continue?",
            default=False,
        ):
            raise typer.Abort()

    console.print("[bold cyan]Initializing QPhase Project...[/bold cyan]")

    # 1. Reset Registry and Re-discover
    console.print("  [dim]Resetting registry...[/dim]")
    registry.reset()
    discovery.reset()
    console.print("  [dim]Discovering plugins...[/dim]")
    discovery.discover_plugins()
    n_local = discovery.discover_local_plugins()
    console.print(f"    Discovered {n_local} local plugin(s)")

    # 2. Load System Config (to get paths)
    system_config = load_system_config()

    # 3. Create Directories
    console.print("  [dim]Creating directories...[/dim]")
    for plugin_dir in system_config.paths.plugin_dirs:
        plugin_path = Path(plugin_dir)
        if not plugin_path.exists():
            plugin_path.mkdir(parents=True, exist_ok=True)
            console.print(f"    Created plugin directory: {plugin_path}")

        # Create a .qphase_plugins.yaml template file
        plugin_config_file = plugin_path / ".qphase_plugins.yaml"
        if not plugin_config_file.exists():
            plugin_config_file.write_text(
                "# QPhase Plugin Directory Configuration\n"
                "# plugins:\n"
                "#   - type: namespace.plugin_name\n"
                "#     target: module.path:ClassName\n"
                "#     description: Plugin description\n"
            )

    for config_dir in system_config.paths.config_dirs:
        config_path = Path(config_dir)
        if not config_path.exists():
            config_path.mkdir(parents=True, exist_ok=True)
            console.print(f"    Created config directory: {config_path}")

    output_path = Path(system_config.paths.output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        console.print(f"    Created output directory: {output_path}")

    # 4. Dynamically construct and save Global Config
    console.print("  [dim]Generating global.yaml from discovered plugins...[/dim]")
    global_config = construct_plugins_config(registry)

    # Show discovered plugin namespaces
    if global_config:
        plugin_namespaces = list(global_config.keys())
        console.print(f"    Plugin namespaces: {', '.join(plugin_namespaces)}")

    # Save global.yaml to the first config dir or current dir
    if system_config.paths.config_dirs:
        global_path = Path(system_config.paths.config_dirs[0]) / "global.yaml"
    else:
        global_path = Path("global.yaml")

    save_global_config(global_config, global_path)
    console.print(f"    [green]Wrote global config to: {global_path}[/green]")

    console.print("\n[bold green]Initialization Complete![/bold green]")
    console.print("You can now run 'qps run jobs <job_name>' to start simulations.")
