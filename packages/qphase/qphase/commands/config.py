"""qphase: Configuration Management CLI Commands
---------------------------------------------------------
Implements the ``qps config`` command group, providing tools to inspect and modify
the system (``system.yaml``) and global (``global.yaml``) configurations. It supports
viewing configurations with syntax highlighting, setting values using dot-notation
paths with automatic type inference, and resetting configurations to their default
states.

Public API
----------
``show`` : Display system or global configuration with syntax highlighting
``set`` : Set configuration values with type inference
``reset`` : Reset configuration to defaults
"""

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.syntax import Syntax

from qphase.core import (
    SystemConfig,
    load_system_config,
    save_user_config,
)
from qphase.core.config_loader import (
    construct_plugins_config,
    load_global_config,
    save_global_config,
)
from qphase.core.registry import registry

app = typer.Typer(help="Manage configuration")
console = Console()


def _get_global_config_path() -> tuple[Path, bool]:
    """Get the path to global.yaml, following config_dirs from system config.

    Returns:
        tuple: (Path to global.yaml, whether it's in config_dirs)

    """
    system_config = load_system_config()
    config_dirs = system_config.paths.get_config_dirs()

    # First, try the first config directory
    if config_dirs:
        global_path = Path(config_dirs[0]) / "global.yaml"
        if global_path.exists():
            return global_path, True

    # Fall back to current directory
    return Path("global.yaml"), False


@app.command("show")
def show_config(
    system: bool = typer.Option(
        False, "--system", "-s", help="Show system configuration instead of global"
    ),
):
    """Show current configuration.

    By default, shows the global configuration (global.yaml).
    Use --system to show the system configuration (system.yaml).
    """
    if system:
        config = load_system_config()
        title = "System Configuration (system.yaml)"
        # Convert Pydantic model to dict
        data = config.model_dump()
    else:
        # Load global config from the correct path
        global_path, from_config_dirs = _get_global_config_path()
        if not global_path.exists():
            if from_config_dirs:
                console.print(
                    f"[yellow]No global.yaml found at {global_path}.[/yellow]"
                )
            else:
                console.print(
                    "[yellow]No global.yaml found in current directory.[/yellow]"
                )
            return

        data = load_global_config(global_path)
        title = f"Global Configuration ({global_path})"

    console.print(f"\n[bold cyan]{title}[/bold cyan]")

    # Use rich Syntax to print YAML
    import yaml  # type: ignore[import-untyped]

    yaml_str = yaml.dump(data, default_flow_style=False)
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
    console.print(syntax)


@app.command("set")
def set_config(
    key: str = typer.Argument(
        ..., help="Configuration key (dot-separated, e.g. paths.output_dir)"
    ),
    value: str = typer.Argument(..., help="Value to set"),
    system: bool = typer.Option(
        False, "--system", "-s", help="Set value in system configuration"
    ),
):
    """Set a configuration value.

    Modifies global.yaml by default, or system.yaml if --system is used.
    """
    if system:
        config = load_system_config()
        try:
            _set_nested_attr(config, key, value)
            save_user_config(config)
            console.print(f"[green]Updated system config: {key} = {value}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to update system config: {e}[/red]")
            raise typer.Exit(code=1) from e

    else:
        global_path, from_config_dirs = _get_global_config_path()
        if not global_path.exists():
            if from_config_dirs:
                console.print(
                    f"[red]global.yaml not found at {global_path}. "
                    "Run 'qps template <plugin>' to generate one.[/red]"
                )
            else:
                console.print(
                    "[red]global.yaml not found in current directory. "
                    "Run 'qps template <plugin>' to generate one.[/red]"
                )
            raise typer.Exit(code=1)

        config_dict = load_global_config(global_path)
        try:
            _set_nested_dict(config_dict, key, value)
            save_global_config(config_dict, global_path)
            console.print(f"[green]Updated global config: {key} = {value}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to update global config: {e}[/red]")
            raise typer.Exit(code=1) from e


@app.command("reset")
def reset_config(
    system: bool = typer.Option(
        False, "--system", "-s", help="Reset system configuration"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force reset without confirmation"
    ),
):
    """Reset configuration to defaults.

    If --system is used, resets system.yaml to factory defaults.
    Otherwise, resets global.yaml by regenerating it from discovered plugins.
    """
    if system:
        if not force and not typer.confirm(
            "Are you sure you want to reset system configuration?"
        ):
            raise typer.Abort()

        try:
            default_config = SystemConfig()
            save_user_config(default_config)
            console.print("[green]System configuration reset to defaults.[/green]")
        except Exception as e:
            console.print(f"[red]Failed to reset system config: {e}[/red]")
            raise typer.Exit(code=1) from e
    else:
        if not force and not typer.confirm(
            "Are you sure you want to reset global configuration?"
        ):
            raise typer.Abort()

        try:
            # Get the path where global.yaml should be saved
            system_config = load_system_config()
            config_dirs = system_config.paths.get_config_dirs()

            if config_dirs:
                global_path = Path(config_dirs[0]) / "global.yaml"
            else:
                global_path = Path("global.yaml")

            # Regenerate global config from discovered plugins
            global_config = construct_plugins_config(registry)
            save_global_config(global_config, global_path)

            console.print(
                f"[green]Global configuration reset to defaults at "
                f"{global_path}.[/green]"
            )
        except Exception as e:
            console.print(f"[red]Failed to reset global config: {e}[/red]")
            raise typer.Exit(code=1) from e


def _set_nested_dict(data: dict[str, Any], key: str, value: Any) -> None:
    """Set nested dictionary value using dot notation."""
    keys = key.split(".")
    current = data
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
        if not isinstance(current, dict):
            raise ValueError(f"Key '{k}' is not a dictionary")

    # Try to infer type
    if value.lower() == "true":
        value = True
    elif value.lower() == "false":
        value = False
    elif value.isdigit():
        value = int(value)
    else:
        try:
            value = float(value)
        except ValueError:
            pass

    current[keys[-1]] = value


def _set_nested_attr(obj: Any, key: str, value: Any) -> None:
    """Set nested attribute using dot notation."""
    keys = key.split(".")
    current = obj
    for k in keys[:-1]:
        if not hasattr(current, k):
            raise ValueError(f"Attribute '{k}' does not exist")
        current = getattr(current, k)

    if not hasattr(current, keys[-1]):
        raise ValueError(f"Attribute '{keys[-1]}' does not exist")

    # Try to infer type
    if value.lower() == "true":
        value = True
    elif value.lower() == "false":
        value = False
    elif value.isdigit():
        value = int(value)
    else:
        try:
            value = float(value)
        except ValueError:
            pass

    setattr(current, keys[-1], value)
