"""Manage machine policy and project plugin defaults."""

from __future__ import annotations

import json
from io import StringIO
from typing import Any

import typer
from rich.console import Console
from rich.syntax import Syntax
from ruamel.yaml import YAML

from qphase.core.config_loader import (
    construct_plugins_config,
    load_project_defaults,
    save_project_defaults,
)
from qphase.core.project import ProjectContext
from qphase.core.registry import discovery, registry
from qphase.core.system_config import (
    load_system_config,
    reset_user_config,
    save_user_config,
)
from qphase.service import RegistryService

app = typer.Typer(help="Manage machine policy and project defaults")
console = Console()


def _discover() -> ProjectContext:
    project = ProjectContext.discover()
    discovery.discover_plugins()
    discovery.discover_local_plugins(project)
    return project


@app.command("options")
def subplugin_options(path: str) -> None:
    """List child implementations accepted by a plugin slot."""
    project = _discover()
    parent, slot = path.rsplit("/", 1) if "/" in path else path.rsplit(".", 1)
    summary = RegistryService(project=project).get_subplugin_options(parent, slot)
    console.print(f"[bold cyan]{parent}/{slot}[/bold cyan]")
    for option in summary.options:
        marker = " [default]" if option.name == summary.default else ""
        console.print(f"  {option.name}{marker}: {option.plugin.description}")


@app.command("schema")
def plugin_schema(path: str) -> None:
    """Display the composite configuration schema for a plugin path."""
    project = _discover()
    payload = RegistryService(project=project).get_composite_schema(path)
    console.print(Syntax(json.dumps(payload, indent=2), "json", theme="monokai"))


@app.command("show")
def show_config(
    system: bool = typer.Option(False, "--system", "-s", help="Show machine policy"),
) -> None:
    """Show project defaults, or machine policy with ``--system``."""
    if system:
        data = load_system_config().model_dump()
        title = "System policy"
    else:
        project = ProjectContext.discover()
        data = load_project_defaults(project.defaults_path)
        title = f"Project defaults ({project.defaults_path})"
    stream = StringIO()
    YAML().dump(data, stream)
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print(Syntax(stream.getvalue(), "yaml", theme="monokai", line_numbers=True))


@app.command("set")
def set_config(
    key: str,
    value: str,
    system: bool = typer.Option(False, "--system", "-s", help="Update machine policy"),
) -> None:
    """Set a project-default or machine-policy value using dot notation."""
    parsed = _parse_value(value)
    if system:
        config = load_system_config()
        _set_nested_attr(config, key, parsed)
        save_user_config(config)
    else:
        project = ProjectContext.discover()
        data = load_project_defaults(project.defaults_path)
        _set_nested_dict(data, key, parsed)
        save_project_defaults(data, project.defaults_path)
    console.print(f"[green]Updated {key} = {parsed!r}[/green]")


@app.command("reset")
def reset_config(
    system: bool = typer.Option(False, "--system", "-s"),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Reset machine overrides or regenerate project plugin defaults."""
    if not force and not typer.confirm("Reset this configuration?"):
        raise typer.Abort()
    if system:
        reset_user_config()
        console.print("[green]System policy reset.[/green]")
        return
    project = _discover()
    save_project_defaults(construct_plugins_config(registry), project.defaults_path)
    console.print(f"[green]Regenerated {project.defaults_path}.[/green]")


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _set_nested_dict(data: dict[str, Any], key: str, value: Any) -> None:
    segments = key.split(".")
    current = data
    for segment in segments[:-1]:
        child = current.setdefault(segment, {})
        if not isinstance(child, dict):
            raise ValueError(f"{segment!r} is not a mapping")
        current = child
    current[segments[-1]] = value


def _set_nested_attr(obj: Any, key: str, value: Any) -> None:
    segments = key.split(".")
    current = obj
    for segment in segments[:-1]:
        current = getattr(current, segment)
    setattr(current, segments[-1], value)
