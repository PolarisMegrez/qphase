"""qphase: Plugin Management CLI Commands
---------------------------------------------------------
Implements the `qps` plugin-related commands (list, show, template),
offering introspection capabilities for the plugin ecosystem. It allows users
to list registered plugins by namespace, view detailed metadata and configuration
schemas for specific plugins, and generate commented YAML/JSON configuration
templates to facilitate job setup.

Public API
----------
`list` : List plugins by category with source and descriptions
`show` : Display detailed plugin info and configuration parameters
`template` : Generate YAML/JSON configuration templates for plugins
"""

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from qphase.core.config_loader import load_global_config
from qphase.core.registry import discovery, registry
from qphase.core.utils import schema_to_yaml_map

plugin_app = typer.Typer(help="Manage and discover plugins")


@plugin_app.command(name="list")
def list_command(
    categories: str | None = typer.Argument(
        default=None,
        help="Plugin categories to list. Use '.' for all or comma-separated list.",
    ),
):
    """List available plugins by category."""
    console = Console()

    try:
        # Ensure plugins are discovered
        discovery.discover_plugins()
        discovery.discover_local_plugins()

        # Parse categories
        if categories is None or categories.strip() == ".":
            category_list = None
        else:
            category_list = [
                c.strip().lower() for c in categories.split(",") if c.strip()
            ]

        # Get plugin list
        all_plugins = registry.list(namespace=None)

        if category_list is None:
            categories_to_show = sorted(all_plugins.keys())
        else:
            categories_to_show = [c for c in category_list if c in all_plugins]
            for c in category_list:
                if c not in all_plugins:
                    all_plugins[c] = []
            categories_to_show = sorted(set(categories_to_show))

        if not categories_to_show:
            console.print(
                f"[yellow]No plugins found for categories: {categories}[/yellow]"
            )
            return

        # Display results
        console.print("\n[blue]Available Plugins[/blue]")
        console.print("[blue]" + "=" * 60 + "[/blue]")

        for category in categories_to_show:
            plugins = all_plugins.get(category, [])
            if not plugins:
                console.print(f"\n[cyan]{category}[/cyan]: (empty)")
                continue

            console.print(
                f"\n[cyan]{category}[/cyan]: "
                f"({len(plugins)} plugin{'s' if len(plugins) != 1 else ''})"
            )

            # Create table for this category
            table = Table(show_header=True, header_style="bold green")
            table.add_column("Name", style="cyan", no_wrap=True)
            table.add_column("Source", style="yellow", no_wrap=True)
            table.add_column("Description", style="yellow")

            for plugin_name in plugins:
                # Get plugin info from registry (includes metadata fields directly)
                plugin_info = registry.list(namespace=category).get(plugin_name, {})

                # Extract source information
                # (metadata fields are directly in plugin_info)
                source_display = _get_source_display(plugin_info)

                # Get and display description
                description = _get_plugin_description(category, plugin_name)
                if not description:
                    description = ""

                table.add_row(plugin_name, source_display, description)

            console.print(table)

        console.print("\n[blue]" + "=" * 60 + "[/blue]")
        console.print(
            f"[green]Total: "
            f"{sum(len(all_plugins.get(c, [])) for c in categories_to_show)} "
            f"plugins across {len(categories_to_show)} categories[/green]"
        )

    except Exception as e:
        console.print(f"[red]Error listing plugins: {e}[/red]")
        raise typer.Exit(code=1) from e


def _get_source_display(metadata: dict) -> str:
    """Extract human-readable source information from metadata.

    Parameters
    ----------
    metadata : dict
        Plugin metadata dictionary (flat dict from registry.list())

    Returns
    -------
    str
        Formatted source string like "package v1.0" or "path/to/dir/"

    """
    # Check if it's a resource package
    package_name = metadata.get("package_name")
    package_version = metadata.get("package_version")

    if package_name:
        # Resource package
        version_str = f" v{package_version}" if package_version else ""
        return f"{package_name}{version_str}"

    # Local plugin - show directory
    source_file = metadata.get("source_file")
    if source_file:
        # Get the parent directory
        try:
            path = Path(source_file)
            # Show the directory containing the plugin file
            dir_name = path.parent.name
            return f"{dir_name}/"
        except Exception:
            # Fallback to showing a generic path indicator
            return "local"

    # Fallback
    return "unknown"


def _get_plugin_description(category: str, plugin_name: str) -> str | None:
    """Get the description for a plugin by directly accessing its class.

    Parameters
    ----------
    category : str
        Plugin category
    plugin_name : str
        Plugin name

    Returns
    -------
    str | None
        Plugin description or None if not available

    """
    try:
        # Get the plugin entry
        category_info = registry.list(namespace=category)
        plugin_info = category_info.get(plugin_name, {})

        if not plugin_info:
            return None

        # Get the target (dotted path to the class)
        target = plugin_info.get("target") or plugin_info.get("module_path")
        if not target:
            return None

        # Import the target class without instantiating
        cls = registry._import_target(target)

        # Check if it has a description ClassVar
        if hasattr(cls, "description"):
            desc = cls.description
            if isinstance(desc, str) and desc.strip():
                return desc

    except Exception:
        # Silently ignore if we can't load the plugin
        pass

    return None


def _display_source_info(plugin_info: dict, console: Console) -> None:
    """Display plugin source information in a structured format."""
    # Determine plugin type
    is_local = plugin_info.get("source_file") is not None
    package_name = plugin_info.get("package_name")

    if is_local:
        # Local plugin
        source_file = plugin_info.get("source_file", "")
        try:
            path = Path(source_file)
            dir_name = path.parent.name
            console.print(f"  Path:      [yellow]{dir_name}/[/yellow]")
        except Exception:
            console.print(f"  Path:      [yellow]{source_file}[/yellow]")

        console.print("  Type:      [yellow]Local plugin[/yellow]")

        # Show module path for local plugins
        module_path = plugin_info.get("module_path", "")
        if module_path:
            console.print(f"  Module:    [yellow dim]{module_path}[/yellow dim]")
    else:
        # Resource package
        package_version = plugin_info.get("package_version", "")
        version_str = f" v{package_version}" if package_version else ""

        console.print(f"  Engine:   [yellow]{package_name}{version_str}[/yellow]")
        console.print("  Type:      [yellow]Resource package[/yellow]")

        # Show module path if available
        module_path = plugin_info.get("module_path", "")
        if module_path:
            console.print(f"  Module:    [yellow dim]{module_path}[/yellow dim]")


@plugin_app.command(name="show")
def show_command(
    # noqa: B008 - typer.Argument is a decorator, not a function call in default value
    plugins: list[str] = typer.Argument(  # noqa: B008
        ...,
        help="Plugin dotted paths (e.g., 'model.vdp_two_mode', 'backend.numpy'). "
        "Accepts multiple arguments.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show additional metadata"
    ),
):
    """Show detailed information about plugins using dotted path notation.

    Accepts one or more plugin specifications in dotted notation (namespace.name).

    Examples
    --------
        qps show model.vdp_two_mode
        qps show backend.numpy integrator.euler
        qps show model.vdp_two_mode backend.cupy

    Displays detailed information about each plugin including source,
    description, and configuration parameters.

    """
    console = Console()

    try:
        # Ensure plugins are discovered (including local plugins)
        discovery.discover_plugins()
        discovery.discover_local_plugins()

        # Process each plugin
        for i, plugin_spec in enumerate(plugins):
            # Parse dotted notation
            if "." not in plugin_spec:
                console.print(
                    f"[red]Error: Invalid plugin specification '{plugin_spec}'. "
                    "Use 'namespace.name' format.[/red]"
                )
                raise typer.Exit(code=1)

            parts = plugin_spec.split(".", 1)
            category = parts[0]
            name = parts[1]

            # Get plugin info
            category = category.lower()
            plugin_info = registry.list(namespace=category).get(name)

            if not plugin_info:
                console.print(
                    f"[red]Plugin '{name}' not found in category '{category}'[/red]"
                )
                raise typer.Exit(code=1)

            # Display plugin header
            if len(plugins) > 1:
                console.print(
                    f"\n[bold cyan]Plugin {i + 1}/{len(plugins)}:[/bold cyan] "
                    f"[yellow]{plugin_spec}[/yellow]"
                )
            else:
                console.print(f"\n[bold cyan]{plugin_spec}[/bold cyan]")
            console.print("[cyan]" + "=" * 60 + "[/cyan]")

            # Display source information
            console.print("\n[bold yellow]Source:[/bold yellow]")
            _display_source_info(plugin_info, console)

            # Display description
            console.print("\n[bold yellow]Description:[/bold yellow]")
            description = _get_plugin_description(category, name)
            if description:
                console.print(f"  {description}")
            else:
                console.print("  [yellow dim]No description available[/yellow dim]")

            # Display configuration parameters
            _display_config_parameters(category, name)

            # Display metadata if verbose
            if verbose:
                _display_metadata(plugin_info, console)

            # Add separator between multiple plugins
            if i < len(plugins) - 1:
                console.print("\n" + "=" * 60 + "\n")

    except Exception as e:
        console.print(f"[red]Error showing plugin: {e}[/red]")
        raise typer.Exit(code=1) from e


def _display_config_parameters(category: str, name: str) -> None:
    """Display configuration parameters in a table format."""
    console = Console()

    try:
        schema = registry.get_plugin_schema(category, name)
        if not schema:
            console.print("\n[bold yellow]Configuration Parameters:[/bold yellow]")
            console.print(
                "  [yellow dim]No configuration schema available[/yellow dim]"
            )
            return

        if not hasattr(schema, "model_fields"):
            console.print("\n[bold yellow]Configuration Parameters:[/bold yellow]")
            console.print("  [yellow dim]No configuration fields found[/yellow dim]")
            return

        fields = schema.model_fields

        if not fields:
            console.print("\n[bold yellow]Configuration Parameters:[/bold yellow]")
            console.print("  [yellow dim]No configuration parameters[/yellow dim]")
            return

        console.print("\n[bold yellow]Configuration Parameters[/bold yellow]")

        # Create Rich Table
        table = Table(show_header=True, header_style="bold green")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Type", style="yellow", no_wrap=True)
        table.add_column("Default", style="magenta", no_wrap=True)
        table.add_column("Description", style="yellow")

        for field_name, field_info in fields.items():
            # Get field type
            field_type = _format_field_type(field_info.annotation)

            # Get default value
            default = _format_field_default(field_info)

            # Get description
            description = _format_field_description(field_info)

            table.add_row(
                field_name, field_type, default, description if description else ""
            )

        console.print(table)

    except Exception as e:
        console.print("\n[bold yellow]Configuration Parameters:[/bold yellow]")
        console.print(
            f"  [yellow dim]Error loading configuration schema: {e}[/yellow dim]"
        )


def _format_field_type(annotation: Any) -> str:
    """Format field type annotation for display."""
    if annotation is None:
        return "Any"

    # Handle class types (e.g., <class 'float'>)
    if hasattr(annotation, "__name__"):
        return annotation.__name__

    # Get string representation
    type_str = str(annotation)

    # Handle <class '...'> format
    if type_str.startswith("<class '") and type_str.endswith("'>"):
        return type_str[8:-2]  # Extract the class name

    # Handle Union types (typing.Union or | syntax)
    if "|" in type_str:
        # Extract union members
        type_str = type_str.replace("typing.Union[", "").replace("]", "")
        type_str = type_str.replace(" | ", " or ")

    # Remove module prefixes for readability
    if type_str.startswith(("typing.", "pydantic.", "collections.abc.")):
        type_str = type_str.split(".", 1)[-1]

    return type_str


def _format_field_default(field_info: Any) -> str:
    """Format field default value for display."""
    try:
        # Check if field has a default value
        if hasattr(field_info, "default") and field_info.default is not ...:
            default = field_info.default
            # Handle Pydantic special values
            if hasattr(default, "__class__") and default.__class__.__name__ in (
                "Undefined",
                "PydanticUndefinedType",
            ):
                return "(required)"  # More user-friendly than "REQUIRED"
            return str(default)

        # Check for Field() default
        if hasattr(field_info, "default_factory") and field_info.default_factory:
            return "<factory>"

        return "(required)"  # More user-friendly than "REQUIRED"
    except Exception:
        return "(required)"


def _format_field_description(field_info: Any) -> str:
    """Format field description for display."""
    try:
        if hasattr(field_info, "description") and field_info.description:
            return field_info.description
        return ""
    except Exception:
        return ""


def _display_metadata(plugin_info: dict, console: Console) -> None:
    """Display plugin metadata."""
    if not plugin_info:
        return

    console.print("\n[bold yellow]Metadata[/bold yellow]")

    # Display key metadata fields
    key_fields = [
        "auto_discovered",
        "module_path",
        "source_file",
        "registered_at",
        "package_name",
        "package_version",
    ]
    for key in key_fields:
        if key in plugin_info and plugin_info[key]:
            console.print(f"  {key}: [yellow dim]{plugin_info[key]}[/yellow dim]")

    # Display any additional metadata
    additional = {k: v for k, v in plugin_info.items() if k not in key_fields and v}
    if additional:
        console.print("\n  [yellow dim]Additional:[/yellow dim]")
        for key, value in sorted(additional.items()):
            console.print(f"    - {key}: [yellow dim]{value}[/yellow dim]")


@plugin_app.command(name="template")
def template_command(
    # noqa: B008 - typer.Argument is a decorator, not a function call in default value
    plugins: list[str] = typer.Argument(  # noqa: B008
        ...,
        help="Plugin dotted paths (e.g., 'model.vdp_two_mode', 'backend.numpy'). "
        "Accepts multiple arguments.",
    ),
    output: str = typer.Option("-", help="Output file (default: stdout)"),
    format: str = typer.Option("yaml", help="Output format (yaml or json)"),
):
    """Generate configuration templates for plugins using dotted path notation.

    Accepts one or more plugin specifications in dotted notation (namespace.name).

    Examples
    --------
        qps template model.vdp_two_mode
        qps template backend.numpy integrator.euler
        qps template model.vdp_two_mode backend.cupy

    Generates configuration templates based on plugin schemas.
    Merges values from global.yaml if available.

    """
    console = Console()

    try:
        # Ensure plugins are discovered (including local plugins)
        discovery.discover_plugins()
        discovery.discover_local_plugins()

        # Collect all plugin configs organized by namespace
        all_configs: dict[str, dict[str, Any]] = {}
        errors = []

        # Process each plugin
        for plugin_spec in plugins:
            # Parse dotted notation
            if "." not in plugin_spec:
                errors.append(
                    f"Invalid plugin specification '{plugin_spec}'. "
                    "Use 'namespace.name' format."
                )
                continue

            parts = plugin_spec.split(".", 1)
            namespace = parts[0]
            name = parts[1]

            # Fetch schema
            schema = registry.get_plugin_schema(namespace, name)

            if not schema:
                errors.append(f"No configuration schema found for '{plugin_spec}'")
                continue

            try:
                # Load global config for merging
                global_config = {}
                global_path = Path("global.yaml")
                if global_path.exists():
                    try:
                        global_config = load_global_config(global_path)
                    except Exception:
                        # Ignore if global config is broken,
                        # just generate default template
                        pass

                # Extract relevant section from global config
                existing_values = global_config.get(namespace, {}).get(name, {})
                if not isinstance(existing_values, dict):
                    existing_values = {}

                # Generate template without 'name' field
                template_data = schema_to_yaml_map(schema, existing_values, name)

                # Organize by namespace
                if namespace not in all_configs:
                    all_configs[namespace] = {}
                all_configs[namespace][name] = template_data

            except Exception as e:
                errors.append(f"Failed to generate template for '{plugin_spec}': {e}")

        # Display errors if any
        if errors:
            for error in errors:
                console.print(f"[red]Error: {error}[/red]")
            raise typer.Exit(code=1)

        if not all_configs:
            console.print("[red]No valid plugins to generate template for.[/red]")
            raise typer.Exit(code=1)

        # Display combined configuration
        console.print("\n[bold cyan]Configuration Template[/bold cyan]\n")
        _display_config(all_configs, format, output)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from e


def _display_config(config: Any, fmt: str, output: str) -> None:
    """Display configuration in specified format."""
    import json

    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    console = Console()

    if fmt == "json":
        content = json.dumps(config, indent=2, default=str)
        syntax = "json"
    else:
        import io

        stream = io.StringIO()
        yaml.dump(config, stream)
        content = stream.getvalue()
        syntax = "yaml"

    if output == "-":
        console.print(Syntax(content, syntax, theme="monokai", line_numbers=True))
    else:
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        console.print(f"[green]Template saved to {output}[/green]")
