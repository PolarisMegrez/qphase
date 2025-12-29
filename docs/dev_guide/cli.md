---
layout: default
title: CLI Architecture
parent: Developer Guide
nav_order: 6
---

# CLI Architecture

The **Command Line Interface (CLI)** serves as the primary entry point for user interaction. It is built upon the **Typer** library, which leverages Python type hints to generate command-line parsers and help documentation automatically.

## Command Structure

The CLI is organized into a hierarchical command group structure, rooted at the `qps` entry point.

*   `qps` (Root)
    *   `init`: Project bootstrapping.
    *   `run`: Simulation execution group.
        *   `jobs`: Execute job configurations.
    *   `list`: Plugin discovery and listing.
    *   `show`: Plugin introspection.
    *   `template`: Configuration scaffolding.

## Implementation Details

### Entry Point

The main application entry point is defined in `qphase.main:app`. This `Typer` instance aggregates sub-commands and handles global flags (e.g., `--verbose`, `--version`).

### Command Registration

Commands are registered using the `@app.command()` decorator. Typer inspects the function signature to determine argument types and options.

```python
@app.command()
def jobs(
    job_name: str = typer.Argument(..., help="Job name"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Execute a simulation job."""
    # ... implementation ...
```

### Extensibility

While the core commands are hardcoded, the CLI architecture allows for future extensibility. The Registry System includes a `command` namespace, which is reserved for dynamically loading additional CLI sub-commands from plugins. This will allow third-party packages to extend the `qps` tool with custom functionality (e.g., `qps plot`, `qps analyze`).
