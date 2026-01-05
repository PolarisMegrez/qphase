---
description: CLI Architecture
---

# CLI Architecture

The **Command Line Interface (CLI)** serves as the primary entry point for user interaction. It is built upon the **Typer** library, which leverages Python type hints to generate command-line parsers and help documentation automatically.

## Command Structure

The CLI is organized into a hierarchical command group structure, rooted at the `qphase` entry point.

*   `qphase` (Root)
    *   `init`: Project bootstrapping.
    *   `run`: Simulation execution.
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

While the core commands are hardcoded, the CLI architecture allows for future extensibility. The Registry System includes a `command` namespace, which is reserved for dynamically loading additional CLI sub-commands from plugins. This will allow third-party packages to extend the `qphase` tool with custom functionality (e.g., `qphase plot`, `qphase analyze`).

## Integration with Scheduler

The CLI acts as a thin client for the `Scheduler`. When `qphase run` is invoked:
1.  It parses the command line arguments.
2.  It loads the `SystemConfig`.
3.  It instantiates the `Scheduler`.
4.  It delegates the execution to `scheduler.run()`.

This separation ensures that the core execution logic is not coupled to the CLI interface, allowing simulations to be triggered programmatically (e.g., from a Jupyter notebook) if needed.
