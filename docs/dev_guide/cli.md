---
layout: default
title: CLI Development
parent: Developer Guide
nav_order: 7
---

# CLI Development

The QPhase CLI is built using [Typer](https://typer.tiangolo.com/), a modern library for building CLI applications based on Python type hints. For terminal output, we use [Rich](https://rich.readthedocs.io/) to provide colored text, tables, and progress bars.

## Architecture

The CLI entry point is `qphase.main:app`. The command structure is modular, with different functional groups (like `run`, `config`, `plugin`) organized into separate sub-applications.

### Directory Structure

```
qphase/
  commands/
    __init__.py
    run.py          # 'qps run' command group
    config.py       # 'qps config' command group
    plugin.py       # 'qps plugin' command group
    ...
  main.py           # Main entry point, aggregates sub-apps
```

## Adding a New Command

To add a new command to QPhase, you typically create a new module in `qphase/commands/` or add to an existing one.

### 1. Create a Command Module

Let's say we want to add a command `qps hello`.

```python
# qphase/commands/hello.py
import typer
from rich.console import Console

app = typer.Typer(help="Greeting commands.")
console = Console()

@app.command()
def world(name: str = typer.Option("User", help="Name to greet")):
    """
    Say hello to someone.
    """
    console.print(f"[bold green]Hello, {name}![/bold green]")
```

### 2. Register in `main.py`

You need to mount the new Typer app into the main application.

```python
# qphase/main.py
from qphase.commands import hello

app = typer.Typer()
app.add_typer(hello.app, name="hello")

if __name__ == "__main__":
    app()
```

Now you can run:
```bash
qps hello world --name "Developer"
```

## Arguments and Options

Typer uses Python type hints to define arguments and options.

*   **Arguments**: Required positional parameters.
*   **Options**: Optional named parameters (flags).

```python
from typing import Annotated
import typer

@app.command()
def process(
    # Positional Argument
    filename: str,
    # Option with default value
    force: bool = False,
    # Option with metadata (help text)
    retries: Annotated[int, typer.Option(help="Number of retries")] = 3
):
    ...
```

## Rich Output

Always use `rich` for output instead of `print()`. This ensures consistent styling across the application.

### Common Patterns

**Status Messages:**
```python
console.print("[green]✓[/green] Operation successful")
console.print("[red]✗[/red] Operation failed")
```

**Tables:**
```python
from rich.table import Table

table = Table(title="Plugins")
table.add_column("Name", style="cyan")
table.add_column("Type", style="magenta")
table.add_row("numpy", "backend")
console.print(table)
```

**Progress Bars:**
```python
from rich.progress import track
import time

for step in track(range(10), description="Processing..."):
    time.sleep(0.1)
```

## Error Handling

CLI commands should handle exceptions gracefully and exit with a non-zero status code on failure.

```python
@app.command()
def risky_operation():
    try:
        do_something()
    except FileNotFoundError:
        console.print("[red]Error: File not found.[/red]")
        raise typer.Exit(code=1)
```

## Testing Commands

Typer provides a `CliRunner` for testing commands without running a subprocess.

```python
from typer.testing import CliRunner
from qphase.main import app

runner = CliRunner()

def test_hello_world():
    result = runner.invoke(app, ["hello", "world", "--name", "Test"])
    assert result.exit_code == 0
    assert "Hello, Test!" in result.stdout
```
