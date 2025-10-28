from __future__ import annotations

import sys
import os
from pathlib import Path
import typer

# Prefer local workspace package over any installed version
packages_dir = Path(__file__).resolve().parents[2]  # .../packages
sys.path.insert(0, str(packages_dir))
sys.path.insert(0, str(packages_dir / "QPhaseSDE"))
import QPhaseSDE  # noqa: F401

from .commands import run as run_cmd
from .commands import analyze as analyze_cmd

app = typer.Typer(help="QPhaseSDE CLI")


@app.callback()
def main():
	"""QPhaseSDE command line interface."""
	pass


app.add_typer(run_cmd.app, name="run", help="Run SDE simulations")
app.add_typer(analyze_cmd.app, name="analyze", help="Analyze existing results (no recompute)")

