"""qphase: Package Entry Point
---------------------------------------------------------
Execution entry point for the qphase package that delegates to the
main CLI application for command-line interface operations.

Public API
----------
None - This module is used as a package execution entry point

Notes
-----
- Enables running the package as a module with `python -m qphase`
- Imports and runs the main Typer application from qphase.main
- Serves as the CLI entry point for the qphase command-line tool

"""

from .main import app

if __name__ == "__main__":
    app()
