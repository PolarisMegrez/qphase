# Development Guide

## Environment Setup

### Standard Development (No GPU Required)

```powershell
# Development environment without optional CUDA support
uv sync --dev

# This includes all dependencies except cupy
```

This is also the environment used by the default CI and documentation workflows.
CUDA extras are intentionally excluded from those workflows because they pull
large GPU-only wheels that are unnecessary for CPU tests and docs.

### With NVIDIA GPU Support (Python 3.13+)

```powershell
# Development environment with CUDA backend
uv sync --all-extras --dev

# This installs cupy-cuda12x for GPU acceleration
```

## Optional Backends

### CUDA Support (`cuda` extra)

- **Requires**: Python 3.13.2 or compatible, NVIDIA GPU
- **Installation**: `uv sync --all-extras` or `uv sync --extra cuda`
- **Purpose**: GPU-accelerated computation via CuPy

To install only CUDA without other extras:

```powershell
uv sync --extra cuda --dev
```

## Pre-commit Hooks

The pre-commit configuration is set up for flexible development:

- **ruff**: Linting and formatting (all environments)
- **mypy**: Type checking with `--ignore-missing-imports` (handles optional dependencies)

To run pre-commit manually:

```powershell
uv run pre-commit run --all-files
```

To run the same checks as the default CI job:

```powershell
uv sync --dev
uv run ruff check .
uv run mypy .
uv run pytest
```

## Running Tests

Tests are layered with pytest markers (registered in `pyproject.toml`):

- `unit` — fast, isolated tests (auto-assigned to any test without a layer marker)
- `integration` — cross-component / cross-plugin tests
- `e2e` — full CLI / workflow end-to-end tests
- `gpu` — requires CUDA/cupy (auto-skipped when unavailable)
- `slow` — long-running tests (typically >5s)

### Fast Loop (Local Development)

```powershell
uv sync --dev
uv run pytest -m unit
```

Runs in ~10 seconds. Use this while iterating.

### Standard Gate (What CI Runs on PRs)

```powershell
uv run pytest -m "not slow"
```

Tests requiring CUDA will be skipped automatically if cupy is not installed.

### Full Testing (Before Release / With GPU Support)

On GPU-equipped machine (Python 3.13+):

```powershell
uv sync --all-extras --dev
uv run pytest
```

All tests will run, including CUDA-accelerated and slow end-to-end tests.

## Notes

- **No GPU?** Just use `uv sync --dev`. You can still develop and contribute!
- **Have GPU?** Use `uv sync --all-extras --dev` to test with CUDA acceleration
- Optional dependencies are gracefully handled - missing imports won't break development
