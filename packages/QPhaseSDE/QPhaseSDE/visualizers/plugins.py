from __future__ import annotations

"""Visualizer plugin discovery helpers.

Two discovery paths are supported in principle:
- Python entry points (group: 'qphasesde.visualizers')
- Config-specified module paths (dynamic import)

Call register_entry_points() at process startup if desired.
"""

from importlib import import_module
from importlib.metadata import entry_points
from typing import Any

from .register import register_lazy


def register_entry_points(group: str = "qphasesde.visualizers") -> None:
    try:
        eps = entry_points(group=group)
    except Exception:
        return
    for ep in eps:
        try:
            # 'name' becomes the registry key; target is module:attr
            register_lazy(ep.name, f"{ep.module}:{ep.attr}", return_callable=True)
        except Exception:
            # Ignore faulty plugins; callers can introspect registry later
            pass


def register_from_paths(paths: list[str]) -> None:
    for path in paths:
        try:
            # import for side effects; module should call visualizers.register.register(...)
            import_module(path)
        except Exception:
            pass
