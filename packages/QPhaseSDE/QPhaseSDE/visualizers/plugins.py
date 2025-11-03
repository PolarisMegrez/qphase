"""
QPhaseSDE: Visualizer plugins
-----------------------------
Discover and register visualization plugins provided by third-party packages or
local modules.

Behavior
- Entry points: scan the ``qphasesde.visualizers`` group and lazily register
    each discovered target by dotted path.
- Module paths: import the given modules for side effects; modules are expected
    to perform their own registration during import.

Notes
- Faulty entry points or imports are ignored; the central registry can be
    inspected later.
- Duplicate registrations are governed by the core registry; this module does
    not raise for them.
"""

from importlib import import_module
from importlib.metadata import entry_points

__all__ = [
    "register_entry_points",
    "register_from_paths",
]

from .register import register_lazy

def register_entry_points(group: str = "qphasesde.visualizers") -> None:
    """
    Discover and register visualizer plugins via Python entry points.

    Scans the specified entry point group and registers each plugin using its name and target.
    Faulty plugins are ignored; callers can introspect the registry later.

    Parameters
    ----------
    group : str, default "qphasesde.visualizers"
        Entry point group to scan for plugins.

    Returns
    -------
    None

    Examples
    --------
    >>> from QPhaseSDE.visualizers.plugins import register_entry_points
    >>> register_entry_points()
    """
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
    """
    Import and register visualizer plugins from a list of module paths.

    Each module is imported for side effects; it should call visualizers.register.register(...).
    Faulty imports are ignored.

    Parameters
    ----------
    paths : list of str
        List of module paths to import and register.

    Returns
    -------
    None

    Examples
    --------
    >>> from QPhaseSDE.visualizers.plugins import register_from_paths
    >>> register_from_paths(["myplugin.module", "otherplugin.module"])
    """
    for path in paths:
        try:
            # import for side effects; module should call visualizers.register.register(...)
            import_module(path)
        except Exception:
            pass
