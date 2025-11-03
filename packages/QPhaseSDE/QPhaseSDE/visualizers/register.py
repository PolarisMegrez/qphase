"""
QPhaseSDE: Visualizer registry
------------------------------
Thin helpers that wrap the core registry to register visualizer builders under
the ``visualizer`` namespace.

Behavior
- ``register(name, builder, **meta)``: eager registration of a callable or
    class that constructs a visualizer.
- ``register_lazy(name, target, **meta)``: defer import by dotted path until
    first use.

Notes
- See ``QPhaseSDE.core.registry`` for authoritative error codes and semantics.
"""

from typing import Any, Callable
from ..core.registry import registry as _registry

__all__ = [
    "register",
    "register_lazy",
]

def register(name: str, builder: Callable[..., Any], **meta: Any) -> None:
    """Register a visualizer builder under the ``visualizer`` namespace.

    Parameters
    ----------
    name : str
        Public name of the visualizer (case-insensitive).
    builder : Callable[..., Any]
        Class or function that constructs/returns the visualizer.
    **meta : Any
        Optional metadata stored with the registry entry (e.g., return_callable,
        tags).

    Raises
    ------
    QPSRegistryError
        - [400] Duplicate registration for the same visualizer name.

    Examples
    --------
    >>> from QPhaseSDE.visualizers.register import register
    >>> class MyVisualizer:
    ...     pass
    >>> register("myvisualizer", MyVisualizer)

    See Also
    --------
    QPhaseSDE.core.registry.register : Core decorator-based registration helper.
    """
    _registry.register("visualizer", name, builder, **meta)

def register_lazy(name: str, target: str, **meta: Any) -> None:
    """Register a visualizer by dotted path without importing immediately.

    Parameters
    ----------
    name : str
        Public name of the visualizer (case-insensitive).
    target : str
        Dotted path to the builder, e.g. "pkg.mod:ClassName" or
        "pkg.mod.function_name" (dot form also supported).
    **meta : Any
        Optional metadata stored with the registry entry (e.g., delayed_import).

    Raises
    ------
    QPSRegistryError
        - [401] Duplicate lazy registration for the same visualizer name.

    Examples
    --------
    >>> from QPhaseSDE.visualizers.register import register_lazy
    >>> register_lazy("myvisualizer", "mypkg.visualizers:MyVisualizer")

    See Also
    --------
    QPhaseSDE.core.registry.register_lazy : Core dotted-path lazy registration.
    """
    _registry.register_lazy("visualizer", name, target, **meta)
