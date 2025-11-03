"""
QPhaseSDE: Backend Registry Helpers
-----------------------------------
Thin wrappers around the central registry for registering backend builders
and dotted-path entries under the 'backend' namespace.

Behavior
--------
- Forward registrations to the core registry with the appropriate namespace;
    error semantics follow the core registry functions.
"""

from typing import Any, Callable
from ..core.registry import registry as _registry

__all__ = [
    "register",
    "register_lazy",
]

def register(name: str, builder: Callable[..., Any], **meta: Any) -> None:
    """Register a backend builder under the backend namespace.

    Parameters
    ----------
    name : str
        Public name of the backend (case-insensitive).
    builder : Callable[..., Any]
        Class or function that constructs/returns the backend.
    **meta : Any
        Optional metadata stored with the registry entry (e.g., return_callable,
        tags).

    Raises
    ------
    QPSRegistryError
        - [400] Duplicate registration for the same backend name.

    Examples
    --------
    >>> from QPhaseSDE.backends.register import register
    >>> class MyBackend:
    ...     pass
    >>> register("mybackend", MyBackend)

    See Also
    --------
    QPhaseSDE.core.registry.register : Core decorator-based registration helper.
    """
    _registry.register("backend", name, builder, **meta)

def register_lazy(name: str, target: str, **meta: Any) -> None:
    """Register a backend by dotted path without importing immediately.

    Parameters
    ----------
    name : str
        Public name of the backend (case-insensitive).
    target : str
        Dotted path to the builder, e.g. "pkg.mod:ClassName" or
        "pkg.mod.function_name" (dot form also supported).
    **meta : Any
        Optional metadata stored with the registry entry (e.g., delayed_import).

    Raises
    ------
    QPSRegistryError
        - [401] Duplicate lazy registration for the same backend name.

    Examples
    --------
    >>> from QPhaseSDE.backends.register import register_lazy
    >>> register_lazy("mybackend", "mypkg.backends:MyBackend")

    See Also
    --------
    QPhaseSDE.core.registry.register_lazy : Core dotted-path lazy registration.
    """
    _registry.register_lazy("backend", name, target, **meta)