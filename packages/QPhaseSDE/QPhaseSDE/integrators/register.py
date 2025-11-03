"""
QPhaseSDE: Integrator Registry Helpers
-------------------------------------
Thin wrappers around the central registry for registering integrator builders
and dotted-path entries under the 'integrator' namespace.

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
    """Register an integrator builder under the ``integrator`` namespace.

    Parameters
    ----------
    name : str
        Public name of the integrator (case-insensitive).
    builder : Callable[..., Any]
        Class or function that constructs/returns the integrator.
    **meta : Any
        Optional metadata stored with the registry entry (e.g., return_callable,
        tags).

    Raises
    ------
    QPSRegistryError
        - [400] Duplicate registration for the same integrator name.

    Examples
    --------
    >>> from QPhaseSDE.integrators.register import register
    >>> class MyIntegrator:
    ...     pass
    >>> register("myintegrator", MyIntegrator)

    See Also
    --------
    QPhaseSDE.core.registry.register : Core decorator-based registration helper.
    """
    _registry.register("integrator", name, builder, **meta)

def register_lazy(name: str, target: str, **meta: Any) -> None:
    """Register an integrator by dotted path without importing immediately.

    Parameters
    ----------
    name : str
        Public name of the integrator (case-insensitive).
    target : str
        Dotted path to the builder, e.g. "pkg.mod:ClassName" or
        "pkg.mod.function_name" (dot form also supported).
    **meta : Any
        Optional metadata stored with the registry entry (e.g., delayed_import).

    Raises
    ------
    QPSRegistryError
        - [401] Duplicate lazy registration for the same integrator name.

    Examples
    --------
    >>> from QPhaseSDE.integrators.register import register_lazy
    >>> register_lazy("myintegrator", "mypkg.integrators:MyIntegrator")

    See Also
    --------
    QPhaseSDE.core.registry.register_lazy : Core dotted-path lazy registration.
    """
    _registry.register_lazy("integrator", name, target, **meta)
