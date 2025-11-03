"""
QPhaseSDE: Noise Model Registry Helpers
--------------------------------------
Lightweight helpers that forward noise model registrations to the core
registry under the 'noise_model' namespace.

Behavior
--------
- Mirror core registry semantics for errors and dotted-path resolution.
"""

from typing import Any, Callable
from ..core.registry import registry as _registry

__all__ = [
    "register",
    "register_lazy",
]

def register(name: str, builder: Callable[..., Any], **meta: Any) -> None:
    """Register an noise_model builder under the ``noise_model`` namespace.

    Parameters
    ----------
    name : str
        Public name of the noise_model (case-insensitive).
    builder : Callable[..., Any]
        Class or function that constructs/returns the noise_model.
    **meta : Any
        Optional metadata stored with the registry entry (e.g., return_callable,
        tags).

    Raises
    ------
    QPSRegistryError
        - [400] Duplicate registration for the same noise_model name.

    Examples
    --------
    >>> from QPhaseSDE.noise_models.register import register
    >>> class MyNoise_model:
    ...     pass
    >>> register("mynoise_model", MyNoise_model)

    See Also
    --------
    QPhaseSDE.core.registry.register : Core decorator-based registration helper.
    """
    _registry.register("noise_model", name, builder, **meta)

def register_lazy(name: str, target: str, **meta: Any) -> None:
    """Register an noise_model by dotted path without importing immediately.

    Parameters
    ----------
    name : str
        Public name of the noise_model (case-insensitive).
    target : str
        Dotted path to the builder, e.g. "pkg.mod:ClassName" or
        "pkg.mod.function_name" (dot form also supported).
    **meta : Any
        Optional metadata stored with the registry entry (e.g., delayed_import).

    Raises
    ------
    QPSRegistryError
        - [401] Duplicate lazy registration for the same noise_model name.

    Examples
    --------
    >>> from QPhaseSDE.noise_models.register import register_lazy
    >>> register_lazy("mynoise_model", "mypkg.noise_models:MyNoise_model")

    See Also
    --------
    QPhaseSDE.core.registry.register_lazy : Core dotted-path lazy registration.
    """
    _registry.register_lazy("noise_model", name, target, **meta)
