"""qphase_sde: Lightweight Registry System
-----------------------------------------

Independent registry system for qphase_sde.
This provides a minimal registration mechanism that is fully compatible
with the qphase registry but operates independently.

Design Philosophy
-----------------
- Lightweight and focused on SDE-specific needs
- Compatible with existing @register decorations
- No dependency on qphase
- Simple and maintainable

Registry Structure
------------------
The registry is organized by namespaces and keys:

    registry.register(namespace, key)(func)
    registry.get_loader(namespace, key)(*args, **kwargs)

Example:
-------
    from qphase_sde.core.registry import register_loader

    @register_loader("loader", "sde_result")
    def load_sde_result(path):
        ...

"""

from collections.abc import Callable
from typing import Any

# Global registry instance
_registry: dict[str, dict[str, Callable[..., Any]]] = {}


def register_loader(
    namespace: str, key: str
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a loader function.

    This decorator is compatible with qphase's register decorator.

    Parameters
    ----------
    namespace : str
        Namespace for the loader (e.g., "loader", "serializer").
    key : str
        Key within the namespace (e.g., "sde_result").

    Returns
    -------
    Callable
        Decorator function that registers the loader.

    Examples
    --------
    >>> @register_loader("loader", "sde_result")
    ... def load_result(path):
    ...     return SDEResult.load(path)

    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Initialize namespace if it doesn't exist
        if namespace not in _registry:
            _registry[namespace] = {}

        # Register the function
        _registry[namespace][key] = func

        return func

    return decorator


def get_loader(namespace: str, key: str) -> Callable[..., Any] | None:
    """Get a registered loader function.

    Parameters
    ----------
    namespace : str
        Namespace to search in.
    key : str
        Key to look up.

    Returns
    -------
    Callable | None
        Registered function if found, None otherwise.

    Examples
    --------
    >>> loader = get_loader("loader", "sde_result")
    >>> if loader:
    ...     result = loader(path)

    """
    return _registry.get(namespace, {}).get(key)


def namespaced(namespace: str):
    """Create register helpers for a specific namespace.

    Parameters
    ----------
    namespace : str
        The namespace to bind the helpers to.

    Returns
    -------
    tuple
        (register, register_lazy) helpers.

    """

    def register(key: str, obj: Any = None):
        if obj is None:
            return register_loader(namespace, key)
        else:
            return register_loader(namespace, key)(obj)

    def register_lazy(key: str, module_path: str, object_name: str):
        # Placeholder for lazy registration compatibility
        pass

    return register, register_lazy


def list_loaders(
    namespace: str | None = None,
) -> dict[str, dict[str, Callable[..., Any]]]:
    """List all registered loaders.

    Parameters
    ----------
    namespace : str | None, optional
        If provided, only list loaders in this namespace.
        If None, list all loaders across all namespaces.

    Returns
    -------
    dict
        Dictionary of {namespace: {key: function}} entries.

    Examples
    --------
    >>> all_loaders = list_loaders()
    >>> loader_namespace = list_loaders("loader")

    """
    if namespace is None:
        return _registry.copy()
    return {namespace: _registry.get(namespace, {})}


def register(
    namespace: str, key: str
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function as a plugin loader.

    This provides compatibility with existing code that uses
    the generic register decorator from qphase.

    Parameters
    ----------
    namespace : str
        Namespace for the registration.
    key : str
        Key within the namespace.

    Returns
    -------
    Callable
        Decorator function that registers the function.

    """
    return register_loader(namespace, key)


# For backward compatibility with qphase-style imports
# This allows existing code to import 'register' from qphase_sde.core.registry
# and it will work seamlessly
__all__ = [
    "register_loader",
    "get_loader",
    "list_loaders",
    "register",
]
