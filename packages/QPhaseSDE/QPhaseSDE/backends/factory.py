"""
QPhaseSDE: Backend Factory
--------------------------
Instantiate compute backends by name via the central registry, enabling
runtime selection of NumPy/Numba/CuPy/Torch implementations.

Behavior
--------
- Accept either short names (e.g., "numpy") or fully qualified keys
    (e.g., "backend:numpy"). Resolution and error semantics are governed by
    the central registry; see function docstrings for details.
"""

from typing import Any
from ..core.registry import registry

__all__ = [
    "get_backend",
]

def get_backend(name: str, **kwargs: Any):
    """Instantiate a backend by name via the central registry.

    Parameters
    ----------
    name : str
        Backend identifier. May include the namespace prefix
        (e.g., ``"backend:numpy"``) or just the short name (e.g., ``"numpy"``).
    **kwargs : Any
        Keyword arguments forwarded to the backend constructor.

    Returns
    -------
    BackendBase
        A concrete backend instance registered under the given name.

    Raises
    ------
    QPSConfigError
        Raised for invalid configuration parameters:

        - [404] Unknown registry key. (``namespace:name`` not registered)
    QPSRegistryError
        Raised when the backend import fails:

        - [402] Backend cannot be imported from its dotted path.

    Examples
    --------
    >>> from QPhaseSDE.backends.factory import get_backend
    >>> be = get_backend("numpy")  # or get_backend("backend:numpy")
    >>> be.backend_name()
    'numpy'
    """
    full = name if ":" in name else f"backend:{name}"
    return registry.create(full, **kwargs)
