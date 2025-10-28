"""Backends package.

Registers available backends in the global registry (lazily when possible).
"""

from ..core.registry import register_lazy

# Register NumPy backend lazily
register_lazy("backend", "numpy", "QPhaseSDE.backends.numpy_backend:NumpyBackend", tags=["builtin"], delayed_import=True)
register_lazy("backend", "np", "QPhaseSDE.backends.numpy_backend:NumpyBackend", tags=["builtin"], delayed_import=True)

__all__ = []
