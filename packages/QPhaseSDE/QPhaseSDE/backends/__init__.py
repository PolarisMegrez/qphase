"""
QPhaseSDE: Backends Subpackage
------------------------------
Compute backends that implement a minimal, unified array/RNG interface across
CPU/GPU. Provide NumPy, Numba, CuPy, and PyTorch implementations for portable
performance.

Usage
-----
Registry keys:
`backend:numpy` | `backend:numba` | `backend:cupy` | `backend:torch`

Factory:
>>> from QPhaseSDE.backends.factory import get_backend;
>>> be = get_backend("numpy")

Notes
-----
- Backends are registered lazily to avoid importing heavy dependencies until
  needed.
"""

from ..core.registry import register_lazy

# Register NumPy backend lazily
register_lazy("backend", "numpy", "QPhaseSDE.backends.numpy_backend:NumpyBackend", tags=["builtin"], delayed_import=True)
register_lazy("backend", "np", "QPhaseSDE.backends.numpy_backend:NumpyBackend", tags=["builtin"], delayed_import=True)

# Register Numba backend lazily (optional acceleration)
register_lazy("backend", "numba", "QPhaseSDE.backends.numba_backend:NumbaBackend", tags=["builtin", "numba"], delayed_import=True)
register_lazy("backend", "nb", "QPhaseSDE.backends.numba_backend:NumbaBackend", tags=["builtin", "numba"], delayed_import=True)

# Register CuPy backend lazily (experimental GPU acceleration)
register_lazy("backend", "cupy", "QPhaseSDE.backends.cupy_backend:CuPyBackend", tags=["builtin", "cupy", "experimental"], delayed_import=True)
register_lazy("backend", "cp", "QPhaseSDE.backends.cupy_backend:CuPyBackend", tags=["builtin", "cupy", "experimental"], delayed_import=True)

# Register PyTorch backend lazily (CPU/CUDA)
register_lazy("backend", "torch", "QPhaseSDE.backends.torch_backend:TorchBackend", tags=["builtin", "torch"], delayed_import=True)
register_lazy("backend", "pt", "QPhaseSDE.backends.torch_backend:TorchBackend", tags=["builtin", "torch"], delayed_import=True)

__all__ = []
