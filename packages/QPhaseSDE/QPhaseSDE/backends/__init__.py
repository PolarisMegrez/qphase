"""
QPhaseSDE: Backends Subpackage
------------------------------
Compute backends that implement a minimal, unified array/RNG interface across
CPU/GPU. Registrations are lazy to avoid importing heavy deps until needed.

Registry keys
-------------
`backend:numpy` | `backend:numba` | `backend:cupy` | `backend:torch`

Factory
-------
>>> from QPhaseSDE.backends.factory import get_backend
>>> be = get_backend("numpy")
"""

from ..core.registry import namespaced
register, register_lazy = namespaced("backend")

# Register NumPy backend lazily
register_lazy("numpy", "QPhaseSDE.backends.numpy_backend:NumpyBackend", tags=["builtin"], delayed_import=True)
register_lazy("np", "QPhaseSDE.backends.numpy_backend:NumpyBackend", tags=["builtin"], delayed_import=True)

# Register Numba backend lazily (optional acceleration)
register_lazy("numba", "QPhaseSDE.backends.numba_backend:NumbaBackend", tags=["builtin", "numba"], delayed_import=True)
register_lazy("nb", "QPhaseSDE.backends.numba_backend:NumbaBackend", tags=["builtin", "numba"], delayed_import=True)

# Register CuPy backend lazily (experimental GPU acceleration)
register_lazy("cupy", "QPhaseSDE.backends.cupy_backend:CuPyBackend", tags=["builtin", "cupy", "experimental"], delayed_import=True)
register_lazy("cp", "QPhaseSDE.backends.cupy_backend:CuPyBackend", tags=["builtin", "cupy", "experimental"], delayed_import=True)

# Register PyTorch backend lazily (CPU/CUDA)
register_lazy("torch", "QPhaseSDE.backends.torch_backend:TorchBackend", tags=["builtin", "torch"], delayed_import=True)
register_lazy("pt", "QPhaseSDE.backends.torch_backend:TorchBackend", tags=["builtin", "torch"], delayed_import=True)

__all__ = []
