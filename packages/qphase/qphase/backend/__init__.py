"""qphase: backend subpackage
---------------------------------------------------------
This subpackage defines the abstract computation layer that decouples high-level
algorithms from specific numerical libraries. It provides a unified ``BackendBase``
protocol for array manipulation, linear algebra, random number generation, and FFT
operations, allowing simulations to run seamlessly on different hardware and
software stacks.

Public API
----------
``BackendBase`` : Protocol defining the backend interface contract
``NumpyBackend`` : Reference CPU backend using NumPy
``NumbaBackend`` : JIT-accelerated CPU backend using Numba
``TorchBackend`` : PyTorch backend with CPU/CUDA support
``CuPyBackend`` : GPU-only backend using CuPy for CUDA acceleration
"""

from .base import BackendBase
from .numpy_backend import NumpyBackend

__all__ = [
    # Base protocols
    "BackendBase",
    # Implementations
    "NumpyBackend",
]

# Optional backends are exposed lazily so that ``import qphase.backend`` works on
# a minimal numpy + CLI install. The entry-point registry still discovers all
# backends declared in pyproject.toml regardless of these imports.
try:
    from .numba_backend import NumbaBackend  # noqa: F401
except ImportError:  # pragma: no cover - numba not installed
    pass
else:
    __all__.append("NumbaBackend")

try:
    from .torch_backend import TorchBackend  # noqa: F401
except ImportError:  # pragma: no cover - torch not installed
    pass
else:
    __all__.append("TorchBackend")

try:
    from .cupy_backend import CuPyBackend  # noqa: F401
except ImportError:  # pragma: no cover - cupy not installed
    pass
else:
    __all__.append("CuPyBackend")
