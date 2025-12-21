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
from .cupy_backend import CuPyBackend
from .numba_backend import NumbaBackend
from .numpy_backend import NumpyBackend
from .torch_backend import TorchBackend

__all__ = [
    # Base protocols
    "BackendBase",
    # Implementations
    "NumpyBackend",
    "NumbaBackend",
    "TorchBackend",
    "CuPyBackend",
]
