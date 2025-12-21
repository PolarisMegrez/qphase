"""qphase: PyTorch Backend with CUDA Support
---------------------------------------------------------
Implements a versatile backend using PyTorch tensors, supporting both CPU and CUDA
execution with automatic device detection. It leverages PyTorch's optimized tensor
operations and device management capabilities to provide a robust, high-performance
execution environment suitable for both research and production workloads.

Public API
----------
``TorchBackend`` : Backend with CPU/CUDA tensor operations
``TorchConfig`` : Configuration with autograd control
``_TorchRNG`` : Internal RNG handle for torch.Generator
``_to_torch_dtype`` : Internal dtype mapping helper
"""

from typing import Any, ClassVar, Literal, cast

import numpy as _np
from pydantic import Field

from qphase.backend.base import BackendBase as Backend
from qphase.backend.base import BackendConfigBase

__all__ = [
    "TorchBackend",
]


class TorchConfig(BackendConfigBase):
    """Configuration for PyTorch backend."""

    enable_grad: bool = Field(default=False, description="Enable autograd")


# Cache for dtype mapping to avoid repeated imports and checks
_DTYPE_MAP: dict[Any, Any] = {}


def _to_torch_dtype(dtype: Any | None):
    """Map common Python/NumPy dtypes to torch dtypes (internal helper).

    Falls back to returning the input dtype when PyTorch is unavailable or the
    dtype is already torch-compatible.
    """
    if dtype is None:
        return None

    # Check cache first
    if dtype in _DTYPE_MAP:
        return _DTYPE_MAP[dtype]

    # Map common Python/NumPy dtypes to torch dtypes
    try:
        import numpy as _nplocal
        import torch as torch

        # Initialize cache if empty
        if not _DTYPE_MAP:
            _DTYPE_MAP.update(
                {
                    complex: torch.complex128,
                    "complex": torch.complex128,
                    float: torch.float64,
                    "float": torch.float64,
                    "float64": torch.float64,
                    int: torch.int64,
                    "int": torch.int64,
                    "int64": torch.int64,
                    bool: torch.bool,
                    "bool": torch.bool,
                    _nplocal.complex128: torch.complex128,
                    _nplocal.complex64: torch.complex64,
                    _nplocal.float64: torch.float64,
                    _nplocal.float32: torch.float32,
                    _nplocal.int64: torch.int64,
                    _nplocal.int32: torch.int32,
                    _nplocal.bool_: torch.bool,
                }
            )

        # Check cache again after initialization
        if dtype in _DTYPE_MAP:
            return _DTYPE_MAP[dtype]

        # Handle numpy dtype objects that might not be in the initial map keys
        # (e.g. np.dtype('float64') vs np.float64 class)
        if isinstance(dtype, _nplocal.dtype):
            if dtype == _nplocal.complex128:
                res = torch.complex128
            elif dtype == _nplocal.complex64:
                res = torch.complex64
            elif dtype == _nplocal.float64:
                res = torch.float64
            elif dtype == _nplocal.float32:
                res = torch.float32
            elif dtype == _nplocal.int64:
                res = torch.int64
            elif dtype == _nplocal.int32:
                res = torch.int32
            elif dtype == _nplocal.bool_:
                res = torch.bool
            else:
                res = cast(Any, dtype)
            _DTYPE_MAP[dtype] = res
            return res

    except Exception:
        # If torch not available at import time, return original dtype
        return dtype

    # Assume it's already a torch dtype or compatible
    return dtype


class _TorchRNG:
    """Internal RNG handle backed by torch.Generator.

    Provides seeding and spawning of independent generators on the selected
    device (CPU or CUDA), used internally by TorchBackend.
    """

    def __init__(self, seed: int | None = None, device: str | None = None):
        try:
            import torch as torch
        except Exception as e:  # pragma: no cover
            raise ImportError("PyTorch is required for TorchBackend") from e
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._gen = torch.Generator(device=self.device)
        if seed is None:
            # derive a seed from entropy
            seed = int(_np.random.SeedSequence().generate_state(1, dtype=_np.uint64)[0])
        self._gen.manual_seed(int(seed))

    def seed(self, value: int | None) -> None:
        if value is None:
            value = int(
                _np.random.SeedSequence().generate_state(1, dtype=_np.uint64)[0]
            )
        self._gen.manual_seed(int(value))

    def spawn(self, n: int) -> list["_TorchRNG"]:
        ss = _np.random.SeedSequence()
        children = ss.spawn(n)
        out: list[_TorchRNG] = []
        for c in children:
            s = int(c.generate_state(1, dtype=_np.uint64)[0])
            out.append(_TorchRNG(s, device=self.device))
        return out

    @property
    def generator(self):  # expose underlying torch.Generator
        return self._gen


class TorchBackend(Backend):
    """PyTorch implementation of the Backend protocol (CPU/CUDA).

    Provides a backend that operates on torch tensors and supports CPU and
    CUDA devices when available. It mirrors the core backend API, including
    array creation/conversion, basic linalg, RNG, and convenience helpers.
    """

    name: ClassVar[str] = "torch"
    description: ClassVar[str] = (
        "PyTorch implementation of the Backend protocol (CPU/CUDA)."
    )
    config_schema: ClassVar[type[TorchConfig]] = TorchConfig

    def __init__(self, config: TorchConfig | None = None, **kwargs: Any) -> None:
        """Initialize Torch backend with configuration."""
        self.config = config or TorchConfig()
        _ = kwargs

    # Identification
    def backend_name(self) -> str:
        return "torch"

    def device(self) -> str | None:
        # Return configured device if set
        if self.config.device:
            return self.config.device

        # Auto-detect
        try:
            import torch as torch

            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                return f"cuda:{idx}"
        except Exception:
            return None
        return "cpu"

    # Array creation / conversion
    def array(self, obj: Any, dtype: Any | None = None) -> Any:
        import torch as torch

        td = _to_torch_dtype(dtype)
        t = torch.as_tensor(obj, dtype=cast(Any, td))
        return t

    def asarray(self, obj: Any, dtype: Any | None = None) -> Any:
        return self.array(obj, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any:
        import torch as torch

        dev = self.device() or "cpu"
        td = _to_torch_dtype(dtype)
        return torch.zeros(*shape, dtype=cast(Any, td), device=dev)

    def empty(self, shape: tuple[int, ...], dtype: Any) -> Any:
        import torch as torch

        dev = self.device() or "cpu"
        td = _to_torch_dtype(dtype)
        return torch.empty(*shape, dtype=cast(Any, td), device=dev)

    def empty_like(self, x: Any) -> Any:
        import torch as torch

        return torch.empty_like(x)

    def copy(self, x: Any) -> Any:
        return x.clone()

    # Ops / linalg
    def einsum(self, subscripts: str, *operands: Any) -> Any:
        import torch as torch

        return torch.einsum(subscripts, *operands)

    def concatenate(self, arrays: tuple[Any, ...], axis: int = -1) -> Any:
        import torch as torch

        return torch.cat(arrays, dim=axis)

    def cholesky(self, a: Any) -> Any:
        import torch as torch

        return torch.linalg.cholesky(a)

    # RNG
    def rng(self, seed: int | None) -> Any:
        return _TorchRNG(seed, device=self.device())

    def randn(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any:
        import torch as torch

        g = cast(_TorchRNG, rng).generator
        dev = self.device() or "cpu"
        t = torch.randn(*shape, generator=g, device=dev)
        td = _to_torch_dtype(dtype)
        return t.to(dtype=cast(Any, td)) if td is not None else t

    def spawn_rngs(self, master_seed: int, n: int) -> list[Any]:
        ss = _np.random.SeedSequence(master_seed)
        children = ss.spawn(n)
        dev = self.device() or "cpu"
        out: list[Any] = []
        for c in children:
            s = int(c.generate_state(1, dtype=_np.uint64)[0])
            out.append(_TorchRNG(s, device=dev))
        return out

    # Complex helpers
    def real(self, x: Any) -> Any:
        import torch as torch

        return torch.real(x)

    def imag(self, x: Any) -> Any:
        import torch as torch

        return torch.imag(x)

    def abs(self, x: Any) -> Any:
        import torch as torch

        return torch.abs(x)

    def mean(self, x: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        import torch as torch

        return torch.mean(x, dim=axis)

    def fft(
        self,
        x: Any,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> Any:
        import torch as torch

        return torch.fft.fft(x, dim=axis, norm=norm)

    def fftfreq(self, n: int, d: float = 1.0) -> Any:
        import torch as torch

        return torch.fft.fftfreq(n, d=d)

    # Convenience for capabilities
    def capabilities(self) -> dict:
        try:
            torch_ok = True
        except Exception:
            torch_ok = False
        return {
            "device": self.device(),
            "optimized_contractions": True,
            "supports_complex_view": True,
            "real_imag_split": True,
            "stack": True,
            "to_device": True,
            "torch": torch_ok,
        }

    # Optional helpers
    def stack(self, arrays: tuple[Any, ...], axis: int = 0) -> Any:
        import torch as torch

        return torch.stack(arrays, dim=axis)

    def to_device(self, x: Any, device: str | None) -> Any:
        if device is None:
            return x
        try:
            return x.to(device)
        except Exception:
            return x
