"""
QPhaseSDE: PyTorch Backend
--------------------------
Backend implementing the Backend protocol with torch tensors for CPU/CUDA.
Designed to be drop-in for the engine and model code that selects array
namespaces dynamically.

Behavior
--------
- Provide torch-based array creation, linalg, RNG, and helpers; device and
    dtype semantics are governed by method-level contracts.

Notes
-----
- Requires PyTorch; CUDA support is used when available.
"""

from typing import Any, Tuple, Optional, List, cast
import numpy as _np
from ..core.protocols import BackendBase as Backend, RNGBase as RNG

__all__ = [
    "TorchBackend",
]

def _to_torch_dtype(dtype: Any | None):
    """Map common Python/NumPy dtypes to torch dtypes (internal helper).

    Falls back to returning the input dtype when PyTorch is unavailable or the
    dtype is already torch-compatible.
    """
    if dtype is None:
        return None
    # Map common Python/NumPy dtypes to torch dtypes
    try:
        import torch as torch  # type: ignore
        import numpy as _nplocal  # type: ignore
        if dtype is complex or str(dtype) == 'complex':
            return torch.complex128
        if dtype is float or str(dtype) in ('float', 'float64'):
            return torch.float64
        if isinstance(dtype, _nplocal.dtype):  # type: ignore[attr-defined]
            if dtype == _nplocal.complex128:
                return torch.complex128
            if dtype == _nplocal.complex64:
                return torch.complex64
            if dtype == _nplocal.float64:
                return torch.float64
            if dtype == _nplocal.float32:
                return torch.float32
    except Exception:
        # If torch not available at import time, return original dtype
        return dtype
    # Assume it's already a torch dtype or compatible
    return dtype

class TorchRNG:
    """Internal RNG handle backed by torch.Generator.

    Provides seeding and spawning of independent generators on the selected
    device (CPU or CUDA), used internally by TorchBackend.
    """
    def __init__(self, seed: Optional[int] = None, device: Optional[str] = None):
        try:
            import torch as torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("PyTorch is required for TorchBackend") from e
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._gen = torch.Generator(device=self.device)
        if seed is None:
            # derive a seed from entropy
            seed = int(_np.random.SeedSequence().generate_state(1, dtype=_np.uint64)[0])
        self._gen.manual_seed(int(seed))

    def seed(self, value: Optional[int]) -> None:
        if value is None:
            value = int(_np.random.SeedSequence().generate_state(1, dtype=_np.uint64)[0])
        self._gen.manual_seed(int(value))

    def spawn(self, n: int) -> list[RNG]:
        ss = _np.random.SeedSequence()
        children = ss.spawn(n)
        out: list[RNG] = []
        for c in children:
            s = int(c.generate_state(1, dtype=_np.uint64)[0])
            out.append(TorchRNG(s, device=self.device))
        return out

    @property
    def generator(self):  # expose underlying torch.Generator
        return self._gen

class TorchBackend(Backend):
    """PyTorch implementation of the Backend protocol (CPU/CUDA).

    Provides a backend that operates on torch tensors and supports CPU and
    CUDA devices when available. It mirrors the core backend API, including
    array creation/conversion, basic linalg, RNG, and convenience helpers.

    Methods
    -------
    backend_name() -> str
        Return backend identifier ("torch").
    device() -> Optional[str]
        Return active device string (e.g., "cpu" or "cuda:{idx}").
    array/asarray/zeros/empty/empty_like/copy
        Array creation and conversion helpers.
    einsum(subscripts, *operands) -> Any
        General tensor contraction via torch.einsum.
    concatenate(arrays, axis) -> Any
        Concatenate tensors along the given axis.
    cholesky(a) -> Any
        Cholesky factorization via torch.linalg.
    rng(seed), spawn_rngs(master_seed, n)
        Create RNG handle and independent random streams.
    randn(rng, shape, dtype) -> Any
        Draw standard normal samples (optionally cast dtype).
    real/imag -> Any
        Complex tensor views for real and imaginary parts.
    stack(arrays, axis), to_device(x, device)
        Optional helpers for batching and device movement.
    capabilities() -> dict
        Report supported features and environment flags.

    Examples
    --------
    >>> be = TorchBackend()
    >>> r = be.rng(0)
    >>> x = be.randn(r, (2, 3), dtype=None)
    >>> x.device.type in ("cpu", "cuda")
    True
    """
    # Identification
    def backend_name(self) -> str:
        return "torch"

    def device(self) -> Optional[str]:
        try:
            import torch as torch  # type: ignore
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                return f"cuda:{idx}"
        except Exception:
            return None
        return "cpu"

    # Array creation / conversion
    def array(self, obj: Any, dtype: Any | None = None) -> Any:
        import torch as torch  # type: ignore
        td = _to_torch_dtype(dtype)
        t = torch.as_tensor(obj, dtype=td)  # type: ignore[arg-type]
        return t

    def asarray(self, obj: Any, dtype: Any | None = None) -> Any:
        return self.array(obj, dtype=dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        import torch as torch  # type: ignore
        dev = self.device() or "cpu"
        td = _to_torch_dtype(dtype)
        return torch.zeros(shape, dtype=td, device=dev)  # type: ignore[arg-type]

    def empty(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        import torch as torch  # type: ignore
        dev = self.device() or "cpu"
        td = _to_torch_dtype(dtype)
        return torch.empty(shape, dtype=td, device=dev)  # type: ignore[arg-type]

    def empty_like(self, x: Any) -> Any:
        import torch as torch  # type: ignore
        return torch.empty_like(x)

    def copy(self, x: Any) -> Any:
        return x.clone()

    # Ops / linalg
    def einsum(self, subscripts: str, *operands: Any) -> Any:
        import torch as torch  # type: ignore
        return torch.einsum(subscripts, *operands)

    def concatenate(self, arrays: Tuple[Any, ...], axis: int = -1) -> Any:
        import torch as torch  # type: ignore
        return torch.cat(arrays, dim=axis)

    def cholesky(self, a: Any) -> Any:
        import torch as torch  # type: ignore
        return torch.linalg.cholesky(a)

    # RNG
    def rng(self, seed: Optional[int]) -> RNG:
        return TorchRNG(seed, device=self.device())

    def randn(self, rng: RNG, shape: Tuple[int, ...], dtype: Any) -> Any:
        import torch as torch  # type: ignore
        g = cast(TorchRNG, rng).generator
        dev = self.device() or "cpu"
        t = torch.randn(shape, generator=g, device=dev)
        td = _to_torch_dtype(dtype)
        return t.to(dtype=td) if td is not None else t  # type: ignore[arg-type]

    def spawn_rngs(self, master_seed: int, n: int) -> List[RNG]:
        ss = _np.random.SeedSequence(master_seed)
        children = ss.spawn(n)
        dev = self.device() or "cpu"
        out: List[RNG] = []
        for c in children:
            s = int(c.generate_state(1, dtype=_np.uint64)[0])
            out.append(TorchRNG(s, device=dev))
        return out

    # Complex helpers
    def real(self, x: Any) -> Any:
        import torch as torch  # type: ignore
        return torch.real(x)

    def imag(self, x: Any) -> Any:
        import torch as torch  # type: ignore
        return torch.imag(x)

    # Convenience for capabilities
    def capabilities(self) -> dict:
        try:
            import torch as _  # type: ignore
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
    def stack(self, arrays: Tuple[Any, ...], axis: int = 0) -> Any:
        import torch as torch  # type: ignore
        return torch.stack(arrays, dim=axis)

    def to_device(self, x: Any, device: Optional[str]) -> Any:
        if device is None:
            return x
        try:
            return x.to(device)
        except Exception:
            return x
