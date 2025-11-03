"""
QPhaseSDE: CuPy Backend
-----------------------
GPU backend implementing the Backend protocol using CuPy arrays. Intended for
experimental acceleration on NVIDIA GPUs while preserving backend-agnostic
interfaces for the engine and domain code.

Behavior
--------
- Provide minimal NumPy-like array/RNG operations on device via CuPy; details
    of array semantics and RNG contracts are documented by class/method docs.

Notes
-----
- Requires CuPy; treat as experimental. Interop with NumPy-based containers may
    involve host/device transfers depending on the calling layer.
"""

from typing import Any, Tuple, cast, List, Optional

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except Exception:  # pragma: no cover
    _CUPY_AVAILABLE = False
    # Create a tiny shim so import errors are explicit when used
    class _CPShim:  # pragma: no cover
        def __getattr__(self, name):
            raise ImportError("cupy is required for the CuPy backend")
    cp = _CPShim()  # type: ignore

import numpy as np
from ..core.protocols import BackendBase as Backend, RNGBase as RNG

__all__ = [
    "CuPyBackend",
]

class CuPyRNG:
    """CuPy-backed random number generator handle (internal).

    Lightweight wrapper used by CuPyBackend to provide RNGBase-compatible
    seeding and stream spawning semantics.
    """
    def __init__(self, seed: int | None):
        # Prefer NumPy SeedSequence for portability of seeds
        if seed is None:
            self._rs = cp.random.RandomState()
        else:
            self._rs = cp.random.RandomState(int(seed))

    def generator(self):  # CuPy RandomState (legacy API)
        return self._rs

    def seed(self, value: Optional[int]) -> None:
        if value is None:
            self._rs = cp.random.RandomState()
        else:
            self._rs = cp.random.RandomState(int(value))

    def spawn(self, n: int) -> list[RNG]:
        # Use NumPy SeedSequence to derive stable integer seeds
        import numpy as _np
        ss = _np.random.SeedSequence()
        children = ss.spawn(n)
        result: list[RNG] = []
        for child in children:
            s = int(child.generate_state(1, dtype=_np.uint64)[0])
            r = CuPyRNG(s)
            result.append(r)
        return result


class CuPyBackend(Backend):
    """CuPy implementation of the Backend protocol (experimental).

    Provides minimal, NumPy-like array/RNG operations on the GPU via CuPy so
    the engine and domains can remain backend-agnostic. Designed to satisfy
    the BackendBase contract; some optional helpers are provided when useful.

    Methods
    -------
    backend_name() -> str
        Return backend identifier ("cupy").
    device() -> Optional[str]
        Return device string (e.g., "cuda:0" when detectable).
    asarray(obj, dtype=None) -> Any
        Convert input to a CuPy array.
    array(obj, dtype=None) -> Any
        Alias of asarray.
    zeros(shape, dtype) / empty(shape, dtype) / empty_like(x)
        Array creation helpers on device.
    copy(x) -> Any
        Deep copy on device.
    einsum(subscripts, *operands) -> Any
        Contract arrays with optional optimization.
    cholesky(a) -> Any
        Cholesky factorization via cupy.linalg.
    rng(seed) -> RNGBase
        Create a RNG handle; spawn_rngs(master_seed, n) -> list[RNGBase].
    randn(rng, shape, dtype) -> Any
        Standard normal samples on device.
    real(x)/imag(x)/abs(x)
        Complex helpers.
    concatenate(arrays, axis=-1)/stack(arrays, axis=0)
        Joining helpers on device.
    to_device(x, device)
        No-op; arrays already on device.
    capabilities() -> dict
        Report capabilities and flags for feature detection.

    Examples
    --------
    >>> be = CuPyBackend()
    >>> r = be.rng(1234)
    >>> z = be.randn(r, (2, 3), dtype=None)
    >>> z.shape
    (2, 3)
    """

    # Identification
    def backend_name(self) -> str:
        return "cupy"

    def device(self) -> Optional[str]:
        try:
            dev = cp.cuda.runtime.getDevice()  # type: ignore[attr-defined]
            return f"cuda:{dev}"
        except Exception:
            return "cuda"

    # Array creation / conversion
    def asarray(self, x: Any, dtype: Any | None = None) -> Any:
        return cp.asarray(x, dtype=dtype) if dtype is not None else cp.asarray(x)

    def array(self, x: Any, dtype: Any | None = None) -> Any:
        return self.asarray(x, dtype=dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        return cp.zeros(shape, dtype=dtype)

    def empty(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        return cp.empty(shape, dtype=dtype)

    def empty_like(self, x: Any) -> Any:
        return cp.empty_like(x)

    def copy(self, x: Any) -> Any:
        return cp.array(x, copy=True)

    # Ops / linalg
    def einsum(self, subscripts: str, *operands: Any) -> Any:
        return cp.einsum(subscripts, *operands, optimize=True)

    def cholesky(self, a: Any) -> Any:
        return cp.linalg.cholesky(a)

    def rng(self, seed: int | None) -> RNG:
        return CuPyRNG(seed)

    def spawn_rngs(self, master_seed: int, n: int) -> List[RNG]:
        # Use NumPy SeedSequence to generate independent seeds
        ss = np.random.SeedSequence(master_seed)
        children = ss.spawn(n)
        rngs: List[RNG] = []
        for child in children:
            s = int(child.generate_state(1, dtype=np.uint64)[0])
            rngs.append(CuPyRNG(s))
        return rngs

    def normal(self, rng: RNG, shape: Tuple[int, ...], dtype: Any) -> Any:
        rr = cast(CuPyRNG, rng)
        out = rr._rs.standard_normal(size=shape)
        # Cast on device if needed
        return out.astype(dtype if dtype is not None else cp.float64, copy=False)

    def randn(self, rng: RNG, shape: Tuple[int, ...], dtype: Any) -> Any:
        return self.normal(rng, shape, dtype)

    def real(self, x: Any) -> Any:
        return cp.real(x)

    def imag(self, x: Any) -> Any:
        return cp.imag(x)

    def abs(self, x: Any) -> Any:
        return cp.abs(x)

    def concatenate(self, arrays: Tuple[Any, ...], axis: int = -1) -> Any:
        return cp.concatenate(arrays, axis=axis)

    # Optional helpers
    def stack(self, arrays: Tuple[Any, ...], axis: int = 0) -> Any:
        return cp.stack(arrays, axis=axis)

    def to_device(self, x: Any, device: Optional[str]) -> Any:
        return x  # cp arrays already on device

    # Capabilities
    def capabilities(self) -> dict:
        return {
            "device": self.device(),
            "optimized_contractions": True,
            "supports_complex_view": False,
            "real_imag_split": True,
            "stack": True,
            "to_device": True,
            "cupy": _CUPY_AVAILABLE,
        }