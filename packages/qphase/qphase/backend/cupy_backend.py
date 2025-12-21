"""qphase: CuPy GPU Backend (Experimental)
---------------------------------------------------------
Implements the experimental GPU-only backend using CuPy for NVIDIA CUDA acceleration.
It provides a NumPy-compatible interface for array operations directly on the GPU,
implementing the full ``BackendBase`` protocol to enable high-performance simulations
on CUDA-capable hardware.

Public API
----------
``CuPyBackend`` : GPU backend with CuPy array operations
``CuPyConfig`` : Backend configuration (inherits device/dtype settings)
``_CuPyRNG`` : Internal RNG handle for CuPy RandomState

Notes
-----
- GPU-only backend with NumPy-like API via CuPy
- Full backend protocol with GPU-optimized operations
- Requires CUDA-capable GPU and 'cupy' package

"""

from typing import Any, ClassVar, Literal, cast

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except Exception:  # pragma: no cover
    _CUPY_AVAILABLE = False

    # Create a tiny shim so import errors are explicit when used
    class _CPShim:  # pragma: no cover
        def __getattr__(self, name):
            raise ImportError("cupy is required for the CuPy backend")

    cp = _CPShim()

import numpy as np

from qphase.backend.base import BackendBase as Backend
from qphase.backend.base import BackendConfigBase

__all__ = [
    "CuPyBackend",
]


class CuPyConfig(BackendConfigBase):
    """Configuration for CuPy backend."""

    pass  # device 和 float_dtype 来自 BackendConfigBase


class _CuPyRNG:
    """CuPy-backed random number generator handle (internal).

    Lightweight wrapper used by CuPyBackend to provide RNGBase-compatible
    seeding and stream spawning semantics using the modern Generator API.
    """

    def __init__(self, seed: int | None):
        # Use modern Generator API (CuPy >= 9.0)
        if seed is None:
            self._gen = cp.random.default_rng()
        else:
            self._gen = cp.random.default_rng(int(seed))

    def generator(self):
        return self._gen

    def seed(self, value: int | None) -> None:
        if value is None:
            self._gen = cp.random.default_rng()
        else:
            self._gen = cp.random.default_rng(int(value))

    def spawn(self, n: int) -> list["_CuPyRNG"]:
        # Use NumPy SeedSequence to derive stable integer seeds
        # CuPy's BitGenerator doesn't support spawn() directly yet in all versions,
        # so we use NumPy's SeedSequence to generate seeds for new CuPy generators.
        import numpy as _np

        # Extract current state or seed if possible, but for simplicity and robustness
        # we create a new SeedSequence. In a rigorous implementation, we should
        # track the SeedSequence.
        ss = _np.random.SeedSequence()
        children = ss.spawn(n)
        result: list[_CuPyRNG] = []
        for child in children:
            s = int(child.generate_state(1, dtype=_np.uint64)[0])
            r = _CuPyRNG(s)
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
    rng(seed) -> Any
        Create a RNG handle; spawn_rngs(master_seed, n) -> list[Any].
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

    name: ClassVar[str] = "cupy"
    description: ClassVar[str] = (
        "CuPy implementation of the Backend protocol (GPU only)."
    )
    config_schema: ClassVar[type[CuPyConfig]] = CuPyConfig

    def __init__(self, config: CuPyConfig | None = None, **kwargs: Any) -> None:
        """Initialize CuPy backend with configuration."""
        self.config = config or CuPyConfig()
        _ = kwargs

    # Identification
    def backend_name(self) -> str:
        return "cupy"

    def device(self) -> str | None:
        # Return configured device if set, otherwise current device
        if self.config.device:
            return self.config.device
        try:
            dev = cp.cuda.runtime.getDevice()
            return f"cuda:{dev}"
        except Exception:
            return "cuda"

    # Array creation / conversion
    def asarray(self, x: Any, dtype: Any | None = None) -> Any:
        return cp.asarray(x, dtype=dtype) if dtype is not None else cp.asarray(x)

    def array(self, x: Any, dtype: Any | None = None) -> Any:
        return self.asarray(x, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any:
        return cp.zeros(shape, dtype=dtype)

    def empty(self, shape: tuple[int, ...], dtype: Any) -> Any:
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

    def rng(self, seed: int | None) -> Any:
        return _CuPyRNG(seed)

    def spawn_rngs(self, master_seed: int, n: int) -> list[Any]:
        # Use NumPy SeedSequence to generate independent seeds
        ss = np.random.SeedSequence(master_seed)
        children = ss.spawn(n)
        rngs: list[Any] = []
        for child in children:
            s = int(child.generate_state(1, dtype=np.uint64)[0])
            rngs.append(_CuPyRNG(s))
        return rngs

    def normal(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any:
        rr = cast(_CuPyRNG, rng)
        out = rr._gen.standard_normal(size=shape)
        # Cast on device if needed
        return out.astype(dtype if dtype is not None else cp.float64, copy=False)

    def randn(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any:
        return self.normal(rng, shape, dtype)

    def real(self, x: Any) -> Any:
        return cp.real(x)

    def imag(self, x: Any) -> Any:
        return cp.imag(x)

    def abs(self, x: Any) -> Any:
        return cp.abs(x)

    def mean(self, x: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        return cp.mean(x, axis=axis)

    def fft(
        self,
        x: Any,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> Any:
        return cp.fft.fft(x, axis=axis, norm=norm)

    def fftfreq(self, n: int, d: float = 1.0) -> Any:
        return cp.fft.fftfreq(n, d=d)

    def concatenate(self, arrays: tuple[Any, ...], axis: int = -1) -> Any:
        return cp.concatenate(arrays, axis=axis)

    # Optional helpers
    def stack(self, arrays: tuple[Any, ...], axis: int = 0) -> Any:
        return cp.stack(arrays, axis=axis)

    def to_device(self, x: Any, device: str | None) -> Any:
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
