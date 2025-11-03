"""
QPhaseSDE: NumPy Backend
------------------------
Reference CPU backend implementing the Backend protocol with NumPy arrays and
linear algebra routines; serves as the compatibility baseline for other backends.

Behavior
--------
- Provide NumPy-based array creation, basic linalg, RNG, and helpers consistent
    with the Backend protocol; semantics are documented by class/method docs.
"""

from typing import Any, Tuple, cast, List, Optional
import numpy as np
from ..core.protocols import BackendBase as Backend, RNGBase as RNG

__all__ = [
	"NumpyBackend",
]

class NumpyRNG:
    """Internal RNG handle backed by NumPy Generator.

    Lightweight adapter used by NumpyBackend to satisfy RNGBase semantics
    (seed, spawn) without exposing implementation details.
    """

    def __init__(self, seed: int | None):
        self._gen = np.random.default_rng(seed)

    def generator(self) -> np.random.Generator:
        return self._gen

    def seed(self, value: Optional[int]) -> None:
        self._gen = np.random.default_rng(value)

    def spawn(self, n: int) -> list[RNG]:
        ss = np.random.SeedSequence(self._gen.bit_generator._seed_seq.entropy)  # type: ignore[attr-defined]
        children = ss.spawn(n)
        result: list[RNG] = []
        for child in children:
            gen = np.random.Generator(np.random.PCG64(child))
            r = NumpyRNG(None)
            r._gen = gen
            result.append(r)
        return result

class NumpyBackend(Backend):
    """NumPy implementation of the Backend protocol (CPU only).

    Provides a reference CPU backend using NumPy arrays and routines. It mirrors
    the common backend API (array creation, basic linalg, RNG, and utilities)
    without JIT acceleration; all operations are executed by NumPy.

    Methods
    -------
    backend_name() -> str
        Return backend identifier ("numpy").
    device() -> Optional[str]
        Return device (None for CPU).
    asarray/array/zeros/empty/empty_like/copy
        Array creation and conversion helpers.
    einsum(subscripts, *operands) -> Any
        General tensor contraction via NumPy's einsum (optimize=True).
    cholesky(a) -> Any
        Cholesky factorization via NumPy.
    rng(seed), spawn_rngs(master_seed, n)
        Create RNG handle and independent streams.
    normal(...), randn(...)
        Draw samples from the (standard) normal distribution.
    real/imag/abs/concatenate/stack/to_device/capabilities
        Convenience and introspection helpers.

    Examples
    --------
    >>> be = NumpyBackend()
    >>> r = be.rng(0)
    >>> x = be.randn(r, (2, 3), dtype=None)
    >>> x.shape
    (2, 3)
    """
    # Identification
    def backend_name(self) -> str:
        return "numpy"

    def device(self) -> Optional[str]:
        return None

    def asarray(self, x: Any, dtype: Any | None = None) -> Any:
        return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)

    def array(self, x: Any, dtype: Any | None = None) -> Any:
        return self.asarray(x, dtype=dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        return np.zeros(shape, dtype=dtype)

    def empty(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        return np.empty(shape, dtype=dtype)

    def empty_like(self, x: Any) -> Any:
        return np.empty_like(x)

    def copy(self, x: Any) -> Any:
        return np.copy(x)

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        # Enable NumPy's contraction path optimization for potential speedups
        return np.einsum(subscripts, *operands, optimize=True)

    def cholesky(self, a: Any) -> Any:
        return np.linalg.cholesky(a)

    def rng(self, seed: int | None) -> RNG:
        return NumpyRNG(seed)

    def spawn_rngs(self, master_seed: int, n: int) -> List[RNG]:
        ss = np.random.SeedSequence(master_seed)
        children = ss.spawn(n)
        rngs: List[RNG] = []
        for child in children:
            # Seed Generator from child SeedSequence for deterministic stream
            gen = np.random.Generator(np.random.PCG64(child))
            nrng = NumpyRNG(None)
            # overwrite internal generator
            nrng._gen = gen
            rngs.append(nrng)
        return rngs

    def normal(self, rng: RNG, shape: Tuple[int, ...], dtype: Any) -> Any:
        # rng is expected to come from this backend's rng(); cast for type checkers
        nrng = cast(NumpyRNG, rng)
        g = nrng._gen
        out = g.normal(size=shape)
        return out.astype(dtype if dtype is not None else np.float64, copy=False)

    # Alias for minimal protocol naming
    def randn(self, rng: RNG, shape: Tuple[int, ...], dtype: Any) -> Any:
        return self.normal(rng, shape, dtype)

    def real(self, x: Any) -> Any:
        return np.real(x)

    def imag(self, x: Any) -> Any:
        return np.imag(x)

    def abs(self, x: Any) -> Any:
        return np.abs(x)

    def concatenate(self, arrays: Tuple[Any, ...], axis: int = -1) -> Any:
        return np.concatenate(arrays, axis=axis)

    # Optional helpers
    def stack(self, arrays: Tuple[Any, ...], axis: int = 0) -> Any:
        return np.stack(arrays, axis=axis)

    def to_device(self, x: Any, device: Optional[str]) -> Any:
        return x

    # Capabilities
    def capabilities(self) -> dict:
        return {
            "device": None,
            "optimized_contractions": False,
            "supports_complex_view": False,
            "real_imag_split": True,
            "stack": True,
            "to_device": False,
            "numpy": True,
        }
