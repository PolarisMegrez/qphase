"""qphase: Numba JIT-Accelerated Backend
---------------------------------------------------------
Implements a high-performance CPU backend that combines NumPy's versatility with
Numba's JIT compilation. It features specialized, pre-compiled kernels for
performance-critical tensor contractions (e.g., SDE diffusion terms) while falling
back to standard NumPy operations for general tasks, offering a significant speedup
for CPU-bound simulations.

Public API
----------
``NumbaBackend`` : Backend with JIT-accelerated contractions
``NumbaConfig`` : Configuration with cache and fast-math options
``_einsum_tnm_tm_to_tn`` : JIT kernel for (tnm, tm) -> (tn) contractions
``_einsum_tm_mk_to_tk`` : JIT kernel for (tm, mk) -> (tk) contractions
"""

from typing import Any, ClassVar, Literal, cast

import numpy as np
from pydantic import Field

try:
    from numba import njit, prange
except Exception as e:  # pragma: no cover - environments without numba
    raise ImportError(
        "NumbaBackend requires the 'numba' package. Install with `pip install numba`."
    ) from e


from qphase.backend.base import BackendBase as Backend
from qphase.backend.base import BackendConfigBase

__all__ = [
    "NumbaBackend",
]


class NumbaConfig(BackendConfigBase):
    """Configuration for Numba backend."""

    enable_cache: bool = Field(default=True, description="Enable JIT cache")
    fast_math: bool = Field(default=True, description="Enable fast math")


@njit(cache=True, fastmath=True, parallel=True)
def _einsum_tnm_tm_to_tn(L: np.ndarray, dW: np.ndarray) -> np.ndarray:
    """Contract (tnm, tm) -> (tn).

    Parameters
    ----------
    L : ndarray of complex128, shape (T, N, M)
            Coefficients per time step and mode.
    dW : ndarray of float64, shape (T, M)
            Real noise increments per time step.

    Returns
    -------
    ndarray of complex128, shape (T, N)
            Contracted result per time step and mode.

    """
    T, N, M = L.shape
    out = np.empty((T, N), dtype=np.complex128)
    for t in prange(T):
        for n in range(N):
            acc_r = 0.0
            acc_i = 0.0
            for m in range(M):
                c = L[t, n, m]
                w = dW[t, m]
                acc_r += c.real * w
                acc_i += c.imag * w
            out[t, n] = acc_r + 1j * acc_i
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _einsum_tm_mk_to_tk(z: np.ndarray, chol_T: np.ndarray) -> np.ndarray:
    """Contract (tm, mk) -> (tk).

    Parameters
    ----------
    z : ndarray of float64, shape (T, M)
            Real matrix per time step.
    chol_T : ndarray of float64, shape (M, K)
            Cholesky factor transposed.

    Returns
    -------
    ndarray of float64, shape (T, K)
            Contracted result per time step.

    """
    T, M = z.shape
    M2, K = chol_T.shape
    # assert M == M2  # Numba cannot assert; trust caller
    out = np.empty((T, K), dtype=np.float64)
    for t in prange(T):
        for k in range(K):
            acc = 0.0
            for m in range(M):
                acc += z[t, m] * chol_T[m, k]
            out[t, k] = acc
    return out


class _NumbaRNG:
    """Internal RNG handle backed by NumPy Generator.

    Lightweight adapter used by NumbaBackend to satisfy RNGBase semantics
    (seed, spawn) without exposing implementation details.
    """

    def __init__(self, seed: int | np.random.SeedSequence | None):
        if isinstance(seed, np.random.SeedSequence):
            self._seed_seq = seed
        elif seed is None:
            self._seed_seq = np.random.SeedSequence()
        else:
            self._seed_seq = np.random.SeedSequence(seed)

        self._gen = np.random.default_rng(self._seed_seq)

    def generator(self) -> np.random.Generator:
        return self._gen

    def seed(self, value: int | None) -> None:
        if value is None:
            self._seed_seq = np.random.SeedSequence()
        else:
            self._seed_seq = np.random.SeedSequence(value)
        self._gen = np.random.default_rng(self._seed_seq)

    def spawn(self, n: int) -> list["_NumbaRNG"]:
        children = self._seed_seq.spawn(n)
        return [_NumbaRNG(child) for child in children]


# ------------------------------ Backend API ------------------------------


class NumbaBackend(Backend):
    """Numba implementation of the Backend protocol (CPU, optional accel).

    Mirrors the NumPy backend API while routing common hot-path contractions
    through Numba-compiled kernels when available, falling back to NumPy for
    general cases.
    """

    name: ClassVar[str] = "numba"
    description: ClassVar[str] = (
        "Numba implementation of the Backend protocol (CPU with JIT)."
    )
    config_schema: ClassVar[type[NumbaConfig]] = NumbaConfig

    def __init__(self, config: NumbaConfig | None = None, **kwargs: Any) -> None:
        """Initialize Numba backend with configuration."""
        self.config = config or NumbaConfig()
        _ = kwargs

    # Identification
    def backend_name(self) -> str:
        return "numba"

    def device(self) -> str | None:
        return None

    # Array creation / conversion
    def asarray(self, x: Any, dtype: Any | None = None) -> Any:
        return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)

    def array(self, x: Any, dtype: Any | None = None) -> Any:
        return self.asarray(x, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any:
        return np.zeros(shape, dtype=dtype)

    def empty(self, shape: tuple[int, ...], dtype: Any) -> Any:
        return np.empty(shape, dtype=dtype)

    def empty_like(self, x: Any) -> Any:
        return np.empty_like(x)

    def copy(self, x: Any) -> Any:
        return np.copy(x)

    # Ops / linalg
    def einsum(self, subscripts: str, *operands: Any) -> Any:
        # Hot paths: match exact patterns to use Numba kernels
        # TODO: Consider prange parallelization for large T/N in kernels above; the
        # current implementation is single-threaded but compiled. Real gains depend
        # on CPU/BLAS availability and problem sizes. Keep fallbacks to NumPy einsum.
        if subscripts == "tnm,tm->tn" and len(operands) == 2:
            L = operands[0]
            dW = operands[1]
            L_arr = np.asarray(L, dtype=np.complex128)
            dW_arr = np.asarray(dW, dtype=np.float64)
            return _einsum_tnm_tm_to_tn(L_arr, dW_arr)
        if subscripts == "tm,mk->tk" and len(operands) == 2:
            z = operands[0]
            chol_T = operands[1]
            z_arr = np.asarray(z, dtype=np.float64)
            cholT_arr = np.asarray(chol_T, dtype=np.float64)
            return _einsum_tm_mk_to_tk(z_arr, cholT_arr)
        # Fallback to NumPy for general cases
        return np.einsum(subscripts, *operands, optimize=True)

    def cholesky(self, a: Any) -> Any:
        return np.linalg.cholesky(a)

    def rng(self, seed: int | None) -> Any:
        return _NumbaRNG(seed)

    def spawn_rngs(self, master_seed: int, n: int) -> list[Any]:
        ss = np.random.SeedSequence(master_seed)
        children = ss.spawn(n)
        return [_NumbaRNG(child) for child in children]

    def normal(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any:
        nrng = cast(_NumbaRNG, rng)
        g = nrng._gen
        out = g.normal(size=shape)
        return out.astype(dtype if dtype is not None else np.float64, copy=False)

    def randn(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any:
        return self.normal(rng, shape, dtype)

    def real(self, x: Any) -> Any:
        return np.real(x)

    def imag(self, x: Any) -> Any:
        return np.imag(x)

    def abs(self, x: Any) -> Any:
        return np.abs(x)

    def mean(self, x: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        return np.mean(x, axis=axis)

    def fft(
        self,
        x: Any,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> Any:
        return np.fft.fft(x, axis=axis, norm=norm)

    def fftfreq(self, n: int, d: float = 1.0) -> Any:
        return np.fft.fftfreq(n, d=d)

    def concatenate(self, arrays: tuple[Any, ...], axis: int = -1) -> Any:
        return np.concatenate(arrays, axis=axis)

    # Optional helpers
    def stack(self, arrays: tuple[Any, ...], axis: int = 0) -> Any:
        return np.stack(arrays, axis=axis)

    def to_device(self, x: Any, device: str | None) -> Any:
        return x

    # Capabilities
    def capabilities(self) -> dict:
        return {
            "device": None,
            "optimized_contractions": True,
            "supports_complex_view": False,
            "real_imag_split": True,
            "stack": True,
            "to_device": False,
            "numba": True,
            "numpy": True,
        }
