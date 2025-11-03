"""
QPhaseSDE: Numba Backend
------------------------
CPU backend mirroring the NumPy API while JIT-accelerating common contraction
patterns with Numba for improved performance on hot paths.

Behavior
--------
- Provide a NumPy-compatible Backend protocol implementation; selected
	einsum patterns are compiled with Numba (e.g., 'tnm,tm->tn', 'tm,mk->tk').
	Fallback to NumPy for general cases; method-level details govern semantics.

Notes
-----
- Requires Numba. Importing this module without Numba installed raises an
	ImportError.
"""

from typing import Any, Tuple, cast, List, Optional
import numpy as np

try:
	from numba import njit, prange  # prange reserved for future kernels
except Exception as e:  # pragma: no cover - environments without numba
	raise ImportError(
		"NumbaBackend requires the 'numba' package. Install with `pip install numba`."
	) from e

from ..core.protocols import BackendBase as Backend, RNGBase as RNG

__all__ = [
	"NumbaBackend",
]

@njit(cache=True, fastmath=False)
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
	for t in range(T):
		for n in range(N):
			acc_r = 0.0
			acc_i = 0.0
			for m in range(M):
				c = L[t, n, m]
				w = dW[t, m]
				acc_r += (c.real * w)
				acc_i += (c.imag * w)
			out[t, n] = acc_r + 1j * acc_i
	return out

@njit(cache=True, fastmath=False)
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
	for t in range(T):
		for k in range(K):
			acc = 0.0
			for m in range(M):
				acc += z[t, m] * chol_T[m, k]
			out[t, k] = acc
	return out

class NumbaRNG:
	"""Internal RNG handle backed by NumPy Generator.

	Lightweight adapter used by NumbaBackend to satisfy RNGBase semantics
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
			r = NumbaRNG(None)
			r._gen = gen
			result.append(r)
		return result


# ------------------------------ Backend API ------------------------------

class NumbaBackend(Backend):
	"""Numba implementation of the Backend protocol (CPU, optional accel).

	Mirrors the NumPy backend API while routing common hot-path contractions
	through Numba-compiled kernels when available, falling back to NumPy for
	general cases.

	Methods
	-------
	backend_name() -> str
		Return backend identifier ("numba").
	device() -> Optional[str]
		Return device (None for CPU).
	asarray/array/zeros/empty/empty_like/copy
		Array creation and conversion helpers.
	einsum(subscripts, *operands) -> Any
		Uses specialized kernels for 'tnm,tm->tn' and 'tm,mk->tk' patterns.
	cholesky(a) -> Any
		Cholesky factorization via NumPy.
	rng(seed), spawn_rngs(master_seed, n)
		Create RNG handle and independent streams.
	randn(rng, shape, dtype) -> Any
		Standard normal samples.
	real/imag/abs/concatenate/stack/to_device/capabilities
		Convenience and introspection helpers.

	Examples
	--------
	>>> be = NumbaBackend()
	>>> r = be.rng(0)
	>>> x = be.randn(r, (4, 2), dtype=None)
	>>> x.shape
	(4, 2)
	"""

    # Identification
	def backend_name(self) -> str:
		return "numba"

	def device(self) -> Optional[str]:
		return None

	# Array creation / conversion
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

	# Ops / linalg
	def einsum(self, subscripts: str, *operands: Any) -> Any:
		# Hot paths: match exact patterns to use Numba kernels
		if subscripts == 'tnm,tm->tn' and len(operands) == 2:
			L = operands[0]
			dW = operands[1]
			L_arr = np.asarray(L, dtype=np.complex128)
			dW_arr = np.asarray(dW, dtype=np.float64)
			return _einsum_tnm_tm_to_tn(L_arr, dW_arr)
		if subscripts == 'tm,mk->tk' and len(operands) == 2:
			z = operands[0]
			chol_T = operands[1]
			z_arr = np.asarray(z, dtype=np.float64)
			cholT_arr = np.asarray(chol_T, dtype=np.float64)
			return _einsum_tm_mk_to_tk(z_arr, cholT_arr)
		# Fallback to NumPy for general cases
		return np.einsum(subscripts, *operands, optimize=True)

	def cholesky(self, a: Any) -> Any:
		return np.linalg.cholesky(a)

	def rng(self, seed: int | None) -> RNG:
		return NumbaRNG(seed)

	def spawn_rngs(self, master_seed: int, n: int) -> List[RNG]:
		ss = np.random.SeedSequence(master_seed)
		children = ss.spawn(n)
		rngs: List[RNG] = []
		for child in children:
			gen = np.random.Generator(np.random.PCG64(child))
			nrng = NumbaRNG(None)
			nrng._gen = gen
			rngs.append(nrng)
		return rngs

	def normal(self, rng: RNG, shape: Tuple[int, ...], dtype: Any) -> Any:
		nrng = cast(NumbaRNG, rng)
		g = nrng._gen
		out = g.normal(size=shape)
		return out.astype(dtype if dtype is not None else np.float64, copy=False)

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
			"optimized_contractions": True,
			"supports_complex_view": False,
			"real_imag_split": True,
			"stack": True,
			"to_device": False,
			"numba": True,
			"numpy": True,
		}

