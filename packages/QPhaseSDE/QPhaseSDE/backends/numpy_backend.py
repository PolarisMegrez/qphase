from __future__ import annotations

"""NumPy backend implementation of the Backend protocol."""

from typing import Any, Tuple, cast, List, Optional
import numpy as np

from ..core.protocols import BackendBase as Backend, RNGBase as RNG


class NumpyRNG:
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
    name = "numpy"

    # Identification
    def backend_name(self) -> str:
        return self.name

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
        return np.einsum(subscripts, *operands)

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

    # Optional introspection of capabilities
    def capabilities(self) -> dict:
        return {
            "stack": True,
            "to_device": False,
            "complex_view": False,
            "real_imag_split": True,
        }
