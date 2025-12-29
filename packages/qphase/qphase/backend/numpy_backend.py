"""qphase: NumPy Backend Reference Implementation
---------------------------------------------------------
Implements the reference CPU backend using standard NumPy arrays. It serves as the
compatibility baseline and default implementation for the framework, providing
reliable, unaccelerated execution of all standard backend operations including
array creation, linear algebra, random number generation, and FFT.

Public API
----------
``NumpyConfig`` : Configuration with einsum optimization toggle
``NumpyBackend`` : Full backend implementation using NumPy arrays
"""

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field

from qphase.backend.base import BackendBase as Backend
from qphase.backend.base import BackendConfigBase

__all__ = [
    "NumpyBackend",
]


class NumpyConfig(BackendConfigBase):
    """Configuration for NumPy backend."""

    optimize_einsum: bool = Field(
        default=True, description="Enable einsum optimization"
    )


class NumpyBackend(Backend):
    """NumPy implementation of the Backend protocol (CPU only).

    Provides a reference CPU backend using NumPy arrays and routines. It mirrors
    the common backend API (array creation, basic linalg, RNG, and utilities)
    without JIT acceleration; all operations are executed by NumPy.
    """

    name: ClassVar[str] = "numpy"
    description: ClassVar[str] = (
        "NumPy implementation of the Backend protocol (CPU only)."
    )
    config_schema: ClassVar[type[NumpyConfig]] = NumpyConfig

    def __init__(self, config: NumpyConfig | None = None, **kwargs: Any) -> None:
        """Initialize NumPy backend with configuration."""
        self.config = config or NumpyConfig()
        _ = kwargs

    # Identification
    def backend_name(self) -> str:
        return "numpy"

    def device(self) -> str | None:
        return None

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

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        # Use configuration for einsum optimization
        return np.einsum(subscripts, *operands, optimize=self.config.optimize_einsum)

    def cholesky(self, a: Any) -> Any:
        return np.linalg.cholesky(a)

    # -------------------------------------------------------------------------
    # Random Number Generation
    # -------------------------------------------------------------------------

    def rng(self, seed: int | None) -> Any:
        return np.random.default_rng(seed)

    def spawn_rngs(self, master_seed: int, n: int) -> list[Any]:
        ss = np.random.SeedSequence(master_seed)
        children = ss.spawn(n)
        # Create independent Generators from spawned SeedSequences
        return [np.random.default_rng(child) for child in children]

    def normal(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any:
        # rng is expected to be a numpy.random.Generator
        out = rng.normal(size=shape)
        return out.astype(dtype if dtype is not None else np.float64, copy=False)

    def randn(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any:
        return self.normal(rng, shape, dtype)

    # -------------------------------------------------------------------------
    # Math & FFT
    # -------------------------------------------------------------------------

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

    def expand_dims(self, x: Any, axis: int) -> Any:
        return np.expand_dims(x, axis=axis)

    def repeat(self, x: Any, repeats: int, axis: int | None = None) -> Any:
        return np.repeat(x, repeats, axis=axis)

    def isnan(self, x: Any) -> Any:
        return np.isnan(x)

    def arange(
        self,
        start: int,
        stop: int | None = None,
        step: int = 1,
        dtype: Any | None = None,
    ) -> Any:
        return np.arange(start, stop, step, dtype=dtype)

    @property
    def pi(self) -> float:
        return np.pi

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
