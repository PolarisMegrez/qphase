"""qphase: Backend Protocol Definitions
---------------------------------------------------------
Defines the core interfaces and configuration models for backend implementations.
It establishes the ``BackendBase`` protocol, a comprehensive contract that all
backends must satisfy, covering array operations, linear algebra, random number
generation, FFT, and device management, ensuring uniform behavior across different
computational engines.

Public API
----------
``BackendConfigBase`` : Base configuration class for all backend plugins
``BackendBase`` : Protocol defining the complete backend interface

Notes
-----
This module is dependency-free and must not import numpy, torch, or similar
libraries. Concrete implementations in backend/*.py must implement all methods.

"""

from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from pydantic import Field

from qphase.core.protocols import PluginBase, PluginConfigBase

__all__ = [
    "BackendConfigBase",
    "BackendBase",
]


class BackendConfigBase(PluginConfigBase):
    """Base configuration class for all Backend plugins.

    Provides common fields for all backend configurations:
    - device: Optional device identifier (e.g., "cpu", "cuda:0")
    - float_dtype: Default floating-point dtype
    """

    device: str | None = Field(
        default=None, description="Device identifier (e.g., 'cpu', 'cuda:0')"
    )
    float_dtype: str = Field(
        default="float64", description="Default floating-point dtype"
    )


@runtime_checkable
class BackendBase(PluginBase, Protocol):
    """Backend protocol that extends PluginBase.

    Concrete backends live under this subpackage and must implement the methods
    below. Core layers rely only on this interface; additional methods may be
    provided but must be guarded by capability checks.

    Backend classes must also define:
    - name: ClassVar[str] - Unique identifier for the backend
    - description: ClassVar[str] - Human-readable description
    - config_schema: ClassVar[type[BackendConfigBase]] - Configuration schema

    Methods
    -------
    backend_name() -> str
        Return backend identifier.
    device() -> Optional[str]
        Return device string when applicable (e.g., "cuda:0"), else None.
    capabilities() -> Dict[str, Any]
        Report unified, backend-agnostic features for discovery.
    array/asarray/zeros/empty/empty_like/copy
        Array creation and conversion helpers.
    einsum(subscripts, *operands) -> Any
        General tensor contraction.
    concatenate(arrays, axis=-1) -> Any
        Concatenate along a given axis.
    cholesky(a) -> Any
        Cholesky factorization.
    real(x)/imag(x) -> Any
        Complex helpers to access views of real/imag parts.
    rng(seed) -> Any
        Create an RNG handle.
    randn(rng, shape, dtype) -> Any
        Standard normal sampling.
    spawn_rngs(master_seed, n) -> list[Any]
        Spawn independent RNG streams deterministically.
    fft(x, axis=-1, norm=None) -> Any
        Compute the 1D FFT along the specified axis.
    fftfreq(n, d=1.0) -> Any
        Return the Discrete Fourier Transform sample frequencies.
    mean(x, axis=None) -> Any
        Compute the arithmetic mean along the specified axis.
    abs(x) -> Any
        Calculate the absolute value element-wise.
    stack(arrays, axis=0) -> Any
        Optional helper to stack arrays along a new axis.
    to_device(x, device) -> Any
        Optional helper to move/adapt arrays to a device.

    """

    # Plugin metadata (inherited from PluginBase via implementation)
    # These must be defined as class variables in concrete implementations
    name: ClassVar[str]
    description: ClassVar[str]
    config_schema: ClassVar[type[BackendConfigBase]]

    # Backend operations
    def backend_name(self) -> str: ...
    def device(self) -> str | None: ...
    def capabilities(self) -> dict[str, Any]: ...  # minimal, unified capability keys

    # Array creation / conversion
    def array(self, obj: Any, dtype: Any | None = None) -> Any: ...
    def asarray(self, obj: Any, dtype: Any | None = None) -> Any: ...
    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any: ...
    def empty(self, shape: tuple[int, ...], dtype: Any) -> Any: ...
    def empty_like(self, x: Any) -> Any: ...
    def copy(self, x: Any) -> Any: ...

    # Basic ops / linalg
    def einsum(self, subscripts: str, *operands: Any) -> Any: ...
    def concatenate(self, arrays: tuple[Any, ...], axis: int = -1) -> Any: ...
    def cholesky(self, a: Any) -> Any: ...

    # Complex helpers
    def real(self, x: Any) -> Any: ...
    def imag(self, x: Any) -> Any: ...
    def abs(self, x: Any) -> Any: ...
    def mean(self, x: Any, axis: int | tuple[int, ...] | None = None) -> Any: ...

    # FFT
    def fft(
        self,
        x: Any,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> Any: ...
    def fftfreq(self, n: int, d: float = 1.0) -> Any: ...

    # RNG
    def rng(self, seed: int | None) -> Any: ...
    def randn(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any: ...
    def spawn_rngs(self, master_seed: int, n: int) -> list[Any]: ...

    # Optional convenience (not required by core, but used opportunistically)
    # Implementers may raise AttributeError if not supported; callers must guard.
    def stack(self, arrays: tuple[Any, ...], axis: int = 0) -> Any: ...  # optional
    def to_device(self, x: Any, device: str | None) -> Any: ...  # optional
