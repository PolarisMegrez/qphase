"""qphase_sde: Core Protocols
----------------------------

Independent protocol definitions for qphase_sde.
These protocols define core contracts without depending on qphase,
enabling SDE to operate as a standalone scientific computing library.

Architecture
------------
This module defines minimal protocols that backend implementations must satisfy.
Protocols are dependency-light and use duck typing, allowing any object
implementing the required methods to serve as a backend.

Protocol Hierarchy
------------------
- SDEBackend: Main computational backend protocol
- RNGHandle: Random number generator handle protocol

These protocols are designed to be compatible with existing backend
implementations (numpy, cupy, torch, numba) while maintaining independence
from the qphase package.
"""

from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "SDEBackend",
]


@runtime_checkable
class SDEBackend(Protocol):
    """Minimal backend protocol for SDE computations.

    This protocol defines the essential operations required by qphase_sde
    for stochastic differential equation simulations. It is designed to be
    compatible with existing backend implementations while maintaining complete
    independence from qphase.

    Core Capabilities
    -----------------
    - Array creation and manipulation
    - Linear algebra operations (eigenvalues, Cholesky)
    - Random number generation
    - Device management (CPU/GPU)
    - Basic mathematical operations

    Implementation Notes
    --------------------
    Backends should implement the required methods below. Optional methods
    may raise AttributeError if not supported; callers should handle this
    gracefully using hasattr() checks.

    Examples
    --------
    >>> # Duck typing - any object with the required methods works
    >>> def simulate(model, backend):
    ...     if not isinstance(backend, SDEBackend):
    ...         raise TypeError("Backend must implement SDEBackend protocol")
    ...     rng = backend.rng(seed=42)
    ...     arrays = backend.zeros((100, 2), dtype=float)
    ...     return arrays

    """

    # -------------------------------------------------------------------------
    # Identification and Metadata
    # -------------------------------------------------------------------------

    def backend_name(self) -> str:
        """Return backend identifier string.

        Returns
        -------
        str
            Backend name (e.g., 'numpy', 'cupy', 'torch', 'numba').

        """
        ...

    def device(self) -> str | None:
        """Return device string when applicable.

        Returns
        -------
        str | None
            Device string (e.g., 'cuda:0', 'mps:0') for GPU backends,
            or None for CPU-only backends.

        """
        ...

    def capabilities(self) -> dict[str, Any]:
        """Report backend capabilities for feature detection.

        Returns
        -------
        dict[str, Any]
            Dictionary of capability flags and values. Recommended keys:
            - 'gpu': bool - GPU acceleration available
            - 'double_precision': bool - float64 support
            - 'complex': bool - complex number support
            - 'sparse': bool - sparse matrix support
            - 'autograd': bool - automatic differentiation

        """
        ...

    # -------------------------------------------------------------------------
    # Array Creation and Conversion
    # -------------------------------------------------------------------------

    def array(self, obj: Any, dtype: Any | None = None) -> Any:
        """Create an array from the given object.

        Parameters
        ----------
        obj : Any
            Input data, can be list, tuple, scalar, or array-like.
        dtype : Any | None, optional
            Desired data type. If None, infers from input.

        Returns
        -------
        Any
            Backend-specific array object.

        """
        ...

    def asarray(self, obj: Any, dtype: Any | None = None) -> Any:
        """Convert input to array (zero-copy when possible).

        Parameters
        ----------
        obj : Any
            Input data to convert.
        dtype : Any | None, optional
            Desired data type.

        Returns
        -------
        Any
            Backend-specific array object.

        """
        ...

    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any:
        """Create an array filled with zeros.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the array.
        dtype : Any
            Data type (e.g., float32, float64).

        Returns
        -------
        Any
            Array filled with zeros.

        """
        ...

    def empty(self, shape: tuple[int, ...], dtype: Any) -> Any:
        """Create an uninitialized array (faster than zeros).

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the array.
        dtype : Any
            Data type.

        Returns
        -------
        Any
            Uninitialized array.

        """
        ...

    def empty_like(self, x: Any) -> Any:
        """Create an uninitialized array with same shape and dtype.

        Parameters
        ----------
        x : Any
            Reference array.

        Returns
        -------
        Any
            Uninitialized array with same shape and dtype as x.

        """
        ...

    def copy(self, x: Any) -> Any:
        """Create a copy of the input array.

        Parameters
        ----------
        x : Any
            Array to copy.

        Returns
        -------
        Any
            Deep copy of the input array.

        """
        ...

    # -------------------------------------------------------------------------
    # Linear Algebra
    # -------------------------------------------------------------------------

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        """Einstein summation for tensor operations.

        Parameters
        ----------
        subscripts : str
            Subscript string (e.g., 'ij,jk->ik').
        *operands : Any
            Arrays to operate on.

        Returns
        -------
        Any
            Result of the Einstein summation.

        """
        ...

    def concatenate(self, arrays: tuple[Any, ...], axis: int = -1) -> Any:
        """Concatenate arrays along specified axis.

        Parameters
        ----------
        arrays : tuple[Any, ...]
            Tuple of arrays to concatenate.
        axis : int, default -1
            Axis along which to concatenate.

        Returns
        -------
        Any
            Concatenated array.

        """
        ...

    def cholesky(self, a: Any) -> Any:
        """Cholesky decomposition for SPD matrices.

        Parameters
        ----------
        a : Any
            Symmetric positive-definite matrix.

        Returns
        -------
        Any
            Lower triangular Cholesky factor.

        """
        ...

    def eigvals(self, a: Any) -> Any:
        """Compute eigenvalues of a matrix.

        Parameters
        ----------
        a : Any
            Square matrix.

        Returns
        -------
        Any
            Eigenvalues.

        """
        ...

    # -------------------------------------------------------------------------
    # Complex Number Support
    # -------------------------------------------------------------------------

    def real(self, x: Any) -> Any:
        """Return the real part of the array.

        Parameters
        ----------
        x : Any
            Input array.

        Returns
        -------
        Any
            Real part of the array.

        """
        ...

    def imag(self, x: Any) -> Any:
        """Return the imaginary part of the array.

        Parameters
        ----------
        x : Any
            Input array.

        Returns
        -------
        Any
            Imaginary part of the array.

        """
        ...

    def abs(self, x: Any) -> Any:
        """Return the absolute value of the array.

        Parameters
        ----------
        x : Any
            Input array.

        Returns
        -------
        Any
            Absolute value of the array.

        """
        ...

    def mean(self, x: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        """Compute the arithmetic mean along the specified axis.

        Parameters
        ----------
        x : Any
            Input array.
        axis : int | tuple[int, ...] | None, default None
            Axis or axes along which to operate.

        Returns
        -------
        Any
            Mean of the array elements.

        """
        ...

    # -------------------------------------------------------------------------
    # FFT
    # -------------------------------------------------------------------------

    def fft(
        self,
        x: Any,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> Any:
        """Compute the 1D FFT along the specified axis.

        Parameters
        ----------
        x : Any
            Input array.
        axis : int, default -1
            Axis along which to compute the FFT.
        norm : str | None, default None
            Normalization mode (e.g., "ortho").

        Returns
        -------
        Any
            FFT of the input array.

        """
        ...

    def fftfreq(self, n: int, d: float = 1.0) -> Any:
        """Return the Discrete Fourier Transform sample frequencies.

        Parameters
        ----------
        n : int
            Window length.
        d : float, default 1.0
            Sample spacing.

        Returns
        -------
        Any
            Array containing the sample frequencies.

        """
        ...

    # -------------------------------------------------------------------------
    # Random Number Generation
    # -------------------------------------------------------------------------

    def rng(self, seed: int | None) -> Any:
        """Create an RNG handle for the backend.

        Parameters
        ----------
        seed : int | None
            Seed for the RNG. If None, random seed is derived.

        Returns
        -------
        Any
            Random number generator handle.

        """
        ...

    def randn(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any:
        """Generate standard normal random numbers.

        Parameters
        ----------
        rng : Any
            Random number generator handle.
        shape : tuple[int, ...]
            Shape of the output array.
        dtype : Any
            Data type (e.g., float32, float64).

        Returns
        -------
        Any
            Array of standard normal random numbers.

        """
        ...

    def spawn_rngs(self, master_seed: int, n: int) -> list[Any]:
        """Spawn independent RNG streams deterministically.

        Parameters
        ----------
        master_seed : int
            Master seed for deterministic stream generation.
        n : int
            Number of RNG streams to spawn.

        Returns
        -------
        list[Any]
            List of independent RNG handles.

        """
        ...

    # -------------------------------------------------------------------------
    # Optional Convenience Methods
    # -------------------------------------------------------------------------
    # These methods are optional but commonly implemented.
    # Backends should raise AttributeError if not supported.

    def stack(self, arrays: tuple[Any, ...], axis: int = 0) -> Any:
        """Stack arrays along a new axis (optional).

        Parameters
        ----------
        arrays : tuple[Any, ...]
            Arrays to stack.
        axis : int, default 0
            Axis along which to stack.

        Returns
        -------
        Any
            Stacked array.

        Raises
        ------
        AttributeError
            If not implemented by backend.

        """
        ...

    def to_device(self, x: Any, device: str | None) -> Any:
        """Transfer array to specified device (optional).

        Parameters
        ----------
        x : Any
            Array to transfer.
        device : str | None
            Target device (e.g., 'cuda:0', 'cpu').

        Returns
        -------
        Any
            Array on target device.

        Raises
        ------
        AttributeError
            If not implemented by backend.

        """
        ...
