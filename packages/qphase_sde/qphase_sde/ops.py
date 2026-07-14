"""qphase_sde: Backend-agnostic SDE operations.
---------------------------------------------------------
Shared, backend-aware primitives used by all integrators. Keeping these
operations in one place ensures that Euler-Maruyama, Milstein, and SRK all
benefit from the same backend-specific optimizations (e.g. cupy matmul
instead of generic einsum) without duplicating code.

Public API
----------
``expand_complex_noise`` : Expand a complex-basis diffusion matrix to a
    real-basis equivalent.
``contract_noise`` : Compute L @ dW across the noise channel dimension.
``scaled_noise`` : Generate sqrt(dt) * N(0,1) noise increments.
"""

from typing import Any

from qphase.backend.base import BackendBase

__all__ = [
    "expand_complex_noise",
    "contract_noise",
    "scaled_noise",
    "supports_kernelized_terms",
]


def expand_complex_noise(Lc: Any, backend: BackendBase) -> Any:
    """Expand complex-basis diffusion matrix to an equivalent real basis.

    Transforms ``Lc`` with shape ``(..., n_modes, M_c)`` into ``Lr`` with
    shape ``(..., n_modes, 2*M_c)`` using only backend operations, preserving
    contraction with real-valued noise increments.
    """
    a = backend.real(Lc)
    b = backend.imag(Lc)
    s = backend.sqrt(backend.asarray(2.0, dtype=a.dtype if hasattr(a, "dtype") else None))
    Lr_real = backend.concatenate((a / s, -b / s), axis=-1)
    Lr_imag = backend.concatenate((b / s, a / s), axis=-1)
    return Lr_real + 1j * Lr_imag


def contract_noise(L: Any, dW: Any, backend: BackendBase) -> Any:
    """Contract diffusion matrix ``L`` with noise increments ``dW``.

    Expected shapes:
    - ``L``: ``(..., n_modes, M)``
    - ``dW``: ``(..., M)``
    - output: ``(..., n_modes)``

    The implementation picks the fastest backend-specific contraction:
    - cupy/torch: batched matrix multiplication
    - numpy/numba: ``einsum("...ij,...j->...i", L, dW)``
    """
    be_name = ""
    try:
        be_name = str(backend.backend_name()).lower()
    except Exception:
        pass

    if be_name == "cupy":
        # Avoid importing cupy at module load time; it may not be installed.
        import cupy as cp

        return cp.matmul(L, dW[..., None]).squeeze(-1)

    if be_name == "torch":
        import torch as th

        return th.bmm(L, dW.unsqueeze(-1)).squeeze(-1)

    # Generic fallback: numpy, numba, and any other backend implementing einsum.
    return backend.einsum("...ij,...j->...i", L, dW)


def scaled_noise(
    rng: Any,
    shape: tuple[int, ...],
    dt: float,
    dtype: Any,
    backend: BackendBase,
) -> Any:
    """Generate Gaussian noise increments scaled by ``sqrt(dt)``.

    The scalar ``sqrt(dt)`` is converted to a backend array with the same
    dtype as the requested noise to avoid unnecessary type promotion and
    keep the computation on the target device.
    """
    raw = backend.randn(rng, shape, dtype=dtype)
    dt_sqrt = backend.asarray(float(dt) ** 0.5, dtype=dtype)
    return raw * dt_sqrt


def supports_kernelized_terms(model: Any, backend: BackendBase) -> bool:
    """Return True if *model* exposes a compatible fused kernel for *backend*."""
    if not hasattr(model, "has_kernelized_terms"):
        return False
    try:
        return bool(model.has_kernelized_terms(backend))
    except Exception:
        return False
