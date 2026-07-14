"""qphase_sde: fused drift/diffusion kernels.

Kernels are optional performance paths. Integrators prefer them when a model
advertises support and fall back to the Python drift/diffusion methods
otherwise.
"""

from __future__ import annotations

from typing import Any

from qphase.backend.base import BackendBase

__all__ = ["compile_cached_kernel", "supports_kernelized_terms"]


# Module-level cache: (kernel_name, dtype_name, code_hash) -> cupy.RawKernel
_CUPY_KERNEL_CACHE: dict[tuple[str, str, int], Any] = {}


def compile_cached_kernel(
    name: str, dtype: str, code: str, *, options: tuple[str, ...] = ()
) -> Any:
    """Compile or retrieve a CuPy RawKernel, caching by name and dtype.

    Parameters
    ----------
    name : str
        Logical kernel identifier.
    dtype : str
        CuTe / CUDA scalar type used in the kernel source, e.g.
        ``"complex<float>"`` or ``"complex<double>"``.
    code : str
        Complete CUDA source for the kernel. Must contain a global function
        named ``{name}_{dtype_slug}`` where dtype_slug is derived from *dtype*.
    options : tuple[str, ...]
        Extra NVTC options.

    Returns
    -------
    cupy.RawKernel
        Compiled kernel ready to launch.
    """
    import cupy as cp

    key = (name, dtype, hash(code))
    if key in _CUPY_KERNEL_CACHE:
        return _CUPY_KERNEL_CACHE[key]

    # Determine a safe slug for the C++ function name.
    slug = dtype.replace("<", "_").replace(">", "_").replace(",", "_").replace(" ", "_")
    func_name = f"{name}_{slug}"
    # Wrap user code with the requested function name.
    full_code = code.replace(f"__{name}_func__", func_name)

    kernel = cp.RawKernel(full_code, func_name, options=options)
    # Compile immediately so errors surface early.
    kernel.compile()
    _CUPY_KERNEL_CACHE[key] = kernel
    return kernel


def supports_kernelized_terms(model: Any, backend: BackendBase) -> bool:
    """Return True if *model* exposes a compatible kernel path for *backend*."""
    if not hasattr(model, "has_kernelized_terms"):
        return False
    try:
        return bool(model.has_kernelized_terms(backend))
    except Exception:
        return False
