"""CuPy RawKernel for the VDP Level 3 model.

Computes fused drift and diffusion for all trajectories in one launch.
The kernel uses only built-in CUDA scalar/complex types so that it compiles
under NVRTC without external headers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qphase.backend.base import BackendBase
from qphase_sde.kernels import compile_cached_kernel

from models.kernels.base import ModelKernelPlugin
from models.kernels.cupy_utils import broadcast_param

# CUDA source for the VDP Level 3 model.
# $T$ is replaced with ``float`` or ``double``; $CT$ with the matching complex
# vector type ``float2`` / ``double2``.
_VDP_SOURCE = r"""
extern "C" __global__
void __vdp_2mode_terms_func__(
    const $CT$* __restrict__ y,
    const double* __restrict__ omega_a,
    const double* __restrict__ omega_b,
    const double* __restrict__ gamma_a,
    const double* __restrict__ gamma_b,
    const double* __restrict__ Gamma,
    const double* __restrict__ g,
    int n,
    $CT$* __restrict__ drift,
    $CT$* __restrict__ diffusion
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    $CT$ alpha = y[i * 2 + 0];
    $CT$ beta  = y[i * 2 + 1];

    double oa = omega_a[i];
    double ob = omega_b[i];
    double ga = gamma_a[i];
    double gb = gamma_b[i];
    double G  = Gamma[i];
    double gc = g[i];

    double ar = alpha.x;
    double ai = alpha.y;
    double br = beta.x;
    double bi = beta.y;

    double n_alpha2 = ar * ar + ai * ai;

    double common_a = ga / 2.0 + G * (1.0 - n_alpha2);
    double dar = common_a * ar + oa * ai + gc * bi;
    double dai = common_a * ai - oa * ar - gc * br;

    double common_b = -gb / 2.0;
    double dbr = common_b * br + ob * bi + gc * ai;
    double dbi = common_b * bi - ob * br - gc * ar;

    drift[i * 2 + 0].x = dar;
    drift[i * 2 + 0].y = dai;
    drift[i * 2 + 1].x = dbr;
    drift[i * 2 + 1].y = dbi;

    double D_alpha = ga / 2.0 + G * (2.0 * n_alpha2 - 1.0);
    double D_beta  = gb / 2.0;
    if (D_alpha < 0.0) D_alpha = 0.0;
    if (D_beta  < 0.0) D_beta  = 0.0;

    $T$ sDa = sqrt(($T$)D_alpha);
    $T$ sDb = sqrt(($T$)D_beta);

    diffusion[i * 4 + 0].x = sDa; diffusion[i * 4 + 0].y = ($T$)0.0;
    diffusion[i * 4 + 1].x = ($T$)0.0; diffusion[i * 4 + 1].y = ($T$)0.0;
    diffusion[i * 4 + 2].x = ($T$)0.0; diffusion[i * 4 + 2].y = ($T$)0.0;
    diffusion[i * 4 + 3].x = sDb; diffusion[i * 4 + 3].y = ($T$)0.0;
}
"""


# Module-level buffer cache for drift/diffusion outputs.  This lives in the
# resource package (not the core qphase_sde package) so the core engine remains
# backend/model agnostic.  We keep a small LRU cache keyed by (n, dtype) to
# avoid per-step GPU memory allocation overhead during batch runs.
_BUFFER_CACHE: dict[tuple[int, Any], tuple[Any, Any]] = {}
_BUFFER_KEYS: list[tuple[int, Any]] = []
_MAX_BUFFERS = 2


def _get_buffers(n: int, dtype: Any) -> tuple[Any, Any]:
    """Return reusable (drift, diffusion) CuPy arrays for the given shape."""
    import cupy as cp

    key = (n, dtype)
    if key in _BUFFER_CACHE:
        # Move to most-recent position
        _BUFFER_KEYS.remove(key)
        _BUFFER_KEYS.append(key)
        return _BUFFER_CACHE[key]

    drift = cp.empty((n, 2), dtype=dtype)
    diffusion = cp.zeros((n, 2, 2), dtype=dtype)
    _BUFFER_CACHE[key] = (drift, diffusion)
    _BUFFER_KEYS.append(key)

    if len(_BUFFER_KEYS) > _MAX_BUFFERS:
        old_key = _BUFFER_KEYS.pop(0)
        del _BUFFER_CACHE[old_key]

    return drift, diffusion


def kernelized_terms(
    y: Any,
    params: dict[str, Any],
    backend: BackendBase,
) -> tuple[Any, Any]:
    """Return (drift, diffusion) for VDP Level 3 using a CuPy kernel.

    Parameters
    ----------
    y : cupy.ndarray
        State array of shape ``(n_traj, 2)`` (complex).
    params : dict[str, Any]
        Model parameters. Each value may be a scalar or a ``(n_traj,)`` array.
    backend : BackendBase
        Active CuPy backend.

    Returns
    -------
    drift : cupy.ndarray
        Shape ``(n_traj, 2)`` complex.
    diffusion : cupy.ndarray
        Shape ``(n_traj, 2, 2)`` complex (complex noise basis).

    """
    n = int(y.shape[0])
    rdtype = y.real.dtype
    if rdtype == np.float32:
        source = _VDP_SOURCE.replace("$T$", "float").replace("$CT$", "float2")
        ctype = "complex<float>"
    else:
        source = _VDP_SOURCE.replace("$T$", "double").replace("$CT$", "double2")
        ctype = "complex<double>"

    kernel = compile_cached_kernel("vdp_2mode_terms", ctype, source)

    # Broadcast all parameters to per-trajectory arrays.
    omega_a = broadcast_param(params["omega_a"], n)
    omega_b = broadcast_param(params["omega_b"], n)
    gamma_a = broadcast_param(params["gamma_a"], n)
    gamma_b = broadcast_param(params["gamma_b"], n)
    Gamma = broadcast_param(params["Gamma"], n)
    g = broadcast_param(params["g"], n)

    drift, diffusion = _get_buffers(n, y.dtype)

    threads = 256
    blocks = (n + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            y,
            omega_a,
            omega_b,
            gamma_a,
            gamma_b,
            Gamma,
            g,
            n,
            drift,
            diffusion,
        ),
    )
    return drift, diffusion


class VDP2ModeEulerCuPyKernel(ModelKernelPlugin):
    """CuPy drift/diffusion provider used by Euler-Maruyama."""

    scheme = "euler_maruyama"
    backend_name = "cupy"
    operations = frozenset({"terms"})

    def terms(
        self, y: Any, params: dict[str, Any], backend: BackendBase
    ) -> tuple[Any, Any]:
        return kernelized_terms(y, params, backend)
