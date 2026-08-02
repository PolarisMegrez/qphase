"""CuPy Euler-Maruyama terms kernel for the two-mode cross-Kerr model."""

from __future__ import annotations

from typing import Any

import numpy as np
from qphase.backend.base import BackendBase

from models.kernels.base import ModelKernelPlugin
from models.kernels.cupy_utils import broadcast_param, compile_cached_kernel

_SOURCE = r"""
extern "C" __global__
void __crosskerr_2mode_terms_func__(
    const $CT$* y, const $T$* omega_a, const $T$* omega_b,
    const $T$* chi, const $T$* gamma_a, const $T$* gamma_b,
    const $T$* g, int n, $CT$* drift, $CT$* diffusion
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    $CT$ a = y[i * 2], b = y[i * 2 + 1];
    $T$ ga = gamma_a[i], gb = gamma_b[i], coupling = g[i];
    $T$ frequency_a = omega_a[i] + chi[i] *
        (b.x * b.x + b.y * b.y - ($T$)0.5);
    $T$ frequency_b = omega_b[i] + chi[i] *
        (a.x * a.x + a.y * a.y - ($T$)0.5);

    drift[i * 2].x = ($T$)0.5 * ga * a.x + frequency_a * a.y
        + coupling * b.y;
    drift[i * 2].y = ($T$)0.5 * ga * a.y - frequency_a * a.x
        - coupling * b.x;
    drift[i * 2 + 1].x = -($T$)0.5 * gb * b.x + frequency_b * b.y
        + coupling * a.y;
    drift[i * 2 + 1].y = -($T$)0.5 * gb * b.y - frequency_b * b.x
        - coupling * a.x;

    $T$ sigma_a = sqrt(($T$)0.5 * ga);
    $T$ sigma_b = sqrt(($T$)0.5 * gb);
    diffusion[i * 4].x = sigma_a;
    diffusion[i * 4].y = ($T$)0.0;
    diffusion[i * 4 + 3].x = sigma_b;
    diffusion[i * 4 + 3].y = ($T$)0.0;
}
"""

_BUFFERS: dict[tuple[int, Any], tuple[Any, Any]] = {}
_LAUNCH_REFERENCES: list[tuple[Any, ...]] = []
_PARAMETER_NAMES = ("omega_a", "omega_b", "chi", "gamma_a", "gamma_b", "g")


def _retain_launch(*values: Any) -> None:
    _LAUNCH_REFERENCES.append(values)
    if len(_LAUNCH_REFERENCES) > 4:
        _LAUNCH_REFERENCES.pop(0)


def kernelized_terms(
    y: Any, params: dict[str, Any], backend: BackendBase
) -> tuple[Any, Any]:
    """Return fused drift and diffusion terms on CuPy."""
    del backend
    import cupy as cp

    n = int(y.shape[0])
    rdtype = y.real.dtype
    if rdtype == np.float32:
        source = _SOURCE.replace("$T$", "float").replace("$CT$", "float2")
        ctype = "complex<float>"
    else:
        source = _SOURCE.replace("$T$", "double").replace("$CT$", "double2")
        ctype = "complex<double>"
    kernel = compile_cached_kernel("crosskerr_2mode_terms", ctype, source)
    key = (n, y.dtype)
    if key not in _BUFFERS:
        _BUFFERS[key] = (
            cp.empty((n, 2), dtype=y.dtype),
            cp.zeros((n, 2, 2), dtype=y.dtype),
        )
    drift, diffusion = _BUFFERS[key]
    values = [broadcast_param(params[name], n, rdtype) for name in _PARAMETER_NAMES]
    threads = 256
    kernel(
        ((n + threads - 1) // threads,),
        (threads,),
        (y, *values, n, drift, diffusion),
    )
    _retain_launch(*values)
    return drift, diffusion


class CrossKerr2ModeEulerCuPyKernel(ModelKernelPlugin):
    """CuPy terms provider for cross-Kerr Euler-Maruyama steps."""

    scheme = "euler_maruyama"
    backend_name = "cupy"
    operations = frozenset({"terms"})

    def terms(
        self, y: Any, params: dict[str, Any], backend: BackendBase
    ) -> tuple[Any, Any]:
        return kernelized_terms(y, params, backend)
