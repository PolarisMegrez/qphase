"""CuPy Euler-Maruyama terms kernel for the three-mode Kerr model."""

from __future__ import annotations

from typing import Any

import numpy as np
from qphase.backend.base import BackendBase
from qphase_sde.kernels import compile_cached_kernel

from models.kernels.base import ModelKernelPlugin
from models.kernels.cupy_utils import broadcast_param

_SOURCE = r"""
extern "C" __global__
void __kerr_3mode_terms_func__(
    const $CT$* y, const $T$* omega_a, const $T$* omega_b,
    const $T$* omega_c, const $T$* chi, const $T$* gamma_a,
    const $T$* gamma_b, const $T$* gamma_c, const $T$* g_ab,
    const $T$* g_ac, int n, $CT$* drift, $CT$* diffusion
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    $CT$ a = y[i * 3], b = y[i * 3 + 1], c = y[i * 3 + 2];
    $T$ ga = gamma_a[i], gb = gamma_b[i], gc = gamma_c[i];
    $T$ gab = g_ab[i], gac = g_ac[i];
    $T$ frequency_a = omega_a[i] + ($T$)2.0 * chi[i] *
        (a.x * a.x + a.y * a.y - ($T$)1.0);

    drift[i * 3].x = -($T$)0.5 * ga * a.x + frequency_a * a.y
        + gab * b.y + gac * c.y;
    drift[i * 3].y = -($T$)0.5 * ga * a.y - frequency_a * a.x
        - gab * b.x - gac * c.x;
    drift[i * 3 + 1].x = -($T$)0.5 * gb * b.x + omega_b[i] * b.y
        + gab * a.y;
    drift[i * 3 + 1].y = -($T$)0.5 * gb * b.y - omega_b[i] * b.x
        - gab * a.x;
    drift[i * 3 + 2].x = ($T$)0.5 * gc * c.x + omega_c[i] * c.y
        + gac * a.y;
    drift[i * 3 + 2].y = ($T$)0.5 * gc * c.y - omega_c[i] * c.x
        - gac * a.x;

    diffusion[i * 9].x = sqrt(($T$)(0.5 * ga));
    diffusion[i * 9].y = ($T$)0.0;
    diffusion[i * 9 + 4].x = sqrt(($T$)(0.5 * gb));
    diffusion[i * 9 + 4].y = ($T$)0.0;
    diffusion[i * 9 + 8].x = sqrt(($T$)(0.5 * gc));
    diffusion[i * 9 + 8].y = ($T$)0.0;
}
"""

_BUFFERS: dict[tuple[int, Any], tuple[Any, Any]] = {}
_LAUNCH_REFERENCES: list[tuple[Any, ...]] = []


def _retain_launch(*values: Any) -> None:
    _LAUNCH_REFERENCES.append(values)
    if len(_LAUNCH_REFERENCES) > 4:
        _LAUNCH_REFERENCES.pop(0)


def kernelized_terms(
    y: Any, params: dict[str, Any], backend: BackendBase
) -> tuple[Any, Any]:
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
    kernel = compile_cached_kernel("kerr_3mode_terms", ctype, source)
    key = (n, y.dtype)
    if key not in _BUFFERS:
        _BUFFERS[key] = (
            cp.empty((n, 3), dtype=y.dtype),
            cp.zeros((n, 3, 3), dtype=y.dtype),
        )
    drift, diffusion = _BUFFERS[key]
    values = [
        broadcast_param(params[name], n, rdtype)
        for name in (
            "omega_a",
            "omega_b",
            "omega_c",
            "chi",
            "gamma_a",
            "gamma_b",
            "gamma_c",
            "g_ab",
            "g_ac",
        )
    ]
    threads = 256
    kernel(
        ((n + threads - 1) // threads,),
        (threads,),
        (y, *values, n, drift, diffusion),
    )
    _retain_launch(*values)
    return drift, diffusion


class Kerr3ModeEulerCuPyKernel(ModelKernelPlugin):
    """CuPy terms provider for three-mode Kerr Euler-Maruyama steps."""

    scheme = "euler_maruyama"
    backend_name = "cupy"
    operations = frozenset({"terms"})

    def terms(
        self, y: Any, params: dict[str, Any], backend: BackendBase
    ) -> tuple[Any, Any]:
        return kernelized_terms(y, params, backend)
