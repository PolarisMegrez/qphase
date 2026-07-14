"""CuPy Cayley-Maruyama step kernel for the two-mode VDP model."""

from __future__ import annotations

from typing import Any

import numpy as np
from qphase.backend.base import BackendBase
from qphase_sde.kernels import compile_cached_kernel

from models.kernels.cupy_utils import broadcast_param

_CAYLEY_SOURCE = r"""
__device__ inline $CT$ c_make($T$ x, $T$ y) {
    $CT$ z; z.x = x; z.y = y; return z;
}

__device__ inline $CT$ c_add($CT$ a, $CT$ b) {
    return c_make(a.x + b.x, a.y + b.y);
}

__device__ inline $CT$ c_sub($CT$ a, $CT$ b) {
    return c_make(a.x - b.x, a.y - b.y);
}

__device__ inline $CT$ c_mul($CT$ a, $CT$ b) {
    return c_make(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ inline $CT$ c_div($CT$ a, $CT$ b) {
    $T$ den = b.x * b.x + b.y * b.y;
    return c_make(
        (a.x * b.x + a.y * b.y) / den,
        (a.y * b.x - a.x * b.y) / den
    );
}

extern "C" __global__
void __vdp_2mode_cayley_step_func__(
    const $CT$* __restrict__ y,
    const $T$* __restrict__ dW,
    const double* __restrict__ omega_a,
    const double* __restrict__ omega_b,
    const double* __restrict__ gamma_a,
    const double* __restrict__ gamma_b,
    const double* __restrict__ Gamma,
    const double* __restrict__ g,
    const double* __restrict__ D,
    $T$ dt,
    int n,
    $CT$* __restrict__ dy
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    $CT$ alpha = y[i * 2 + 0];
    $CT$ beta  = y[i * 2 + 1];

    $T$ wa = ($T$)omega_a[i];
    $T$ wb = ($T$)omega_b[i];
    $T$ ga = ($T$)gamma_a[i];
    $T$ gb = ($T$)gamma_b[i];
    $T$ G  = ($T$)Gamma[i];
    $T$ gc = ($T$)g[i];
    $T$ d  = ($T$)D[i];
    $T$ half_dt = ($T$)0.5 * dt;

    $T$ n_alpha2 = alpha.x * alpha.x + alpha.y * alpha.y;
    $T$ gain = ga / ($T$)2.0 + G * (($T$)1.0 - n_alpha2);

    $CT$ drift_a = c_make(
        gain * alpha.x + wa * alpha.y + gc * beta.y,
        gain * alpha.y - wa * alpha.x - gc * beta.x
    );
    $CT$ drift_b = c_make(
        -gb / ($T$)2.0 * beta.x + wb * beta.y + gc * alpha.y,
        -gb / ($T$)2.0 * beta.y - wb * beta.x - gc * alpha.x
    );

    $T$ D_alpha = d * (ga / ($T$)2.0 + G * (($T$)2.0 * n_alpha2 - ($T$)1.0));
    $T$ D_beta = d * gb / ($T$)2.0;
    if (D_alpha < ($T$)0.0) D_alpha = ($T$)0.0;
    if (D_beta < ($T$)0.0) D_beta = ($T$)0.0;

    $T$ sDa = sqrt(D_alpha / ($T$)2.0);
    $T$ sDb = sqrt(D_beta / ($T$)2.0);
    $CT$ eta_a = c_make(sDa * dW[i * 4 + 0], sDa * dW[i * 4 + 2]);
    $CT$ eta_b = c_make(sDb * dW[i * 4 + 1], sDb * dW[i * 4 + 3]);

    $CT$ rhs_a = c_add(
        c_add(alpha, c_make(half_dt * drift_a.x, half_dt * drift_a.y)),
        eta_a
    );
    $CT$ rhs_b = c_add(
        c_add(beta, c_make(half_dt * drift_b.x, half_dt * drift_b.y)),
        eta_b
    );

    $CT$ m00 = c_make(($T$)1.0 - half_dt * gain, half_dt * wa);
    $CT$ m01 = c_make(($T$)0.0, half_dt * gc);
    $CT$ m10 = m01;
    $CT$ m11 = c_make(($T$)1.0 + half_dt * gb / ($T$)2.0, half_dt * wb);

    $CT$ det = c_sub(c_mul(m00, m11), c_mul(m01, m10));
    $CT$ next_a = c_div(c_sub(c_mul(rhs_a, m11), c_mul(m01, rhs_b)), det);
    $CT$ next_b = c_div(c_sub(c_mul(m00, rhs_b), c_mul(rhs_a, m10)), det);

    dy[i * 2 + 0] = c_sub(next_a, alpha);
    dy[i * 2 + 1] = c_sub(next_b, beta);
}
"""

_BUFFER_CACHE: dict[tuple[int, Any], Any] = {}
_BUFFER_KEYS: list[tuple[int, Any]] = []
_MAX_BUFFERS = 2


def _get_buffer(n: int, dtype: Any) -> Any:
    import cupy as cp

    key = (n, dtype)
    if key in _BUFFER_CACHE:
        _BUFFER_KEYS.remove(key)
        _BUFFER_KEYS.append(key)
        return _BUFFER_CACHE[key]

    buffer = cp.empty((n, 2), dtype=dtype)
    _BUFFER_CACHE[key] = buffer
    _BUFFER_KEYS.append(key)
    if len(_BUFFER_KEYS) > _MAX_BUFFERS:
        old_key = _BUFFER_KEYS.pop(0)
        del _BUFFER_CACHE[old_key]
    return buffer


def fused_step(
    y: Any,
    dt: float,
    params: dict[str, Any],
    noise: Any,
    backend: BackendBase,
) -> Any:
    """Return one fused Cayley-Maruyama increment on CuPy."""
    import cupy as cp

    n = int(y.shape[0])
    if tuple(y.shape) != (n, 2):
        raise ValueError(f"VDP Cayley kernel expects state shape (n, 2), got {y.shape}")
    if tuple(noise.shape) != (n, 4):
        raise ValueError(
            f"VDP Cayley kernel expects noise shape (n, 4), got {noise.shape}"
        )

    rdtype = y.real.dtype
    if rdtype == np.float32:
        source = _CAYLEY_SOURCE.replace("$T$", "float").replace("$CT$", "float2")
        ctype = "complex<float>"
    else:
        source = _CAYLEY_SOURCE.replace("$T$", "double").replace("$CT$", "double2")
        ctype = "complex<double>"

    kernel = compile_cached_kernel("vdp_2mode_cayley_step", ctype, source)
    params_device = [
        broadcast_param(params[name], n)
        for name in ("omega_a", "omega_b", "gamma_a", "gamma_b", "Gamma", "g", "D")
    ]
    d_w = cp.asarray(noise, dtype=rdtype)
    dy = _get_buffer(n, y.dtype)

    threads = 64
    blocks = (n + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (y, d_w, *params_device, rdtype.type(dt), n, dy),
    )
    return dy


class VDP2ModeCayleyCuPyKernel:
    """CuPy fused-step provider for Cayley-Maruyama."""

    scheme = "cayley_maruyama"
    backend_name = "cupy"

    def step(
        self,
        y: Any,
        t: float,
        dt: float,
        params: dict[str, Any],
        noise: Any,
        backend: BackendBase,
    ) -> Any:
        del t
        return fused_step(y, dt, params, noise, backend)
