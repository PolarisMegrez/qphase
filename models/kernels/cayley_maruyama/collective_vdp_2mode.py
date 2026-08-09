"""CuPy Cayley-Maruyama kernels for the collective-loss VDP dimer."""

from __future__ import annotations

from typing import Any

import numpy as np
from qphase.backend.base import BackendBase

from models.kernels.base import ModelKernelPlugin
from models.kernels.cupy_utils import (
    broadcast_param,
    compile_cached_kernel,
    get_lru_buffer,
)

_DEVICE_SOURCE = r"""
static __device__ __forceinline__ $CT$ cv_make_$S$($T$ x, $T$ y) {
    $CT$ z; z.x = x; z.y = y; return z;
}

static __device__ __forceinline__ $CT$ cv_add_$S$($CT$ a, $CT$ b) {
    return cv_make_$S$(a.x + b.x, a.y + b.y);
}

static __device__ __forceinline__ $CT$ cv_sub_$S$($CT$ a, $CT$ b) {
    return cv_make_$S$(a.x - b.x, a.y - b.y);
}

static __device__ __forceinline__ $CT$ cv_mul_$S$($CT$ a, $CT$ b) {
    return cv_make_$S$(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

static __device__ __forceinline__ $CT$ cv_div_$S$($CT$ a, $CT$ b) {
    $T$ den = b.x * b.x + b.y * b.y;
    return cv_make_$S$(
        (a.x * b.x + a.y * b.y) / den,
        (a.y * b.x - a.x * b.y) / den
    );
}

static __device__ __forceinline__ void advance_collective_vdp_2mode_$S$(
    $CT$* alpha,
    $CT$* beta,
    const $T$* noise,
    $T$ omega_a,
    $T$ omega_b,
    $T$ nonlinear_loss,
    $T$ coupling,
    $T$ pump_a,
    $T$ kappa_bright,
    $T$ kappa_dark,
    $T$ dt
) {
    $CT$ a = *alpha;
    $CT$ b = *beta;
    $T$ half_dt = ($T$)0.5 * dt;
    $T$ occupation = a.x * a.x + a.y * a.y;
    $T$ mean_loss = (kappa_bright + kappa_dark) / ($T$)4.0;
    $T$ dissipative_coupling = (kappa_dark - kappa_bright) / ($T$)4.0;
    $T$ gain = nonlinear_loss * (($T$)1.0 - occupation)
        - mean_loss + pump_a / ($T$)2.0;

    $CT$ a00 = cv_make_$S$(gain, -omega_a);
    $CT$ a01 = cv_make_$S$(dissipative_coupling, -coupling);
    $CT$ a10 = a01;
    $CT$ a11 = cv_make_$S$(-mean_loss, -omega_b);
    $CT$ drift_a = cv_add_$S$(cv_mul_$S$(a00, a), cv_mul_$S$(a01, b));
    $CT$ drift_b = cv_add_$S$(cv_mul_$S$(a10, a), cv_mul_$S$(a11, b));

    $T$ d00 = ($T$)2.0 * nonlinear_loss * occupation - nonlinear_loss
        + mean_loss + pump_a / ($T$)2.0;
    $T$ d10 = (kappa_bright - kappa_dark) / ($T$)4.0;
    $T$ d11 = mean_loss;
    if (d00 < ($T$)0.0) d00 = ($T$)0.0;
    if (d11 < ($T$)0.0) d11 = ($T$)0.0;
    $T$ l00 = sqrt(d00);
    $T$ l10 = l00 > ($T$)0.0 ? d10 / l00 : ($T$)0.0;
    $T$ remainder = d11 - l10 * l10;
    if (remainder < ($T$)0.0) remainder = ($T$)0.0;
    $T$ l11 = sqrt(remainder);
    $T$ inv_sqrt_two = ($T$)1.0 / sqrt(($T$)2.0);
    $CT$ z0 = cv_make_$S$(
        noise[0] * inv_sqrt_two, noise[2] * inv_sqrt_two
    );
    $CT$ z1 = cv_make_$S$(
        noise[1] * inv_sqrt_two, noise[3] * inv_sqrt_two
    );
    $CT$ eta_a = cv_make_$S$(l00 * z0.x, l00 * z0.y);
    $CT$ eta_b = cv_add_$S$(
        cv_make_$S$(l10 * z0.x, l10 * z0.y),
        cv_make_$S$(l11 * z1.x, l11 * z1.y)
    );

    $CT$ rhs_a = cv_add_$S$(
        cv_add_$S$(a, cv_make_$S$(half_dt * drift_a.x, half_dt * drift_a.y)),
        eta_a
    );
    $CT$ rhs_b = cv_add_$S$(
        cv_add_$S$(b, cv_make_$S$(half_dt * drift_b.x, half_dt * drift_b.y)),
        eta_b
    );
    $CT$ identity = cv_make_$S$(($T$)1.0, ($T$)0.0);
    $CT$ m00 = cv_sub_$S$(
        identity, cv_make_$S$(half_dt * a00.x, half_dt * a00.y)
    );
    $CT$ m01 = cv_make_$S$(-half_dt * a01.x, -half_dt * a01.y);
    $CT$ m10 = cv_make_$S$(-half_dt * a10.x, -half_dt * a10.y);
    $CT$ m11 = cv_sub_$S$(
        identity, cv_make_$S$(half_dt * a11.x, half_dt * a11.y)
    );
    $CT$ determinant = cv_sub_$S$(cv_mul_$S$(m00, m11), cv_mul_$S$(m01, m10));
    *alpha = cv_div_$S$(
        cv_sub_$S$(cv_mul_$S$(rhs_a, m11), cv_mul_$S$(m01, rhs_b)),
        determinant
    );
    *beta = cv_div_$S$(
        cv_sub_$S$(cv_mul_$S$(m00, rhs_b), cv_mul_$S$(rhs_a, m10)),
        determinant
    );
}
"""

_STEP_SOURCE = (
    _DEVICE_SOURCE
    + r"""
extern "C" __global__
void __collective_vdp_2mode_cayley_step_func__(
    const $CT$* __restrict__ y,
    const $T$* __restrict__ dW,
    const $T$* __restrict__ omega_a,
    const $T$* __restrict__ omega_b,
    const $T$* __restrict__ Gamma,
    const $T$* __restrict__ g,
    const $T$* __restrict__ pump_a,
    const $T$* __restrict__ kappa_bright,
    const $T$* __restrict__ kappa_dark,
    $T$ dt,
    int n,
    $CT$* __restrict__ dy
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    $CT$ alpha = y[i * 2];
    $CT$ beta = y[i * 2 + 1];
    $CT$ old_alpha = alpha;
    $CT$ old_beta = beta;
    advance_collective_vdp_2mode_$S$(
        &alpha, &beta, &dW[i * 4], omega_a[i], omega_b[i], Gamma[i], g[i],
        pump_a[i], kappa_bright[i], kappa_dark[i], dt
    );
    dy[i * 2] = cv_sub_$S$(alpha, old_alpha);
    dy[i * 2 + 1] = cv_sub_$S$(beta, old_beta);
}
"""
)

_CHUNK_SOURCE = (
    _DEVICE_SOURCE
    + r"""
extern "C" __global__
void __collective_vdp_2mode_cayley_chunk_func__(
    const $CT$* __restrict__ y,
    const $T$* __restrict__ dW,
    const $T$* __restrict__ omega_a,
    const $T$* __restrict__ omega_b,
    const $T$* __restrict__ Gamma,
    const $T$* __restrict__ g,
    const $T$* __restrict__ pump_a,
    const $T$* __restrict__ kappa_bright,
    const $T$* __restrict__ kappa_dark,
    $T$ dt,
    int n_steps,
    int n,
    const int* __restrict__ save_offsets,
    int n_saves,
    const int* __restrict__ record_modes,
    int n_record_modes,
    $CT$* __restrict__ final_state,
    $CT$* __restrict__ saved
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    $CT$ alpha = y[i * 2];
    $CT$ beta = y[i * 2 + 1];
    $T$ wa = omega_a[i], wb = omega_b[i], nl = Gamma[i], coupling = g[i];
    $T$ pump = pump_a[i], bright = kappa_bright[i], dark = kappa_dark[i];
    int save_cursor = 0;
    for (int step = 0; step < n_steps; ++step) {
        int noise_base = (step * n + i) * 4;
        advance_collective_vdp_2mode_$S$(
            &alpha, &beta, &dW[noise_base], wa, wb, nl, coupling, pump,
            bright, dark, dt
        );
        if (save_cursor < n_saves && step + 1 == save_offsets[save_cursor]) {
            int save_base = (i * n_saves + save_cursor) * n_record_modes;
            for (int cursor = 0; cursor < n_record_modes; ++cursor) {
                int mode = record_modes[cursor];
                saved[save_base + cursor] = mode == 0 ? alpha : beta;
            }
            ++save_cursor;
        }
    }
    final_state[i * 2] = alpha;
    final_state[i * 2 + 1] = beta;
}
"""
)

_PARAMETER_NAMES = (
    "omega_a",
    "omega_b",
    "Gamma",
    "g",
    "pump_a",
    "kappa_bright",
    "kappa_dark",
)
_CHUNK_BUFFER_CACHE: dict[tuple[int, int, int, Any], tuple[Any, Any]] = {}
_CHUNK_BUFFER_KEYS: list[tuple[int, int, int, Any]] = []


def _typed_source(source: str, dtype: Any, operation: str) -> tuple[str, str]:
    if dtype == np.float32:
        typed = source.replace("$S$", f"f32_{operation}")
        return typed.replace("$T$", "float").replace("$CT$", "float2"), "complex<float>"
    typed = source.replace("$S$", f"f64_{operation}")
    return typed.replace("$T$", "double").replace("$CT$", "double2"), "complex<double>"


def fused_step(
    y: Any,
    dt: float,
    params: dict[str, Any],
    noise: Any,
    backend: BackendBase,
) -> Any:
    """Return one fused collective-VDP Cayley-Maruyama increment."""
    del backend
    import cupy as cp

    n = int(y.shape[0])
    if tuple(y.shape) != (n, 2) or tuple(noise.shape) != (n, 4):
        raise ValueError(
            "CollectiveVDP2 Cayley step expects (n,2) state and (n,4) noise"
        )
    rdtype = y.real.dtype
    source, ctype = _typed_source(_STEP_SOURCE, rdtype, "step")
    precision = "f32" if rdtype == np.float32 else "f64"
    kernel_name = f"collective_vdp_2mode_cayley_step_{precision}"
    source = source.replace(
        "__collective_vdp_2mode_cayley_step_func__", f"__{kernel_name}_func__"
    )
    kernel = compile_cached_kernel(kernel_name, ctype, source)
    params_device = [
        broadcast_param(params[name], n, rdtype) for name in _PARAMETER_NAMES
    ]
    d_w = cp.asarray(noise, dtype=rdtype)
    dy = cp.empty_like(y)
    threads = 32
    kernel(
        ((n + threads - 1) // threads,),
        (threads,),
        (y, d_w, *params_device, rdtype.type(dt), np.int32(n), dy),
        stream=cp.cuda.get_current_stream(),
    )
    cp.cuda.get_current_stream().synchronize()
    return dy


def fused_step_chunk(
    y: Any,
    dt: float,
    params: dict[str, Any],
    noise: Any,
    backend: BackendBase,
    *,
    n_steps: int,
    save_offsets: tuple[int, ...],
    record_modes: tuple[int, ...],
) -> tuple[Any, Any]:
    """Advance one fused collective-VDP chunk."""
    del backend
    import cupy as cp

    n = int(y.shape[0])
    if tuple(y.shape) != (n, 2):
        raise ValueError("CollectiveVDP2 Cayley chunk expects state shape (n,2)")
    if tuple(noise.shape) != (n_steps, n, 4):
        raise ValueError(f"expected noise shape ({n_steps}, {n}, 4), got {noise.shape}")
    if not record_modes or any(mode not in (0, 1) for mode in record_modes):
        raise ValueError("record_modes must contain mode indices 0 or 1")
    if len(set(record_modes)) != len(record_modes):
        raise ValueError("record_modes must be unique")
    if any(offset < 1 or offset > n_steps for offset in save_offsets):
        raise ValueError("save_offsets must be within the chunk")
    if tuple(sorted(set(save_offsets))) != save_offsets:
        raise ValueError("save_offsets must be sorted and unique")

    rdtype = y.real.dtype
    source, ctype = _typed_source(_CHUNK_SOURCE, rdtype, "chunk")
    precision = "f32" if rdtype == np.float32 else "f64"
    kernel_name = f"collective_vdp_2mode_cayley_chunk_{precision}"
    source = source.replace(
        "__collective_vdp_2mode_cayley_chunk_func__", f"__{kernel_name}_func__"
    )
    kernel = compile_cached_kernel(kernel_name, ctype, source)
    params_device = [
        broadcast_param(params[name], n, rdtype) for name in _PARAMETER_NAMES
    ]
    d_w = cp.asarray(noise, dtype=rdtype)
    offsets = cp.asarray(save_offsets or (0,), dtype=cp.int32)
    modes_device = cp.asarray(record_modes, dtype=cp.int32)
    n_saves, n_record_modes = len(save_offsets), len(record_modes)
    key = (n, n_saves, n_record_modes, y.dtype)
    final_state, saved = get_lru_buffer(
        _CHUNK_BUFFER_CACHE,
        _CHUNK_BUFFER_KEYS,
        key,
        lambda: (
            cp.empty_like(y),
            cp.empty((n, max(1, n_saves), n_record_modes), dtype=y.dtype),
        ),
    )
    threads = 32
    kernel(
        ((n + threads - 1) // threads,),
        (threads,),
        (
            y,
            d_w,
            *params_device,
            rdtype.type(dt),
            np.int32(n_steps),
            np.int32(n),
            offsets,
            np.int32(n_saves),
            modes_device,
            np.int32(n_record_modes),
            final_state,
            saved,
        ),
        stream=cp.cuda.get_current_stream(),
    )
    cp.cuda.get_current_stream().synchronize()
    return final_state, saved[:, :n_saves, :]


class CollectiveVDP2ModeCayleyCuPyKernel(ModelKernelPlugin):
    """CuPy fused provider for collective-VDP Cayley-Maruyama steps."""

    scheme = "cayley_maruyama"
    backend_name = "cupy"
    operations = frozenset({"step", "step_chunk"})

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

    def step_chunk(
        self,
        y: Any,
        t: float,
        dt: float,
        params: dict[str, Any],
        noise: Any,
        backend: BackendBase,
        *,
        n_steps: int,
        save_offsets: tuple[int, ...],
        record_modes: tuple[int, ...],
    ) -> tuple[Any, Any]:
        del t
        return fused_step_chunk(
            y,
            dt,
            params,
            noise,
            backend,
            n_steps=n_steps,
            save_offsets=save_offsets,
            record_modes=record_modes,
        )
