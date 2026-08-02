"""CuPy Cayley-Maruyama kernels for the two-mode cross-Kerr model."""

from __future__ import annotations

from typing import Any

import numpy as np
from qphase.backend.base import BackendBase
from qphase_sde.kernels import compile_cached_kernel

from models.kernels.base import ModelKernelPlugin
from models.kernels.cupy_utils import broadcast_param, get_lru_buffer

_DEVICE_SOURCE = r"""
static __device__ __forceinline__ $CT$ cx_make_$S$($T$ x, $T$ y) {
    $CT$ z; z.x = x; z.y = y; return z;
}

static __device__ __forceinline__ $CT$ cx_add_$S$($CT$ a, $CT$ b) {
    return cx_make_$S$(a.x + b.x, a.y + b.y);
}

static __device__ __forceinline__ $CT$ cx_sub_$S$($CT$ a, $CT$ b) {
    return cx_make_$S$(a.x - b.x, a.y - b.y);
}

static __device__ __forceinline__ $CT$ cx_mul_$S$($CT$ a, $CT$ b) {
    return cx_make_$S$(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

static __device__ __forceinline__ $CT$ cx_div_$S$($CT$ a, $CT$ b) {
    $T$ den = b.x * b.x + b.y * b.y;
    return cx_make_$S$(
        (a.x * b.x + a.y * b.y) / den,
        (a.y * b.x - a.x * b.y) / den
    );
}

static __device__ __forceinline__ void advance_crosskerr_2mode_$S$(
    $CT$* alpha,
    $CT$* beta,
    const $T$* noise,
    $T$ omega_a,
    $T$ omega_b,
    $T$ chi,
    $T$ gamma_a,
    $T$ gamma_b,
    $T$ coupling,
    $T$ dt
) {
    $CT$ a = *alpha;
    $CT$ b = *beta;
    $T$ half_dt = ($T$)0.5 * dt;
    $T$ frequency_a = omega_a + chi * (
        b.x * b.x + b.y * b.y - ($T$)0.5
    );
    $T$ frequency_b = omega_b + chi * (
        a.x * a.x + a.y * a.y - ($T$)0.5
    );

    $CT$ drift_a = cx_make_$S$(
        ($T$)0.5 * gamma_a * a.x + frequency_a * a.y + coupling * b.y,
        ($T$)0.5 * gamma_a * a.y - frequency_a * a.x - coupling * b.x
    );
    $CT$ drift_b = cx_make_$S$(
        -($T$)0.5 * gamma_b * b.x + frequency_b * b.y + coupling * a.y,
        -($T$)0.5 * gamma_b * b.y - frequency_b * b.x - coupling * a.x
    );

    $T$ sigma_a = sqrt(gamma_a / ($T$)4.0);
    $T$ sigma_b = sqrt(gamma_b / ($T$)4.0);
    $CT$ eta_a = cx_make_$S$(sigma_a * noise[0], sigma_a * noise[2]);
    $CT$ eta_b = cx_make_$S$(sigma_b * noise[1], sigma_b * noise[3]);
    $CT$ rhs_a = cx_add_$S$(
        cx_add_$S$(a, cx_make_$S$(half_dt * drift_a.x, half_dt * drift_a.y)),
        eta_a
    );
    $CT$ rhs_b = cx_add_$S$(
        cx_add_$S$(b, cx_make_$S$(half_dt * drift_b.x, half_dt * drift_b.y)),
        eta_b
    );

    $CT$ m00 = cx_make_$S$(
        ($T$)1.0 - half_dt * gamma_a / ($T$)2.0,
        half_dt * frequency_a
    );
    $CT$ m01 = cx_make_$S$(($T$)0.0, half_dt * coupling);
    $CT$ m10 = m01;
    $CT$ m11 = cx_make_$S$(
        ($T$)1.0 + half_dt * gamma_b / ($T$)2.0,
        half_dt * frequency_b
    );
    $CT$ det = cx_sub_$S$(cx_mul_$S$(m00, m11), cx_mul_$S$(m01, m10));
    *alpha = cx_div_$S$(
        cx_sub_$S$(cx_mul_$S$(rhs_a, m11), cx_mul_$S$(m01, rhs_b)), det
    );
    *beta = cx_div_$S$(
        cx_sub_$S$(cx_mul_$S$(m00, rhs_b), cx_mul_$S$(rhs_a, m10)), det
    );
}
"""

_STEP_SOURCE = (
    _DEVICE_SOURCE
    + r"""
extern "C" __global__
void __crosskerr_2mode_cayley_step_func__(
    const $CT$* __restrict__ y,
    const $T$* __restrict__ dW,
    const $T$* __restrict__ omega_a,
    const $T$* __restrict__ omega_b,
    const $T$* __restrict__ chi,
    const $T$* __restrict__ gamma_a,
    const $T$* __restrict__ gamma_b,
    const $T$* __restrict__ g,
    $T$ dt,
    int n,
    $CT$* __restrict__ dy
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    $CT$ alpha = y[i * 2 + 0];
    $CT$ beta = y[i * 2 + 1];
    $CT$ old_alpha = alpha;
    $CT$ old_beta = beta;
    advance_crosskerr_2mode_$S$(
        &alpha, &beta, &dW[i * 4],
        ($T$)omega_a[i], ($T$)omega_b[i], ($T$)chi[i],
        ($T$)gamma_a[i], ($T$)gamma_b[i], ($T$)g[i], dt
    );
    dy[i * 2 + 0] = cx_sub_$S$(alpha, old_alpha);
    dy[i * 2 + 1] = cx_sub_$S$(beta, old_beta);
}
"""
)

_CHUNK_SOURCE = (
    _DEVICE_SOURCE
    + r"""
extern "C" __global__
void __crosskerr_2mode_cayley_chunk_func__(
    const $CT$* __restrict__ y,
    const $T$* __restrict__ dW,
    const $T$* __restrict__ omega_a,
    const $T$* __restrict__ omega_b,
    const $T$* __restrict__ chi,
    const $T$* __restrict__ gamma_a,
    const $T$* __restrict__ gamma_b,
    const $T$* __restrict__ g,
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
    $CT$ alpha = y[i * 2 + 0];
    $CT$ beta = y[i * 2 + 1];
    $T$ wa = ($T$)omega_a[i];
    $T$ wb = ($T$)omega_b[i];
    $T$ ch = ($T$)chi[i];
    $T$ ga = ($T$)gamma_a[i];
    $T$ gb = ($T$)gamma_b[i];
    $T$ coupling = ($T$)g[i];
    int save_cursor = 0;

    for (int step = 0; step < n_steps; ++step) {
        int noise_base = (step * n + i) * 4;
        advance_crosskerr_2mode_$S$(
            &alpha, &beta, &dW[noise_base],
            wa, wb, ch, ga, gb, coupling, dt
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
    final_state[i * 2 + 0] = alpha;
    final_state[i * 2 + 1] = beta;
}
"""
)

_BUFFER_CACHE: dict[tuple[int, Any], Any] = {}
_BUFFER_KEYS: list[tuple[int, Any]] = []
_MAX_BUFFERS = 2
_CHUNK_BUFFER_CACHE: dict[tuple[int, int, int, Any], tuple[Any, Any]] = {}
_CHUNK_BUFFER_KEYS: list[tuple[int, int, int, Any]] = []
_LAUNCH_REFERENCES: list[tuple[Any, ...]] = []
_PARAMETER_NAMES = ("omega_a", "omega_b", "chi", "gamma_a", "gamma_b", "g")


def _typed_source(source: str, dtype: Any, operation: str) -> tuple[str, str]:
    if dtype == np.float32:
        typed = source.replace("$S$", f"f32_{operation}")
        typed = typed.replace("$T$", "float").replace("$CT$", "float2")
        return typed, "complex<float>"
    typed = source.replace("$S$", f"f64_{operation}")
    typed = typed.replace("$T$", "double").replace("$CT$", "double2")
    return typed, "complex<double>"


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
        del _BUFFER_CACHE[_BUFFER_KEYS.pop(0)]
    return buffer


def _retain_launch(*values: Any) -> None:
    _LAUNCH_REFERENCES.append(values)
    if len(_LAUNCH_REFERENCES) > 4:
        _LAUNCH_REFERENCES.pop(0)


def fused_step(
    y: Any,
    dt: float,
    params: dict[str, Any],
    noise: Any,
    backend: BackendBase,
) -> Any:
    """Return one fused cross-Kerr Cayley-Maruyama increment on CuPy."""
    del backend
    import cupy as cp

    n = int(y.shape[0])
    if tuple(y.shape) != (n, 2):
        raise ValueError(
            f"CrossKerr2 Cayley kernel expects state shape (n, 2), got {y.shape}"
        )
    if tuple(noise.shape) != (n, 4):
        raise ValueError(
            f"CrossKerr2 Cayley kernel expects noise shape (n, 4), got {noise.shape}"
        )
    rdtype = y.real.dtype
    source, ctype = _typed_source(_STEP_SOURCE, rdtype, "step")
    kernel = compile_cached_kernel("crosskerr_2mode_cayley_step", ctype, source)
    params_device = [
        broadcast_param(params[name], n, rdtype) for name in _PARAMETER_NAMES
    ]
    d_w = cp.asarray(noise, dtype=rdtype)
    dy = _get_buffer(n, y.dtype)
    threads = 64
    kernel(
        ((n + threads - 1) // threads,),
        (threads,),
        (y, d_w, *params_device, rdtype.type(dt), n, dy),
    )
    _retain_launch(d_w, *params_device)
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
    """Advance a fused cross-Kerr chunk and return final and saved states."""
    del backend
    import cupy as cp

    n = int(y.shape[0])
    if tuple(y.shape) != (n, 2):
        raise ValueError(
            f"CrossKerr2 Cayley kernel expects state shape (n, 2), got {y.shape}"
        )
    if not record_modes or len(set(record_modes)) != len(record_modes):
        raise ValueError("record_modes must be non-empty and unique")
    if any(mode not in (0, 1) for mode in record_modes):
        raise ValueError("CrossKerr2 chunk record_modes must contain only 0 or 1")
    if tuple(noise.shape) != (n_steps, n, 4):
        raise ValueError(
            "CrossKerr2 Cayley chunk noise must have shape "
            f"({n_steps}, {n}, 4); got {noise.shape}"
        )
    if any(offset < 1 or offset > n_steps for offset in save_offsets):
        raise ValueError("save_offsets must be within the chunk")
    if tuple(sorted(set(save_offsets))) != save_offsets:
        raise ValueError("save_offsets must be sorted and unique")

    rdtype = y.real.dtype
    source, ctype = _typed_source(_CHUNK_SOURCE, rdtype, "chunk")
    kernel = compile_cached_kernel("crosskerr_2mode_cayley_chunk", ctype, source)
    params_device = [
        broadcast_param(params[name], n, rdtype) for name in _PARAMETER_NAMES
    ]
    d_w = cp.asarray(noise, dtype=rdtype)
    offsets = cp.asarray(save_offsets or (0,), dtype=cp.int32)
    modes_device = cp.asarray(record_modes, dtype=cp.int32)
    n_saves = len(save_offsets)
    n_record_modes = len(record_modes)
    key = (n, n_saves, n_record_modes, y.dtype)
    final_state, saved_storage = get_lru_buffer(
        _CHUNK_BUFFER_CACHE,
        _CHUNK_BUFFER_KEYS,
        key,
        lambda: (
            cp.empty_like(y),
            cp.empty((n, max(1, n_saves), n_record_modes), dtype=y.dtype),
        ),
    )
    threads = 64
    kernel(
        ((n + threads - 1) // threads,),
        (threads,),
        (
            y,
            d_w,
            *params_device,
            rdtype.type(dt),
            n_steps,
            n,
            offsets,
            n_saves,
            modes_device,
            n_record_modes,
            final_state,
            saved_storage,
        ),
    )
    _retain_launch(d_w, offsets, modes_device, *params_device)
    return final_state, saved_storage[:, :n_saves, :]


class CrossKerr2ModeCayleyCuPyKernel(ModelKernelPlugin):
    """CuPy fused provider for cross-Kerr Cayley-Maruyama steps."""

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
