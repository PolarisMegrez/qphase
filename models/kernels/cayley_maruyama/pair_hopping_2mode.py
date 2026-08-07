"""CuPy Cayley-Maruyama kernels for the two-mode pair-hopping model."""

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

static __device__ __forceinline__ void advance_pair_hopping_2mode_$S$(
    $CT$* alpha,
    $CT$* beta,
    const $T$* noise,
    $T$ omega_a,
    $T$ omega_b,
    $T$ coupling,
    $T$ pair_coupling,
    $T$ gamma_a,
    $T$ gamma_b,
    $T$ dt
) {
    $CT$ a = *alpha;
    $CT$ b = *beta;
    $T$ half_dt = ($T$)0.5 * dt;

    // h_ab = g + 2 k conj(alpha) beta and h_ba = conj(h_ab).
    $T$ overlap_real = a.x * b.x + a.y * b.y;
    $T$ overlap_imag = a.x * b.y - a.y * b.x;
    $T$ h_real = coupling + ($T$)2.0 * pair_coupling * overlap_real;
    $T$ h_imag = ($T$)2.0 * pair_coupling * overlap_imag;

    $CT$ a00 = cx_make_$S$(($T$)0.5 * gamma_a, -omega_a);
    $CT$ a01 = cx_make_$S$(h_imag, -h_real);
    $CT$ a10 = cx_make_$S$(-h_imag, -h_real);
    $CT$ a11 = cx_make_$S$(-($T$)0.5 * gamma_b, -omega_b);
    $CT$ drift_a = cx_add_$S$(cx_mul_$S$(a00, a), cx_mul_$S$(a01, b));
    $CT$ drift_b = cx_add_$S$(cx_mul_$S$(a10, a), cx_mul_$S$(a11, b));

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

    $CT$ m00 = cx_sub_$S$(cx_make_$S$(($T$)1.0, ($T$)0.0),
                              cx_make_$S$(half_dt * a00.x, half_dt * a00.y));
    $CT$ m01 = cx_make_$S$(-half_dt * a01.x, -half_dt * a01.y);
    $CT$ m10 = cx_make_$S$(-half_dt * a10.x, -half_dt * a10.y);
    $CT$ m11 = cx_sub_$S$(cx_make_$S$(($T$)1.0, ($T$)0.0),
                              cx_make_$S$(half_dt * a11.x, half_dt * a11.y));
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
void __pair_hopping_2mode_cayley_step_func__(
    const $CT$* __restrict__ y,
    const $T$* __restrict__ dW,
    const $T$* __restrict__ omega_a,
    const $T$* __restrict__ omega_b,
    const $T$* __restrict__ g,
    const $T$* __restrict__ k,
    const $T$* __restrict__ gamma_a,
    const $T$* __restrict__ gamma_b,
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
    advance_pair_hopping_2mode_$S$(
        &alpha, &beta, &dW[i * 4], omega_a[i], omega_b[i], g[i], k[i],
        gamma_a[i], gamma_b[i], dt
    );
    dy[i * 2] = cx_sub_$S$(alpha, old_alpha);
    dy[i * 2 + 1] = cx_sub_$S$(beta, old_beta);
}
"""
)

_CHUNK_SOURCE = (
    _DEVICE_SOURCE
    + r"""
extern "C" __global__
void __pair_hopping_2mode_cayley_chunk_func__(
    const $CT$* __restrict__ y,
    const $T$* __restrict__ dW,
    const $T$* __restrict__ omega_a,
    const $T$* __restrict__ omega_b,
    const $T$* __restrict__ g,
    const $T$* __restrict__ k,
    const $T$* __restrict__ gamma_a,
    const $T$* __restrict__ gamma_b,
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
    $T$ wa = omega_a[i], wb = omega_b[i], coupling = g[i];
    $T$ pair_coupling = k[i], ga = gamma_a[i], gb = gamma_b[i];
    int save_cursor = 0;
    for (int step = 0; step < n_steps; ++step) {
        int noise_base = (step * n + i) * 4;
        advance_pair_hopping_2mode_$S$(
            &alpha, &beta, &dW[noise_base], wa, wb, coupling, pair_coupling,
            ga, gb, dt
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

_CHUNK_BUFFER_CACHE: dict[tuple[int, int, int, Any], tuple[Any, Any]] = {}
_CHUNK_BUFFER_KEYS: list[tuple[int, int, int, Any]] = []
_LAUNCH_REFERENCES: list[tuple[Any, tuple[Any, ...]]] = []
_PARAMETER_NAMES = ("omega_a", "omega_b", "g", "k", "gamma_a", "gamma_b")


def _typed_source(source: str, dtype: Any, operation: str) -> tuple[str, str]:
    if dtype == np.float32:
        typed = source.replace("$S$", f"f32_{operation}")
        return typed.replace("$T$", "float").replace("$CT$", "float2"), "complex<float>"
    typed = source.replace("$S$", f"f64_{operation}")
    return typed.replace("$T$", "double").replace("$CT$", "double2"), "complex<double>"


def _retain_launch(*values: Any) -> None:
    import cupy as cp

    event = cp.cuda.Event()
    event.record(cp.cuda.get_current_stream())
    _LAUNCH_REFERENCES.append((event, values))
    while _LAUNCH_REFERENCES and _LAUNCH_REFERENCES[0][0].done:
        _LAUNCH_REFERENCES.pop(0)


def fused_step(
    y: Any,
    dt: float,
    params: dict[str, Any],
    noise: Any,
    backend: BackendBase,
) -> Any:
    """Return one fused pair-hopping Cayley-Maruyama increment."""
    del backend
    import cupy as cp

    n = int(y.shape[0])
    if tuple(y.shape) != (n, 2) or tuple(noise.shape) != (n, 4):
        raise ValueError("PairHopping2 Cayley step expects (n,2) state and (n,4) noise")
    rdtype = y.real.dtype
    source, ctype = _typed_source(_STEP_SOURCE, rdtype, "step")
    precision = "f32" if rdtype == np.float32 else "f64"
    kernel_name = f"pair_hopping_2mode_cayley_step_{precision}"
    source = source.replace(
        "__pair_hopping_2mode_cayley_step_func__", f"__{kernel_name}_func__"
    )
    kernel = compile_cached_kernel(kernel_name, ctype, source)
    params_device = [
        broadcast_param(params[name], n, rdtype) for name in _PARAMETER_NAMES
    ]
    d_w = cp.asarray(noise, dtype=rdtype)
    # The caller may consume the increment asynchronously. Reusing one output
    # buffer here can race with the next launch on a different CuPy stream.
    dy = cp.empty((n, 2), dtype=y.dtype)
    threads = 32
    kernel(
        ((n + threads - 1) // threads,),
        (threads,),
        (y, d_w, *params_device, rdtype.type(dt), np.int32(n), dy),
        stream=cp.cuda.get_current_stream(),
    )
    # ``step`` returns an increment that the integrator immediately combines
    # through a separate CuPy operation. Ensure that operation cannot observe a
    # partially written RawKernel output. Production uses ``step_chunk``.
    cp.cuda.get_current_stream().synchronize()
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
    """Advance one fused pair-hopping chunk."""
    del backend
    import cupy as cp

    n = int(y.shape[0])
    if tuple(y.shape) != (n, 2):
        raise ValueError("PairHopping2 Cayley chunk expects state shape (n,2)")
    if tuple(noise.shape) != (n_steps, n, 4):
        raise ValueError(f"expected noise shape ({n_steps}, {n}, 4), got {noise.shape}")
    if not record_modes or any(mode not in (0, 1) for mode in record_modes):
        raise ValueError("record_modes must contain unique mode indices 0 or 1")
    if len(set(record_modes)) != len(record_modes):
        raise ValueError("record_modes must be unique")
    if any(offset < 1 or offset > n_steps for offset in save_offsets):
        raise ValueError("save_offsets must be within the chunk")
    if tuple(sorted(set(save_offsets))) != save_offsets:
        raise ValueError("save_offsets must be sorted and unique")

    rdtype = y.real.dtype
    source, ctype = _typed_source(_CHUNK_SOURCE, rdtype, "chunk")
    precision = "f32" if rdtype == np.float32 else "f64"
    kernel_name = f"pair_hopping_2mode_cayley_chunk_{precision}"
    source = source.replace(
        "__pair_hopping_2mode_cayley_chunk_func__", f"__{kernel_name}_func__"
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
    # The engine may recycle the noise buffer as soon as this function returns,
    # and callers may copy the cached output before issuing another kernel.
    # Complete this launch on its actual stream before either can happen.
    cp.cuda.get_current_stream().synchronize()
    _retain_launch(d_w, offsets, modes_device, *params_device)
    return final_state, saved[:, :n_saves, :]


class PairHopping2ModeCayleyCuPyKernel(ModelKernelPlugin):
    """CuPy fused provider for pair-hopping Cayley-Maruyama steps."""

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
