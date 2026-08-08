"""fpgen-driven CuPy Cayley-Maruyama kernel generation."""

from __future__ import annotations

from functools import cached_property
from typing import Any, ClassVar

import numpy as np
from qphase.backend.base import BackendBase

from models.kernels.base import ModelKernelPlugin
from models.kernels.cupy_utils import (
    broadcast_param,
    compile_cached_kernel,
    get_lru_buffer,
)

__all__ = ["FPGenCayleyCuPyKernel"]

_LAUNCH_REFERENCES: list[tuple[Any, ...]] = []

_DEVICE_TEMPLATE = r"""
// Model equations generated from fpgen fingerprint: $FINGERPRINT$
#define QP_N_MODES $N_MODES$

static __device__ __forceinline__ $CT$ fg_make_$S$($T$ x, $T$ y) {
    $CT$ z; z.x = x; z.y = y; return z;
}

static __device__ __forceinline__ $CT$ fg_add_$S$($CT$ a, $CT$ b) {
    return fg_make_$S$(a.x + b.x, a.y + b.y);
}

static __device__ __forceinline__ $CT$ fg_sub_$S$($CT$ a, $CT$ b) {
    return fg_make_$S$(a.x - b.x, a.y - b.y);
}

static __device__ __forceinline__ $CT$ fg_mul_$S$($CT$ a, $CT$ b) {
    return fg_make_$S$(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

static __device__ __forceinline__ $CT$ fg_div_$S$($CT$ a, $CT$ b) {
    $T$ den = b.x * b.x + b.y * b.y;
    return fg_make_$S$(
        (a.x * b.x + a.y * b.y) / den,
        (a.y * b.x - a.x * b.y) / den
    );
}

static __device__ __forceinline__ $T$ fg_abs2_$S$($CT$ value) {
    return value.x * value.x + value.y * value.y;
}

static __device__ __forceinline__ void advance_fpgen_$S$(
    $CT$* state,
    const $T$* noise,
    const $CT$* diffusion_factor,
$ADVANCE_PARAMETERS$
    $T$ dt
) {
    $T$ half_dt = ($T$)0.5 * dt;
$COORDINATES$
    $CT$ drift_matrix[QP_N_MODES][QP_N_MODES];
$DRIFT_MATRIX$

    $CT$ rhs[QP_N_MODES];
    $CT$ lhs[QP_N_MODES][QP_N_MODES];
    $T$ inv_sqrt_two = ($T$)1.0 / sqrt(($T$)2.0);
    for (int row = 0; row < QP_N_MODES; ++row) {
        $CT$ drift = fg_make_$S$(($T$)0.0, ($T$)0.0);
        $CT$ stochastic = fg_make_$S$(($T$)0.0, ($T$)0.0);
        for (int column = 0; column < QP_N_MODES; ++column) {
            drift = fg_add_$S$(
                drift, fg_mul_$S$(drift_matrix[row][column], state[column])
            );
            $CT$ complex_noise = fg_make_$S$(
                noise[column] * inv_sqrt_two,
                noise[QP_N_MODES + column] * inv_sqrt_two
            );
            stochastic = fg_add_$S$(
                stochastic,
                fg_mul_$S$(
                    diffusion_factor[row * QP_N_MODES + column], complex_noise
                )
            );
            $CT$ identity = fg_make_$S$(
                row == column ? ($T$)1.0 : ($T$)0.0, ($T$)0.0
            );
            $CT$ scaled = fg_make_$S$(
                half_dt * drift_matrix[row][column].x,
                half_dt * drift_matrix[row][column].y
            );
            lhs[row][column] = fg_sub_$S$(identity, scaled);
        }
        rhs[row] = fg_add_$S$(
            fg_add_$S$(
                state[row], fg_make_$S$(half_dt * drift.x, half_dt * drift.y)
            ),
            stochastic
        );
    }

    // Fixed-size complex Gaussian elimination with partial pivoting.
    for (int pivot = 0; pivot < QP_N_MODES; ++pivot) {
        int pivot_row = pivot;
        $T$ pivot_norm = fg_abs2_$S$(lhs[pivot][pivot]);
        for (int row = pivot + 1; row < QP_N_MODES; ++row) {
            $T$ candidate = fg_abs2_$S$(lhs[row][pivot]);
            if (candidate > pivot_norm) {
                pivot_norm = candidate;
                pivot_row = row;
            }
        }
        if (pivot_row != pivot) {
            for (int column = pivot; column < QP_N_MODES; ++column) {
                $CT$ temporary = lhs[pivot][column];
                lhs[pivot][column] = lhs[pivot_row][column];
                lhs[pivot_row][column] = temporary;
            }
            $CT$ temporary = rhs[pivot];
            rhs[pivot] = rhs[pivot_row];
            rhs[pivot_row] = temporary;
        }
        $CT$ divisor = lhs[pivot][pivot];
        for (int column = pivot; column < QP_N_MODES; ++column) {
            lhs[pivot][column] = fg_div_$S$(lhs[pivot][column], divisor);
        }
        rhs[pivot] = fg_div_$S$(rhs[pivot], divisor);
        for (int row = 0; row < QP_N_MODES; ++row) {
            if (row == pivot) continue;
            $CT$ factor = lhs[row][pivot];
            for (int column = pivot; column < QP_N_MODES; ++column) {
                lhs[row][column] = fg_sub_$S$(
                    lhs[row][column], fg_mul_$S$(factor, lhs[pivot][column])
                );
            }
            rhs[row] = fg_sub_$S$(rhs[row], fg_mul_$S$(factor, rhs[pivot]));
        }
    }
    for (int mode = 0; mode < QP_N_MODES; ++mode) state[mode] = rhs[mode];
}
"""


def _canonical_coordinate_code(n_modes: int, coordinate_count: int) -> str:
    pairs = tuple(
        (i, j) for i in range(n_modes) for j in range(i + 1, n_modes)
    )
    lines = [
        f"    $T$ c{mode} = fg_abs2_$S$(state[{mode}]);"
        for mode in range(n_modes)
    ]
    for offset, (i, j) in enumerate(pairs, start=n_modes):
        lines.append(
            f"    $T$ c{offset} = state[{i}].x * state[{j}].x "
            f"+ state[{i}].y * state[{j}].y;"
        )
    for offset, (i, j) in enumerate(
        pairs, start=n_modes + len(pairs)
    ):
        lines.append(
            f"    $T$ c{offset} = state[{i}].y * state[{j}].x "
            f"- state[{i}].x * state[{j}].y;"
        )
    if len(lines) != coordinate_count:
        raise ValueError(
            "fpgen coordinate count does not match the canonical Hermitian layout"
        )
    return "\n".join(lines)


def _cuda_expression(expression: Any) -> str:
    import sympy as sp

    source = sp.ccode(sp.simplify(expression))
    return source.replace("M_SQRT2", "sqrt(($T$)2.0)")


def _fpgen_device_source(model: Any) -> tuple[str, tuple[str, ...]]:
    import sympy as sp

    dynamics = model.cam_fpgen_dynamics()
    n_modes = int(model.n_modes)
    hamiltonian = sp.Matrix(dynamics.hamiltonian)
    diffusion = sp.Matrix(dynamics.diffusion)
    if hamiltonian.shape != (n_modes, n_modes):
        raise ValueError("fpgen Hamiltonian shape does not match the model")
    coordinate_symbols = tuple(dynamics.coordinates)
    coordinate_set = set(coordinate_symbols)
    if any(expression.free_symbols & coordinate_set for expression in diffusion):
        raise ValueError(
            "fused fpgen Cayley chunks currently require state-independent diffusion"
        )

    parameter_names = tuple(item.symbol.name for item in dynamics.parameter_spec)
    substitutions = {
        symbol: sp.Symbol(f"c{index}", real=True)
        for index, symbol in enumerate(coordinate_symbols)
    }
    substitutions.update(
        {
            item.symbol: sp.Symbol(f"p{index}", real=True)
            for index, item in enumerate(dynamics.parameter_spec)
        }
    )
    drift_matrix = (-sp.I * hamiltonian).xreplace(substitutions)
    assignments = []
    for row in range(n_modes):
        for column in range(n_modes):
            value = sp.expand_complex(drift_matrix[row, column])
            real = _cuda_expression(sp.re(value))
            imag = _cuda_expression(sp.im(value))
            assignments.append(
                "    drift_matrix"
                f"[{row}][{column}] = fg_make_$S$(({real}), ({imag}));"
            )

    advance_parameters = "".join(
        f"    $T$ p{index},\n" for index in range(len(parameter_names))
    )
    fingerprint = dynamics.to_model_spec(name=model.name).fingerprint
    source = (
        _DEVICE_TEMPLATE.replace("$FINGERPRINT$", fingerprint)
        .replace("$N_MODES$", str(n_modes))
        .replace("$ADVANCE_PARAMETERS$", advance_parameters)
        .replace(
            "$COORDINATES$",
            _canonical_coordinate_code(n_modes, len(coordinate_symbols)),
        )
        .replace("$DRIFT_MATRIX$", "\n".join(assignments))
    )
    return source, parameter_names


def _step_source(
    kernel_name: str, device_source: str, n_modes: int, n_params: int
) -> str:
    parameter_declarations = "".join(
        f"    const $T$* __restrict__ p{index},\n" for index in range(n_params)
    )
    parameter_values = "".join(f"        p{index}[i],\n" for index in range(n_params))
    return (
        device_source
        + r"""
extern "C" __global__
void __$KERNEL_NAME$_func__(
    const $CT$* __restrict__ y,
    const $T$* __restrict__ dW,
    const $CT$* __restrict__ diffusion_factor,
$PARAMETER_DECLARATIONS$
    $T$ dt,
    int n,
    $CT$* __restrict__ dy
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    $CT$ state[QP_N_MODES];
    $CT$ previous[QP_N_MODES];
    for (int mode = 0; mode < QP_N_MODES; ++mode) {
        state[mode] = y[i * QP_N_MODES + mode];
        previous[mode] = state[mode];
    }
    advance_fpgen_$S$(
        state,
        &dW[i * (2 * QP_N_MODES)],
        &diffusion_factor[i * QP_N_MODES * QP_N_MODES],
$PARAMETER_VALUES$
        dt
    );
    for (int mode = 0; mode < QP_N_MODES; ++mode) {
        dy[i * QP_N_MODES + mode] = fg_sub_$S$(state[mode], previous[mode]);
    }
}
"""
    ).replace("$KERNEL_NAME$", kernel_name).replace(
        "$PARAMETER_DECLARATIONS$", parameter_declarations
    ).replace("$PARAMETER_VALUES$", parameter_values)


def _chunk_source(
    kernel_name: str, device_source: str, n_modes: int, n_params: int
) -> str:
    parameter_declarations = "".join(
        f"    const $T$* __restrict__ p{index},\n" for index in range(n_params)
    )
    parameter_locals = "".join(
        f"    $T$ q{index} = p{index}[i];\n" for index in range(n_params)
    )
    parameter_values = "".join(f"            q{index},\n" for index in range(n_params))
    return (
        device_source
        + r"""
extern "C" __global__
void __$KERNEL_NAME$_func__(
    const $CT$* __restrict__ y,
    const $T$* __restrict__ dW,
    const $CT$* __restrict__ diffusion_factor,
$PARAMETER_DECLARATIONS$
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
    $CT$ state[QP_N_MODES];
    for (int mode = 0; mode < QP_N_MODES; ++mode) {
        state[mode] = y[i * QP_N_MODES + mode];
    }
$PARAMETER_LOCALS$
    int save_cursor = 0;
    for (int step = 0; step < n_steps; ++step) {
        int noise_base = (step * n + i) * (2 * QP_N_MODES);
        advance_fpgen_$S$(
            state,
            &dW[noise_base],
            &diffusion_factor[i * QP_N_MODES * QP_N_MODES],
$PARAMETER_VALUES$
            dt
        );
        if (save_cursor < n_saves && step + 1 == save_offsets[save_cursor]) {
            int save_base = (i * n_saves + save_cursor) * n_record_modes;
            for (int cursor = 0; cursor < n_record_modes; ++cursor) {
                saved[save_base + cursor] = state[record_modes[cursor]];
            }
            ++save_cursor;
        }
    }
    for (int mode = 0; mode < QP_N_MODES; ++mode) {
        final_state[i * QP_N_MODES + mode] = state[mode];
    }
}
"""
    ).replace("$KERNEL_NAME$", kernel_name).replace(
        "$PARAMETER_DECLARATIONS$", parameter_declarations
    ).replace("$PARAMETER_LOCALS$", parameter_locals).replace(
        "$PARAMETER_VALUES$", parameter_values
    )


def _typed_source(source: str, dtype: Any, operation: str) -> tuple[str, str]:
    if dtype == np.float32:
        typed = source.replace("$S$", f"f32_{operation}")
        typed = typed.replace("$T$", "float").replace("$CT$", "float2")
        return typed, "complex<float>"
    typed = source.replace("$S$", f"f64_{operation}")
    typed = typed.replace("$T$", "double").replace("$CT$", "double2")
    return typed, "complex<double>"


class FPGenCayleyCuPyKernel(ModelKernelPlugin):
    """Generate fused Cayley kernels from one fpgen-backed model."""

    scheme = "cayley_maruyama"
    backend_name = "cupy"
    operations = frozenset({"step", "step_chunk"})
    kernel_slug: ClassVar[str]
    mode_count: ClassVar[int]

    def __init__(self, model: Any) -> None:
        self._model = model
        self._step_buffers: dict[tuple[int, Any], Any] = {}
        self._step_buffer_keys: list[tuple[int, Any]] = []
        self._chunk_buffers: dict[tuple[int, int, int, Any], tuple[Any, Any]] = {}
        self._chunk_buffer_keys: list[tuple[int, int, int, Any]] = []
        super().__init__()
        if int(model.n_modes) != self.mode_count:
            raise ValueError("kernel mode count does not match its fpgen model")

    @cached_property
    def _generated(self) -> tuple[str, str, tuple[str, ...]]:
        device, parameter_names = _fpgen_device_source(self._model)
        step_name = f"{self.kernel_slug}_cayley_step"
        chunk_name = f"{self.kernel_slug}_cayley_chunk"
        return (
            _step_source(step_name, device, self.mode_count, len(parameter_names)),
            _chunk_source(
                chunk_name, device, self.mode_count, len(parameter_names)
            ),
            parameter_names,
        )

    def _retain_launch(self, *values: Any) -> None:
        _LAUNCH_REFERENCES.append(values)
        if len(_LAUNCH_REFERENCES) > 8:
            _LAUNCH_REFERENCES.pop(0)

    def _parameter_arrays(
        self, params: dict[str, Any], n: int, dtype: Any
    ) -> list[Any]:
        return [
            broadcast_param(params[name], n, dtype) for name in self._generated[2]
        ]

    def _diffusion_factor(self, y: Any, params: dict[str, Any]) -> Any:
        import cupy as cp

        factor = cp.ascontiguousarray(self._model.diffusion(y, 0.0, params))
        expected = (int(y.shape[0]), self.mode_count, self.mode_count)
        if tuple(factor.shape) != expected:
            raise ValueError(
                f"fpgen diffusion factor must have shape {expected}; got {factor.shape}"
            )
        return factor

    def step(
        self,
        y: Any,
        t: float,
        dt: float,
        params: dict[str, Any],
        noise: Any,
        backend: BackendBase,
    ) -> Any:
        del t, backend
        import cupy as cp

        n = int(y.shape[0])
        expected_state = (n, self.mode_count)
        expected_noise = (n, 2 * self.mode_count)
        if tuple(y.shape) != expected_state or tuple(noise.shape) != expected_noise:
            raise ValueError(
                f"{self.kernel_slug} Cayley step expects state {expected_state} "
                f"and noise {expected_noise}; got {y.shape} and {noise.shape}"
            )
        rdtype = y.real.dtype
        source, ctype = _typed_source(
            self._generated[0], rdtype, f"{self.kernel_slug}_step"
        )
        kernel_name = f"{self.kernel_slug}_cayley_step"
        kernel = compile_cached_kernel(kernel_name, ctype, source)
        parameters = self._parameter_arrays(params, n, rdtype)
        d_w = cp.ascontiguousarray(noise, dtype=rdtype)
        diffusion_factor = self._diffusion_factor(y, params)
        key = (n, y.dtype)
        dy = get_lru_buffer(
            self._step_buffers,
            self._step_buffer_keys,
            key,
            lambda: cp.empty_like(y),
        )
        threads = 64
        kernel(
            ((n + threads - 1) // threads,),
            (threads,),
            (
                y,
                d_w,
                diffusion_factor,
                *parameters,
                rdtype.type(dt),
                n,
                dy,
            ),
        )
        self._retain_launch(d_w, diffusion_factor, *parameters)
        return dy

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
        del t, backend
        import cupy as cp

        n = int(y.shape[0])
        expected_state = (n, self.mode_count)
        expected_noise = (n_steps, n, 2 * self.mode_count)
        if tuple(y.shape) != expected_state or tuple(noise.shape) != expected_noise:
            raise ValueError(
                f"{self.kernel_slug} Cayley chunk expects state {expected_state} "
                f"and noise {expected_noise}; got {y.shape} and {noise.shape}"
            )
        if not record_modes or len(set(record_modes)) != len(record_modes):
            raise ValueError("record_modes must be non-empty and unique")
        if any(mode < 0 or mode >= self.mode_count for mode in record_modes):
            raise ValueError("record_modes contains an out-of-range mode")
        if any(offset < 1 or offset > n_steps for offset in save_offsets):
            raise ValueError("save_offsets must be within the chunk")
        if tuple(sorted(set(save_offsets))) != save_offsets:
            raise ValueError("save_offsets must be sorted and unique")

        rdtype = y.real.dtype
        source, ctype = _typed_source(
            self._generated[1], rdtype, f"{self.kernel_slug}_chunk"
        )
        kernel_name = f"{self.kernel_slug}_cayley_chunk"
        kernel = compile_cached_kernel(kernel_name, ctype, source)
        parameters = self._parameter_arrays(params, n, rdtype)
        d_w = cp.ascontiguousarray(noise, dtype=rdtype)
        diffusion_factor = self._diffusion_factor(y, params)
        offsets = cp.asarray(save_offsets or (0,), dtype=cp.int32)
        modes_device = cp.asarray(record_modes, dtype=cp.int32)
        n_saves = len(save_offsets)
        n_record_modes = len(record_modes)
        key = (n, n_saves, n_record_modes, y.dtype)
        final_state, saved_storage = get_lru_buffer(
            self._chunk_buffers,
            self._chunk_buffer_keys,
            key,
            lambda: (
                cp.empty_like(y),
                cp.empty(
                    (n, max(1, n_saves), n_record_modes), dtype=y.dtype
                ),
            ),
        )
        threads = 64
        kernel(
            ((n + threads - 1) // threads,),
            (threads,),
            (
                y,
                d_w,
                diffusion_factor,
                *parameters,
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
        self._retain_launch(
            d_w, diffusion_factor, offsets, modes_device, *parameters
        )
        return final_state, saved_storage[:, :n_saves, :]
