"""Memory-budget and workspace accounting for the SDE execution planner.

The numbers here decide how a scan is split into tiles and trajectory
batches; they read backend capabilities but never mutate the plan.
"""

from __future__ import annotations

import math
from typing import Any

from .plans import _AnalyserWorkspace

_MIB = 1024**2
_DEFAULT_DEVICE_FRACTION = 0.75
_MAX_RESERVE_BYTES = 256 * _MIB


def _validate_analyser_sampling(config: Any, analysers: dict[str, Any]) -> None:
    for name, analyser in analysers.items():
        analyser_config = getattr(analyser, "config", None)
        expected = getattr(analyser_config, "expected_freq_max", None)
        if expected is None:
            continue
        override_dt = getattr(analyser_config, "dt", None)
        sample_dt = (
            float(override_dt)
            if override_dt is not None
            else float(config.dt) * int(config.save_stride)
        )
        convention = getattr(analyser_config, "convention", "symmetric")
        nyquist = (
            math.pi / sample_dt
            if convention in {"symmetric", "unitary"}
            else 0.5 / sample_dt
        )
        if float(expected) >= nyquist:
            factor = math.pi if convention in {"symmetric", "unitary"} else 0.5
            if override_dt is None:
                maximum_stride = max(
                    0,
                    math.ceil(factor / (float(config.dt) * float(expected))) - 1,
                )
                remedy = f"use save_stride <= {maximum_stride}"
            else:
                remedy = "reduce the analyser dt override"
            raise ValueError(
                f"analyser {name!r} expected_freq_max={float(expected):.6g} "
                f"reaches or exceeds the Nyquist limit {nyquist:.6g} for "
                f"sample dt={sample_dt:.6g}; {remedy}"
            )

def _memory_budget(
    backend_name: str,
    resources: Any | None,
    *,
    device_memory: tuple[int, int] | None,
) -> tuple[str, int | None, int | None, int | None, float | None]:
    if backend_name == "cupy":
        fraction = getattr(resources, "gpu_memory_fraction", None)
        fraction = _DEFAULT_DEVICE_FRACTION if fraction is None else float(fraction)
        memory = device_memory if device_memory is not None else _cupy_memory_info()
        if memory is None:
            return "gpu", None, None, None, fraction
        free, total = (int(memory[0]), int(memory[1]))
        budget = min(free, int(total * fraction))
        return "gpu", budget, free, total, fraction

    limit_mib = getattr(resources, "memory_limit_mib", None)
    hardware = getattr(resources, "hardware", None)
    available_mib = getattr(hardware, "available_memory_mib", None)
    total_mib = getattr(hardware, "total_memory_mib", None)
    available_bytes = None if available_mib is None else int(available_mib) * _MIB
    total_bytes = None if total_mib is None else int(total_mib) * _MIB
    if limit_mib is not None:
        configured = int(limit_mib) * _MIB
        budget = (
            configured
            if available_bytes is None
            else min(configured, int(available_bytes * _DEFAULT_DEVICE_FRACTION))
        )
        return "host", budget, available_bytes, total_bytes, None
    if available_bytes is None:
        return "host", None, None, total_bytes, None
    return (
        "host",
        int(available_bytes * _DEFAULT_DEVICE_FRACTION),
        available_bytes,
        total_bytes,
        None,
    )

def _host_memory_budget(
    resources: Any | None,
) -> tuple[int | None, int | None, int | None]:
    """Return the host budget even when the active backend is a GPU."""
    limit_mib = getattr(resources, "memory_limit_mib", None)
    hardware = getattr(resources, "hardware", None)
    available_mib = getattr(hardware, "available_memory_mib", None)
    total_mib = getattr(hardware, "total_memory_mib", None)
    available = None if available_mib is None else int(available_mib) * _MIB
    total = None if total_mib is None else int(total_mib) * _MIB
    if limit_mib is not None:
        configured = int(limit_mib) * _MIB
        budget = (
            configured
            if available is None
            else min(configured, int(available * _DEFAULT_DEVICE_FRACTION))
        )
        return budget, available, total
    if available is None:
        return None, None, total
    return int(available * _DEFAULT_DEVICE_FRACTION), available, total

def _cupy_memory_info() -> tuple[int, int] | None:
    try:
        import cupy as cp

        free, total = cp.cuda.runtime.memGetInfo()
        return int(free), int(total)
    except Exception:
        return None

def _real_dtype(backend: Any) -> tuple[str, int]:
    value = str(getattr(getattr(backend, "config", None), "float_dtype", "float64"))
    if value in {"float16", "half"}:
        return "float16", 2
    if value in {"float32", "single"}:
        return "float32", 4
    return "float64", 8

def _backend_name(backend: Any) -> str:
    try:
        return str(backend.backend_name()).lower()
    except Exception:
        return type(backend).__name__.lower()

def _analyzer_workspace_bytes(
    trajectory_per_point: int,
    analysers: dict[str, Any],
    *,
    n_traj: int,
    total_n_traj: int,
    saved_samples: int,
    n_record_modes: int,
    real_itemsize: int,
    backend_name: str,
) -> _AnalyserWorkspace:
    """Estimate per-point analyser memory by location and lifetime.

    New analysers provide their own estimator. A capability-based fallback is
    retained for lightweight test doubles, but retained reducer state must be
    declared explicitly.
    """
    from qphase_sde.contracts.analyser import (
        AnalyserWorkspaceEstimate,
        AnalyserWorkspaceRequest,
    )

    request = AnalyserWorkspaceRequest(
        trajectory_bytes=trajectory_per_point,
        n_traj=n_traj,
        saved_samples=saved_samples,
        n_record_modes=n_record_modes,
        real_itemsize=real_itemsize,
        backend_name=backend_name,
        total_n_traj=total_n_traj,
    )
    device_total = 0
    host_total = 0
    retained_device_total = 0
    retained_host_total = 0
    for name, analyser in analysers.items():
        estimator_fn = getattr(analyser, "workspace", None)
        if callable(estimator_fn):
            estimate = estimator_fn(request)
            if not isinstance(estimate, AnalyserWorkspaceEstimate):
                raise TypeError(
                    f"analyser {name!r} returned an invalid workspace estimate"
                )
            device_total += max(0, int(estimate.device_bytes))
            host_total += max(0, int(estimate.host_bytes))
            retained_device_total += max(
                0, int(estimate.retained_device_bytes)
            )
            retained_host_total += max(0, int(estimate.retained_host_bytes))
            continue
        estimator = getattr(analyser, "estimator", None)
        capability_fn = getattr(estimator, "capabilities", None)
        capabilities = capability_fn() if callable(capability_fn) else None
        if (
            capabilities is not None
            and not getattr(capabilities, "backend_native", True)
            and backend_name == "cupy"
        ):
            host_total += 2 * trajectory_per_point
            continue
        chunk = getattr(
            getattr(estimator, "config", None), "fft_chunk_trajectories", None
        )
        if chunk is not None and 0 < int(chunk) < n_traj:
            # Chunked trajectory FFT: workspace scales with the chunk size.
            workspace = max(1, 2 * trajectory_per_point * int(chunk) // n_traj)
            if backend_name == "cupy":
                device_total += workspace
            else:
                host_total += workspace
            continue
        # One complex FFT, one real power array, and a conservative cuFFT margin.
        workspace = (
            trajectory_per_point * 2
            if estimator is not None
            else trajectory_per_point // 2
        )
        if backend_name == "cupy":
            device_total += workspace
        else:
            host_total += workspace
    return _AnalyserWorkspace(
        device_bytes=device_total,
        host_bytes=host_total,
        retained_device_bytes=retained_device_total,
        retained_host_bytes=retained_host_total,
    )

def _analyzer_capabilities(analyser: Any) -> Any:
    """Return a conservative capability object for old analyser instances."""
    from qphase_sde.contracts.analyser import AnalyserExecutionCapabilities

    provider = getattr(analyser, "execution_capabilities", None)
    if callable(provider):
        capabilities = provider()
        if isinstance(capabilities, AnalyserExecutionCapabilities):
            return capabilities
        raise TypeError("analyser execution_capabilities() returned an invalid value")
    return AnalyserExecutionCapabilities(
        supports_trajectory_batching=callable(
            getattr(analyser, "create_result_accumulator", None)
        )
    )

def _reserve_bytes(budget: int | None) -> int:
    if budget is None:
        return 0
    return max(_MIB, min(_MAX_RESERVE_BYTES, budget // 16))

def _tile_capacity(budget: int | None, fixed: int, per_point: int) -> int:
    """Return the number of scan points that fit one memory location."""
    if budget is None or per_point == 0:
        return (1 << 63) - 1
    return max(0, (budget - fixed) // per_point)

def _minimum_working_set_message(budget: int, required: int, backend_name: str) -> str:
    return (
        "A single scan point does not fit the available SDE working-set budget: "
        f"{required / _MIB:.1f} MiB estimated versus {budget / _MIB:.1f} MiB "
        f"for backend {backend_name!r}. Reduce n_traj/time samples, use a smaller "
        "dtype, or add trajectory-level reduction support."
    )

def _checked_product(*values: int, label: str) -> int:
    result = 1
    for value in values:
        if int(value) < 0:
            raise ValueError(f"{label} dimensions must be non-negative")
        result *= int(value)
        if result > (1 << 63) - 1:
            raise OverflowError(f"{label} size exceeds a signed 64-bit byte count")
    return result
