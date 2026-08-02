"""Resource-aware execution planning for logical SDE scans."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "SDEExecutionPlan",
    "SDEMemoryEstimate",
    "SDEMemoryPlanningError",
    "build_execution_plan",
]


_MIB = 1024**2
_DEFAULT_DEVICE_FRACTION = 0.75
_MAX_RESERVE_BYTES = 256 * _MIB


class SDEMemoryPlanningError(RuntimeError):
    """Raised when no valid SDE execution plan fits the supplied resources."""


@dataclass(frozen=True)
class SDEMemoryEstimate:
    """Conservative peak-memory components for one scan tile."""

    trajectory_bytes_per_point: int
    state_bytes_per_point: int
    noise_workspace_bytes_per_point: int
    analyzer_workspace_bytes: int
    reserve_bytes: int
    full_scan_trajectory_bytes: int
    estimated_peak_bytes: int

    def to_dict(self) -> dict[str, int]:
        """Return JSON-compatible estimate metadata."""
        return asdict(self)


@dataclass(frozen=True)
class SDEExecutionPlan:
    """An engine-owned plan compiled from one logical scan and resource object."""

    scan_size: int
    scan_shape: tuple[int, ...]
    n_traj_per_point: int
    trajectory_batch_size: int
    trajectory_batch_count: int
    logical_rng_group_size: int
    steps: int
    saved_samples: int
    scan_tile_size: int
    tile_count: int
    chunk_steps: int
    real_dtype: str
    budget_kind: str
    budget_bytes: int | None
    available_device_bytes: int | None
    device_total_bytes: int | None
    gpu_memory_fraction: float | None
    stream_analysis: bool
    rng_strategy: str
    memory: SDEMemoryEstimate

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible plan metadata."""
        payload = asdict(self)
        payload["scan_shape"] = list(self.scan_shape)
        return payload


def build_execution_plan(
    *,
    config: Any,
    grid: Any | None,
    model: Any,
    backend: Any,
    integrator: Any,
    analysers: dict[str, Any],
    resources: Any | None,
    device_memory: tuple[int, int] | None = None,
) -> SDEExecutionPlan:
    """Validate a job and choose an SDE-internal scan tile size."""
    steps = _validate_time_grid(config)
    _validate_analyser_sampling(config, analysers)

    scan_size = int(getattr(grid, "size", 1)) if grid is not None else 1
    scan_shape = tuple(getattr(grid, "shape", ())) if grid is not None else ()
    n_traj = int(config.n_traj)
    if n_traj < 1:
        raise ValueError("n_traj must be positive")

    record_modes = config.record_modes
    n_record_modes = (
        len(record_modes) if record_modes is not None else int(model.n_modes)
    )
    saved_samples = steps // int(config.save_stride) + 1
    real_dtype, real_itemsize = _real_dtype(backend)
    complex_itemsize = real_itemsize * 2
    chunk_steps = max(
        1, int(getattr(getattr(integrator, "config", None), "chunk_steps", 1))
    )

    trajectory_per_point = _checked_product(
        n_traj,
        saved_samples,
        n_record_modes,
        complex_itemsize,
        label="trajectory",
    )
    state_per_point = _checked_product(
        n_traj,
        int(model.n_modes),
        complex_itemsize,
        3,
        label="state workspace",
    )
    noise_per_point = _checked_product(
        n_traj,
        int(model.noise_dim),
        chunk_steps,
        real_itemsize,
        label="noise workspace",
    )
    analyzer_workspace = _analyzer_workspace_bytes(trajectory_per_point, analysers)

    backend_name = _backend_name(backend)
    budget_kind, budget, available, total, fraction = _memory_budget(
        backend_name,
        resources,
        device_memory=device_memory,
    )
    reserve = _reserve_bytes(budget)
    analysis_enabled = bool(analysers)
    should_keep = config.keep_traj is True or not analysis_enabled
    if backend_name == "cupy" and config.keep_traj is True:
        should_keep = False
    incremental = bool(analysers) and all(
        callable(getattr(analyser, "create_result_accumulator", None))
        for analyser in analysers.values()
    )
    batching_mode = str(getattr(config, "trajectory_batching", "auto"))
    requested_batch = getattr(config, "trajectory_batch_size", None)
    trajectory_batch = n_traj
    per_trajectory_work = (
        trajectory_per_point + state_per_point + noise_per_point
    ) // n_traj
    accumulator_workspace = _accumulator_workspace_bytes(
        saved_samples=saved_samples,
        n_record_modes=n_record_modes,
        real_itemsize=real_itemsize,
        analysers=analysers,
    )
    if requested_batch is not None:
        trajectory_batch = min(n_traj, int(requested_batch))
    elif budget is not None and incremental and not should_keep:
        available_for_trajectories = max(
            0, budget - reserve - accumulator_workspace
        )
        trajectory_batch = min(
            n_traj,
            max(1, available_for_trajectories // max(1, per_trajectory_work)),
        )
    if batching_mode == "off":
        trajectory_batch = n_traj
    if batching_mode == "required" and trajectory_batch == n_traj and n_traj > 1:
        trajectory_batch = max(1, n_traj // 2)
    if trajectory_batch < n_traj and (not incremental or should_keep):
        raise SDEMemoryPlanningError(
            "trajectory batching requires keep_traj=false and every configured "
            "analyser to provide an incremental result accumulator"
        )

    logical_rng_group = min(64, n_traj)
    if trajectory_batch < n_traj and trajectory_batch >= logical_rng_group:
        trajectory_batch = max(
            logical_rng_group,
            trajectory_batch // logical_rng_group * logical_rng_group,
        )
    elif trajectory_batch < logical_rng_group:
        logical_rng_group = trajectory_batch
    trajectory_batch_count = math.ceil(n_traj / trajectory_batch)
    stream_analysis = analysis_enabled and not should_keep and (
        scan_size > 1 or trajectory_batch_count > 1
    )

    working_per_point = per_trajectory_work * trajectory_batch
    retained_analysis_workspace = (
        accumulator_workspace
        if trajectory_batch_count > 1
        else analyzer_workspace
    )
    fixed = reserve + retained_analysis_workspace
    if trajectory_batch_count > 1:
        if budget is not None and fixed + working_per_point > budget:
            raise SDEMemoryPlanningError(
                _minimum_working_set_message(
                    budget, fixed + working_per_point, backend_name
                )
            )
        tile_size = 1
    elif budget is None:
        tile_size = scan_size
    elif stream_analysis:
        if fixed + working_per_point > budget:
            raise SDEMemoryPlanningError(
                _minimum_working_set_message(
                    budget, fixed + working_per_point, backend_name
                )
            )
        tile_size = max(1, min(scan_size, (budget - fixed) // working_per_point))
    else:
        required = fixed + working_per_point * scan_size
        if required > budget:
            raise SDEMemoryPlanningError(
                "The SDE result requires materializing the complete scan trajectory "
                f"({required / _MIB:.1f} MiB estimated) but the {budget_kind} "
                f"budget is {budget / _MIB:.1f} MiB. Configure an analyser with "
                "keep_traj=false so qphase_sde can process scan tiles, reduce the "
                "trajectory size, or increase the resource limit."
            )
        tile_size = scan_size

    tile_size = int(max(1, tile_size))
    tile_count = math.ceil(scan_size / tile_size)
    peak = fixed + working_per_point * tile_size
    memory = SDEMemoryEstimate(
        trajectory_bytes_per_point=trajectory_per_point,
        state_bytes_per_point=state_per_point,
        noise_workspace_bytes_per_point=noise_per_point,
        analyzer_workspace_bytes=retained_analysis_workspace,
        reserve_bytes=reserve,
        full_scan_trajectory_bytes=trajectory_per_point * scan_size,
        estimated_peak_bytes=peak,
    )
    return SDEExecutionPlan(
        scan_size=scan_size,
        scan_shape=scan_shape,
        n_traj_per_point=n_traj,
        trajectory_batch_size=trajectory_batch,
        trajectory_batch_count=trajectory_batch_count,
        logical_rng_group_size=logical_rng_group,
        steps=steps,
        saved_samples=saved_samples,
        scan_tile_size=tile_size,
        tile_count=tile_count,
        chunk_steps=chunk_steps,
        real_dtype=real_dtype,
        budget_kind=budget_kind,
        budget_bytes=budget,
        available_device_bytes=available,
        device_total_bytes=total,
        gpu_memory_fraction=fraction,
        stream_analysis=stream_analysis,
        rng_strategy=(
            "logical_trajectory_group_seedsequence_v1"
            if trajectory_batch_count > 1
            else "scan_point_seedsequence_v1"
            if grid is not None and stream_analysis and tile_count > 1
            else "legacy_batched_v1"
        ),
        memory=memory,
    )


def _validate_time_grid(config: Any) -> int:
    dt = float(config.dt)
    duration = float(config.t1) - float(config.t0)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if duration <= 0.0:
        raise ValueError("t1 must be greater than t0")
    steps = int(duration / dt)
    if steps < 1:
        raise ValueError("the configured time interval contains no integration steps")
    return steps


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
    available_bytes = (
        None if available_mib is None else int(available_mib) * _MIB
    )
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
    trajectory_per_point: int, analysers: dict[str, Any]
) -> int:
    if any(
        str(name).lower() == "psd"
        or str(getattr(analyser, "name", "")).lower() == "psd"
        for name, analyser in analysers.items()
    ):
        # One complex FFT, one real power array, and a conservative cuFFT margin.
        return trajectory_per_point * 2
    return trajectory_per_point // 2 if analysers else 0


def _accumulator_workspace_bytes(
    *,
    saved_samples: int,
    n_record_modes: int,
    real_itemsize: int,
    analysers: dict[str, Any],
) -> int:
    """Estimate fixed host/device arrays retained by incremental analysers."""
    if not analysers:
        return 0
    if any(
        str(name).lower() == "psd"
        or str(getattr(analyser, "name", "")).lower() == "psd"
        for name, analyser in analysers.items()
    ):
        # Axis, mean, M2, standard deviation, and temporary finalization margin.
        return _checked_product(
            saved_samples,
            n_record_modes,
            real_itemsize,
            5,
            label="PSD accumulator workspace",
        )
    return _checked_product(
        saved_samples,
        n_record_modes,
        real_itemsize,
        2,
        label="analyser accumulator workspace",
    )


def _reserve_bytes(budget: int | None) -> int:
    if budget is None:
        return 0
    return max(_MIB, min(_MAX_RESERVE_BYTES, budget // 16))


def _minimum_working_set_message(
    budget: int, required: int, backend_name: str
) -> str:
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
