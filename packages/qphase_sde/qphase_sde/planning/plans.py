"""Plan and time-grid records of the SDE execution planner."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


class SDEMemoryPlanningError(RuntimeError):
    """Raised when no valid SDE execution plan fits the supplied resources."""

@dataclass(frozen=True)
class SDETimeGrid:
    """Resolved integration and observation intervals for one SDE run."""

    integration_t0: float
    observation_t0: float
    integration_t1: float
    dt: float
    total_steps: int
    warmup_steps: int
    observation_steps: int

    def saved_samples(self, save_stride: int) -> int:
        """Return samples retained from the observation window, including t0."""
        return self.observation_steps // int(save_stride) + 1

@dataclass(frozen=True)
class SDEMemoryEstimate:
    """Conservative peak-memory components for one scan tile."""

    trajectory_bytes_per_point: int
    state_bytes_per_point: int
    noise_workspace_bytes_per_point: int
    analyzer_workspace_bytes: int
    analyzer_device_workspace_bytes: int
    analyzer_host_workspace_bytes: int
    reserve_bytes: int
    host_reserve_bytes: int
    full_scan_trajectory_bytes: int
    estimated_peak_bytes: int
    estimated_device_peak_bytes: int
    estimated_host_peak_bytes: int

    def to_dict(self) -> dict[str, int]:
        """Return JSON-compatible estimate metadata."""
        return asdict(self)

@dataclass(frozen=True)
class _AnalyserWorkspace:
    """Per-scan-point analyser memory split by lifetime and location."""

    device_bytes: int = 0
    host_bytes: int = 0
    retained_device_bytes: int = 0
    retained_host_bytes: int = 0

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
    warmup_steps: int
    observation_steps: int
    saved_samples: int
    scan_tile_size: int
    tile_count: int
    chunk_steps: int
    real_dtype: str
    budget_kind: str
    budget_bytes: int | None
    available_device_bytes: int | None
    device_total_bytes: int | None
    host_budget_bytes: int | None
    available_host_bytes: int | None
    host_total_bytes: int | None
    gpu_memory_fraction: float | None
    stream_analysis: bool
    rng_strategy: str
    memory: SDEMemoryEstimate
    observer_check_interval_steps: int | None = None
    effective_chunk_steps: int | None = None
    deferred_analysers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible plan metadata.

        The observer-derived fields are omitted when not applicable so plans
        without observers keep their previous shape; ``deferred_analysers``
        is likewise omitted when no analyser is deferred to post-merge
        execution.
        """
        payload = asdict(self)
        payload["scan_shape"] = list(self.scan_shape)
        payload["deferred_analysers"] = list(self.deferred_analysers)
        if self.observer_check_interval_steps is None:
            payload.pop("observer_check_interval_steps")
        if self.effective_chunk_steps is None:
            payload.pop("effective_chunk_steps")
        if not self.deferred_analysers:
            payload.pop("deferred_analysers")
        return payload

def resolve_time_grid(*, t0: float, t1: float, dt: float) -> SDETimeGrid:
    """Resolve ``[0, t0)`` warm-up and ``[t0, t1]`` observation intervals."""
    dt = float(dt)
    observation_t0 = float(t0)
    integration_t1 = float(t1)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if observation_t0 < 0.0:
        raise ValueError("t0 must be non-negative")
    if integration_t1 <= observation_t0:
        raise ValueError("t1 must be greater than t0")

    total_steps = _aligned_step_count(integration_t1, dt, "t1")
    warmup_steps = _aligned_step_count(observation_t0, dt, "t0")
    observation_steps = total_steps - warmup_steps
    if observation_steps < 1:
        raise ValueError("the configured observation interval contains no steps")
    return SDETimeGrid(
        integration_t0=0.0,
        observation_t0=observation_t0,
        integration_t1=integration_t1,
        dt=dt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        observation_steps=observation_steps,
    )

def _aligned_step_count(value: float, dt: float, name: str) -> int:
    raw = value / dt
    rounded = round(raw)
    if not math.isclose(raw, rounded, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError(
            f"{name}={value:.12g} must be an integer multiple of dt={dt:.12g}"
        )
    return int(rounded)
