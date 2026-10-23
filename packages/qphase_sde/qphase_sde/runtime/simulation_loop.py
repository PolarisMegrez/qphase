"""qphase_sde: The SDE integration loop.

One integration pass: build the RNG streams, step the integrator, poll
observers and hand back a retained trajectory set. Analysers never run here -
the typed product layer above decides what gets published.
"""

from __future__ import annotations

import time as _time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy, get_xp

from qphase_sde.integrator.base import Integrator
from qphase_sde.model import NoiseSpec, SDEModel
from qphase_sde.observer.base import (
    ObserverContext,
    ObserverDecision,
    ObserverTriggeredError,
)
from qphase_sde.planning import resolve_time_grid
from qphase_sde.runtime.buffers import SDEBufferCache
from qphase_sde.runtime.rng import (
    _AdaptiveNoiseRNG,
    _GroupedRNG,
    draw_standard_normal_into,
)
from qphase_sde.state import TrajectorySet


class TrajectoryDivergenceError(RuntimeError):
    """Raised when an SDE trajectory leaves a configured state-norm bound."""


def run_sde(
    engine: Any,
    model: SDEModel,
    ic,
    time: dict,
    n_traj: int,
    solver: Integrator | None = None,
    backend: BackendBase | None = None,
    noise_spec: NoiseSpec | None = None,
    seed: int | None = None,
    master_seed: int | None = None,
    per_traj_seeds: list[int] | None = None,
    return_stride: int = 1,
    rng_stream: str = "per_trajectory",
    *,
    _progress_hook: Callable[[int, int, float, int, int], None] | None = None,
    progress_interval_seconds: float = 1.0,
    ic_index: int = 0,
    ic_total: int = 1,
    warmup_min_steps: int = 0,
    warmup_min_seconds: float = 0.0,
    rng: Any | None = None,
    rng_group_seeds: tuple[int, ...] | None = None,
    rng_group_size: int | None = None,
    rng_group_counts: tuple[int, ...] | None = None,
    observers: dict[str, Any] | None = None,
) -> TrajectorySet:
    """Run a multi-trajectory SDE simulation.

    This method implements the full simulation logic, including:
    - Backend and integrator resolution
    - State class resolution
    - RNG setup
    - Simulation loop with progress reporting
    - Result collection and packaging

    Note
    ----
    Backend pre-checks and RNG setup can be done by the scheduler
    before calling this method to provide better error messages.

    Parameters
    ----------
    engine : Engine
        Facade supplying the resolved configuration and plugin instances.
    model : SDEModel
        SDE model providing drift/diffusion and metadata
    ic : array-like
        Initial conditions
    time : dict
        Time spec with keys: t0 (optional integration start), dt, steps,
        and optional observation_start. Samples before observation_start
        are integrated but not retained.
    n_traj : int
        Number of trajectories
    solver : Integrator, optional
        Solver instance; overrides Engine default if provided
    backend : BackendBase, optional
        Backend instance; overrides Engine default if provided
    noise_spec : NoiseSpec, optional
        Noise specification
    seed : int, optional
        RNG seed
    master_seed : int, optional
        Master seed for per-trajectory streams
    per_traj_seeds : list[int], optional
        Explicit per-trajectory seeds
    return_stride : int
        Decimation factor for returned TrajectorySet
    rng_stream : str
        RNG strategy: 'per_trajectory' or 'batched'
    progress_interval_seconds : float
        Minimum time between progress reports
    ic_index : int
        Current IC index (for progress reporting)
    ic_total : int
        Total IC count (for progress reporting)
    warmup_min_steps : int
        Minimum steps before ETA estimation
    warmup_min_seconds : float
        Minimum time before ETA estimation
    rng : any, optional
        Pre-configured RNG handle(s) (from scheduler)
    rng_group_seeds : tuple[int, ...], optional
        Stable per-group seeds covering the trajectory ensemble.
    rng_group_size : int, optional
        Number of adjacent trajectories driven by each grouped seed.
    rng_group_counts : tuple[int, ...], optional
        Explicit per-group trajectory counts when the last group is
        short; ``None`` derives uniform groups from ``rng_group_size``.
    observers : dict[str, Any], optional
        Online observers (SDEObserverProtocol) checked against the live
        state at their configured cadence. Observers never mutate the
        state or consume RNG; ``stop_batch`` truncates the output at the
        confirming check and ``fail_job`` raises ObserverTriggeredError.

    Returns
    -------
    TrajectorySet
        The simulation result.

    """
    # Resolve dependencies
    be = backend or engine._default_backend
    if be is None:
        raise RuntimeError("No backend provided or configured.")

    integrator = solver or engine._default_integrator
    if integrator is None:
        raise RuntimeError("No integrator provided or configured.")

    # Parse time config
    t0 = float(time.get("t0", 0.0))
    observation_t0 = float(time.get("observation_start", t0))
    dt = float(time["dt"])
    steps = int(time["steps"])
    if observation_t0 < t0:
        raise ValueError("observation_start must not precede integration t0")
    observation_offset = (observation_t0 - t0) / dt
    warmup_steps = round(observation_offset)
    if not np.isclose(observation_offset, warmup_steps, rtol=1e-10, atol=1e-9):
        raise ValueError("observation_start must align with the integration dt")
    if warmup_steps < 0 or warmup_steps >= steps:
        raise ValueError("observation_start must lie before the integration end")

    # Initialize state
    # Ensure IC is on the correct backend
    if hasattr(ic, "to_backend"):
        ic_be = ic.to_backend(be)
        y0 = ic_be.data
    else:
        # Handle string ICs (e.g. from YAML)
        def _parse_complex(val):
            if isinstance(val, str):
                try:
                    return complex(val.replace(" ", ""))
                except ValueError:
                    return val
            return val

        if isinstance(ic, list):
            # Handle 1D or 2D lists
            if ic and isinstance(ic[0], list):
                ic = [[_parse_complex(x) for x in row] for row in ic]
            else:
                ic = [_parse_complex(x) for x in ic]

        # Determine target dtype from backend config
        target_dtype = None
        if hasattr(be, "config") and hasattr(be.config, "float_dtype"):

            def _is_complex_recursive(obj):
                if isinstance(obj, complex):
                    return True
                dtype = getattr(obj, "dtype", None)
                if getattr(dtype, "kind", None) == "c":
                    return True
                if isinstance(obj, list):
                    return any(_is_complex_recursive(x) for x in obj)
                return False

            if _is_complex_recursive(ic):
                float_dtype = be.config.float_dtype
                target_dtype = (
                    "complex64" if float_dtype == "float32" else "complex128"
                )
            else:
                target_dtype = be.config.float_dtype

        y0 = be.asarray(ic, dtype=target_dtype)

    # Broadcast IC if necessary to match n_traj
    # Expected shape: (n_traj, n_modes)
    if y0.ndim == 1:
        y0 = be.expand_dims(y0, 0)

    if y0.shape[0] != n_traj:
        if y0.shape[0] == 1:
            # Broadcast
            y0 = be.repeat(y0, n_traj, axis=0)

    # Initialize state variables (unwrap State object for loop performance)
    # We keep 'y' (data) and 't' (time) as separate variables to avoid
    # creating State objects in the inner loop.
    y = y0
    t = float(t0)
    state_norm_limit = (
        None
        if engine.config is None or engine.config.max_state_norm is None
        else float(engine.config.max_state_norm)
    )
    state_check_interval = (
        1 if engine.config is None else int(engine.config.state_check_interval_steps)
    )
    if state_norm_limit is not None:
        _check_state_norm(y, t, state_norm_limit)
    next_state_check_step = state_check_interval

    # Setup RNG if not pre-configured
    if rng is None:
        try:
            if rng_group_seeds is not None:
                group_size = int(rng_group_size or 0)
                if group_size < 1 or not rng_group_seeds:
                    raise ValueError(
                        "rng_group_seeds and rng_group_size do not cover n_traj"
                    )
                if rng_group_counts is None:
                    n_groups = len(rng_group_seeds)
                    if (
                        n_groups * group_size < n_traj
                        or (n_groups - 1) * group_size >= n_traj
                    ):
                        raise ValueError(
                            "rng_group_seeds and rng_group_size do not cover "
                            "n_traj"
                        )
                    counts = (group_size,) * (n_groups - 1) + (
                        n_traj - (n_groups - 1) * group_size,
                    )
                else:
                    counts = tuple(int(value) for value in rng_group_counts)
                    if (
                        len(counts) != len(rng_group_seeds)
                        or any(value < 1 for value in counts)
                        or sum(counts) != n_traj
                    ):
                        raise ValueError(
                            "rng_group_counts must match rng_group_seeds and "
                            "cover n_traj"
                        )
                handles = tuple(be.rng(int(value)) for value in rng_group_seeds)
                rng = (
                    handles[0]
                    if len(handles) == 1
                    else _GroupedRNG(handles, group_size, counts)
                )
            elif per_traj_seeds is not None and len(per_traj_seeds) == n_traj:
                rng = [be.rng(int(s)) for s in per_traj_seeds]
            elif master_seed is not None:
                if str(rng_stream) == "per_trajectory":
                    try:
                        rng = be.spawn_rngs(int(master_seed), n_traj)
                    except Exception:
                        rng = be.rng(int(master_seed))
                else:
                    rng = be.rng(int(master_seed))
            else:
                rng = be.rng(seed)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RNG: {e}") from e

    # Prepare output
    rs = max(1, int(return_stride))
    if engine.config is not None and engine.config.record_modes is not None:
        record_modes = tuple(int(mode) for mode in engine.config.record_modes)
        if len(set(record_modes)) != len(record_modes):
            raise ValueError("record_modes must not contain duplicates")
        if any(mode < 0 or mode >= model.n_modes for mode in record_modes):
            raise ValueError(f"record_modes must be within 0..{model.n_modes - 1}")
    else:
        record_modes = tuple(range(model.n_modes))
    observation_steps = steps - warmup_steps
    n_keep = (observation_steps // rs) + 1
    out = be.empty((n_traj, n_keep, len(record_modes)), dtype=y.dtype)
    keep_counter = 0
    if warmup_steps == 0:
        out[:, 0, :] = y[:, record_modes]
        keep_counter = 1

    # Online observers: initialize against the run layout and check the
    # initial condition. The engine schedules observe rounds at multiples
    # of each observer's cadence; fused chunks are clamped so they always
    # end exactly on a due step (observers never sample inside a chunk).
    observer_items: list[tuple[str, Any]] = (
        list(observers.items()) if observers else []
    )
    observer_intervals = [
        max(1, int(observer.check_interval_steps)) for _, observer in observer_items
    ]
    next_observe_step: int | None = None
    stop_observer: str | None = None
    if observer_items:
        observer_context = ObserverContext(
            n_traj=n_traj,
            n_modes=model.n_modes,
            record_modes=record_modes,
            dt=dt,
            integration_t0=t0,
            observation_t0=observation_t0,
            total_steps=steps,
            backend=be,
        )
        for _, observer in observer_items:
            observer.initialize(observer_context)
        next_observe_step = min(observer_intervals)
        stop_observer = _handle_observer_actions(observer_items, y, t, 0)

    # Progress tracking
    last_report_step = 0
    last_report_time = None
    start_time = _time.monotonic()
    next_report_time = start_time + max(0.1, float(progress_interval_seconds))
    s_ema = None
    alpha = 0.2
    warmup_time_thr = max(0.0, float(warmup_min_seconds))

    # Main simulation loop
    t_end = t0 + steps * dt
    save_dt = dt * rs
    next_save_step = warmup_steps + (rs if keep_counter else 0)
    next_save_time = t0 + next_save_step * dt

    # Adaptive stepping setup
    use_adaptive = False
    noise_spec = None
    adaptive_rng = None
    # Ensure config is not None (use default if None)
    config = engine.config
    if config is None:  # pragma: no cover - the engine always configures
        from qphase_sde.engine import EngineConfig

        config = EngineConfig()
    if (
        config.adaptive
        and hasattr(integrator, "supports_adaptive_step")
        and integrator.supports_adaptive_step()
    ):  # noqa: E501
        use_adaptive = True
        # Use config tolerance if available, else fallback to integrator default
        tol = (
            config.atol
            if config.atol is not None
            else getattr(integrator, "tol", 1e-3)
        )

        # Update integrator bounds from config
        if hasattr(integrator, "min_dt"):
            integrator.min_dt = config.min_dt
        if hasattr(integrator, "max_dt"):
            integrator.max_dt = config.max_dt

        noise_spec = NoiseSpec(kind="independent", dim=model.noise_dim)
        # Adaptive integrators draw noise internally; hand grouped
        # streams through a Generator-style facade so per-group
        # provenance survives without leaking the private container.
        adaptive_rng = (
            _AdaptiveNoiseRNG(be, rng) if isinstance(rng, _GroupedRNG) else rng
        )

    current_dt = dt
    k = 0

    integrator_config = getattr(integrator, "config", None)
    requested_chunk_steps = int(getattr(integrator_config, "chunk_steps", 1))
    supports_chunk = getattr(integrator, "supports_chunk_step", None)
    chunk_step = getattr(integrator, "step_chunk", None)
    use_chunked = (
        not use_adaptive
        and requested_chunk_steps > 1
        and callable(supports_chunk)
        and callable(chunk_step)
        and bool(supports_chunk(model, be))
    )

    # Buffer cache for temporary arrays used inside the integration loop.
    # This reduces per-step allocation overhead, especially on GPU backends.
    buf_cache = SDEBufferCache(be, max_entries_per_key=2)

    while t < t_end - 1e-12 and stop_observer is None:
        if use_chunked:
            assert rng is not None, "RNG not initialized"
            assert callable(chunk_step)
            n_chunk = min(requested_chunk_steps, steps - k)
            if next_observe_step is not None:
                # Clamp fused chunks so they end exactly on the next due
                # observer check; cadence stays exact without sampling
                # inside a chunk.
                n_chunk = min(n_chunk, max(1, next_observe_step - k))
            noise_dtype = y.real.dtype if hasattr(y, "real") else y.dtype
            noise_shape = (n_chunk, n_traj, model.noise_dim)
            raw_noise = buf_cache.get(noise_shape, noise_dtype)
            try:
                draw_standard_normal_into(be, rng, raw_noise)
                dt_sqrt = be.asarray(dt**0.5, dtype=raw_noise.dtype)
                raw_noise *= dt_sqrt
                save_offsets_list: list[int] = []
                target_step = next_save_step
                while target_step <= k + n_chunk:
                    if target_step > k:
                        save_offsets_list.append(target_step - k)
                    target_step += rs
                save_offsets = tuple(save_offsets_list)
                result = chunk_step(
                    y,
                    t,
                    dt,
                    model,
                    raw_noise,
                    be,
                    n_steps=n_chunk,
                    save_offsets=save_offsets,
                    record_modes=record_modes,
                )
            finally:
                buf_cache.put(raw_noise)
            y = result.final_state
            next_k = k + n_chunk
            next_t = t0 + next_k * dt
            if state_norm_limit is not None and (
                next_k >= next_state_check_step or next_k >= steps
            ):
                _check_state_norm(y, next_t, state_norm_limit)
                next_state_check_step = (
                    next_k // state_check_interval + 1
                ) * state_check_interval
            n_saved = len(save_offsets)
            if n_saved:
                end = keep_counter + n_saved
                out[:, keep_counter:end, :] = result.saved_states
                keep_counter = end
                next_save_step += n_saved * rs
                next_save_time = t0 + next_save_step * dt
            k += n_chunk
            t = t0 + k * dt
        else:
            k += 1
            y_prev = y
            t_prev = t

            if use_adaptive:
                assert noise_spec is not None
                y_next, t_next, next_dt, error = integrator.step_adaptive(
                    y, t, current_dt, tol, model, noise_spec, be, adaptive_rng
                )
                y = y_next
                t = float(t_next)
                current_dt = float(next_dt)
            else:
                assert rng is not None, "RNG not initialized"
                noise_dtype = float
                if hasattr(y, "real") and hasattr(y.real, "dtype"):
                    noise_dtype = y.real.dtype
                elif hasattr(y, "dtype"):
                    noise_dtype = y.dtype

                step_noise_shape = (n_traj, model.noise_dim)
                raw_noise = buf_cache.get(step_noise_shape, noise_dtype)
                try:
                    draw_standard_normal_into(be, rng, raw_noise)
                    dt_sqrt = current_dt**0.5
                    if hasattr(raw_noise, "dtype"):
                        dt_sqrt = be.asarray(dt_sqrt, dtype=raw_noise.dtype)

                    raw_noise *= dt_sqrt
                    dy = integrator.step(y, t, current_dt, model, raw_noise, be)
                finally:
                    buf_cache.put(raw_noise)
                y = y + dy
                t += current_dt

            if state_norm_limit is not None and (
                k >= next_state_check_step or k >= steps
            ):
                _check_state_norm(y, t, state_norm_limit)
                next_state_check_step = (
                    k // state_check_interval + 1
                ) * state_check_interval

            while t >= next_save_time - 1e-12 and keep_counter < n_keep:
                y_interp = buf_cache.get((n_traj, model.n_modes), y.dtype)
                try:
                    if t > t_prev + 1e-12:
                        frac = (next_save_time - t_prev) / (t - t_prev)
                        frac = max(0.0, min(1.0, frac))
                        y_interp[...] = y_prev + (y - y_prev) * frac
                    else:
                        y_interp[...] = y

                    out[:, keep_counter, :] = y_interp[:, record_modes]
                    keep_counter += 1
                    next_save_step += rs
                    next_save_time += save_dt
                finally:
                    buf_cache.put(y_interp)

        # Online observers run on their due steps (including chunk
        # boundaries); actions are fail_job > stop_batch > record.
        if (
            observer_items
            and next_observe_step is not None
            and k >= next_observe_step
        ):
            stop_observer = _handle_observer_actions(observer_items, y, t, k)
            next_observe_step = min(
                (k // interval + 1) * interval for interval in observer_intervals
            )

        # Progress reporting
        if _progress_hook is not None:
            now = _time.monotonic()
            if now >= next_report_time:
                progress = (t - t0) / (t_end - t0)
                progress = max(0.0, min(1.0, progress))

                steps_delta = k - last_report_step
                if steps_delta > 0:
                    dt_wall = now - (
                        last_report_time if last_report_time else start_time
                    )
                    s_inst = dt_wall / steps_delta
                    s_ema = (
                        s_inst
                        if s_ema is None
                        else alpha * s_inst + (1.0 - alpha) * s_ema
                    )

                last_report_step = k
                last_report_time = now
                next_report_time = now + max(0.1, float(progress_interval_seconds))

                elapsed = now - start_time
                eta = float("nan")
                # Ensure progress is float
                progress = float(progress)
                if progress > 0 and elapsed >= warmup_time_thr:
                    eta = elapsed / progress * (1 - progress)

                try:
                    _progress_hook(
                        int(progress * steps), steps, eta, ic_index, ic_total
                    )
                except Exception:
                    pass

    if stop_observer is None and keep_counter != n_keep:
        raise RuntimeError(
            f"SDE observation saved {keep_counter} samples, expected {n_keep}"
        )
    if stop_observer is not None and keep_counter < n_keep:
        out = out[:, :keep_counter]

    meta: dict[str, Any] = {
        "mode_indices": list(record_modes),
        "integration_t0": t0,
        "observation_t0": observation_t0,
        "integration_t1": t_end,
        "warmup_steps": warmup_steps,
    }
    if stop_observer is not None:
        meta.update(
            {
                "stopped_early": True,
                "stop_reason": f"observer:{stop_observer}",
                "effective_steps": k,
            }
        )
    if observer_items:
        observer_payloads: dict[str, Any] = {}
        for name, observer in observer_items:
            observer.note_end_of_run(float(t), k)
            observer_payloads[name] = observer.finalize()
        meta["observers"] = observer_payloads

    return TrajectorySet(
        data=out,
        t0=observation_t0,
        dt=dt * rs,
        meta=meta,
    )

def _check_state_norm(y: Any, t: float, limit: float) -> None:
    """Fail before downstream analysis when a trajectory has escaped."""
    xp = get_xp(y)
    squared_norm = xp.sum(xp.abs(y) ** 2, axis=-1)
    max_norm = float(
        np.asarray(convert_to_numpy(xp.sqrt(xp.max(squared_norm)))).item()
    )
    if not np.isfinite(max_norm) or max_norm > limit:
        raise TrajectoryDivergenceError(
            "SDE trajectory escaped the configured state bound at "
            f"t={t:.9g}: max ||y||={max_norm:.9g}, limit={limit:.9g}. "
            "No downstream analysis was produced because the trajectory "
            "ensemble is not stationary within the requested observation window."
        )


def _handle_observer_actions(
    observer_items: list[tuple[str, Any]],
    y: Any,
    t: float,
    step: int,
) -> str | None:
    """Run one observer round and apply whole-batch control-flow actions.

    Every observer sees the state even when another one requests an
    action. ``fail_job`` raises ObserverTriggeredError; the first
    ``stop_batch`` requester is returned so the caller can stop the loop.
    """
    decisions = [
        (name, observer.observe(y, t, step)) for name, observer in observer_items
    ]
    failures = [
        (name, decision)
        for name, decision in decisions
        if decision is not None and decision.action == "fail_job"
    ]
    if failures:
        raise _observer_fail_job_error(failures, t, step)
    stops = [
        name
        for name, decision in decisions
        if decision is not None and decision.action == "stop_batch"
    ]
    return stops[0] if stops else None


def _observer_fail_job_error(
    failures: list[tuple[str, ObserverDecision]],
    t: float,
    step: int,
) -> ObserverTriggeredError:
    """Build a generic error from plugin-owned failure details."""
    details: dict[str, Any] = {}
    parts: list[str] = []
    for name, decision in failures:
        details[name] = dict(decision.details)
        parts.append(f"observer {name!r}: {decision.message}")
    message = (
        f"SDE observer fail_job action triggered at t={float(t):.9g} "
        f"(step {int(step)}): " + "; ".join(parts)
    )
    return ObserverTriggeredError(
        message, {"time": float(t), "step": int(step), "observers": details}
    )


@dataclass(frozen=True)
class SimulatedRun:
    """In-memory outcome of one integration pass, before typed products.

    Analysers never run inside the integration pass: the trajectory feeds
    the typed product graph as the ``trajectories`` product, and observer
    payloads ride along separately so they survive trajectory retention
    choices. This is engine-internal — it is never persisted or returned
    from the plugin boundary.
    """

    trajectory: TrajectorySet
    meta: dict[str, Any]
    observer_payloads: dict[str, Any]


def run_simulate(
    engine: Any,
    data: Any | None,
    *,
    reporter: Any | None = None,
    context: Any | None = None,
    scan_tile: Any | None = None,
    rng_group_seeds: tuple[int, ...] | None = None,
    rng_group_size: int | None = None,
    rng_group_counts: tuple[int, ...] | None = None,
    observers: dict[str, Any] | None = None,
    progress_offset: int = 0,
    progress_scale: int | None = None,
    progress_total: int | None = None,
    progress_label: str | None = None,
) -> SimulatedRun:
    """Execute one SDE integration pass; analysers never run here.

    The trajectory always comes back retained: whether it is published
    as a product is decided by the typed graph/merge layer above, never
    inside the integration pass.
    """
    assert engine.config is not None

    if observers is None:
        observers = engine._normalised_observers()

    model = engine._required_model()

    time_grid = resolve_time_grid(
        t0=engine.config.t0,
        t1=engine.config.t1,
        dt=engine.config.dt,
    )
    time_cfg = {
        "t0": time_grid.integration_t0,
        "observation_start": time_grid.observation_t0,
        "dt": engine.config.dt,
        "steps": time_grid.total_steps,
    }
    if progress_scale is None:
        progress_scale = int(engine.config.n_traj)
    effective_progress_total = (
        progress_total
        if progress_total is not None
        else progress_offset + progress_scale * time_cfg["steps"]
    )

    ic = engine.config.ic
    if ic is None:
        if hasattr(model, "default_ic"):
            ic = model.default_ic
        elif data is not None:
            ic = data
        else:
            raise RuntimeError("No IC provided.")

    # Adapt the integration loop's counters to structured, monotonic work.
    progress_hook = None
    if reporter is not None:

        def _sde_cb(
            k: int, steps: int, eta: float, ic_index: int, ic_total: int
        ) -> None:
            del eta
            prefix = f"{progress_label} | " if progress_label else ""
            msg = f"{prefix}Traj {ic_index + 1}/{ic_total} | Step {k}/{steps}"
            stage = "warmup" if k < time_grid.warmup_steps else "sampling"
            metadata: dict[str, Any] = {
                "ic_index": ic_index,
                "ic_total": ic_total,
            }
            if context is not None:
                context_metadata = getattr(context, "metadata", {})
                scan_summary = context_metadata.get("scan_summary")
                if scan_summary is not None:
                    metadata["scan_summary"] = scan_summary
            reporter.update(
                completed=progress_offset + progress_scale * k,
                total=effective_progress_total,
                unit="trajectory-step",
                stage=stage,
                message=msg,
                metadata=metadata,
            )

        progress_hook = _sde_cb

    traj_set: TrajectorySet = run_sde(engine,
        model=model,
        ic=ic,
        time=time_cfg,
        n_traj=engine.config.n_traj,
        seed=engine.config.seed,
        return_stride=engine.config.save_stride,
        _progress_hook=progress_hook,
        rng_group_seeds=rng_group_seeds,
        rng_group_size=rng_group_size,
        rng_group_counts=rng_group_counts,
        observers=observers,
    )
    # Lift observer payloads out of the trajectory metadata immediately so
    # they ride along independently of trajectory retention choices in the
    # typed product layer. They intentionally do not participate in the
    # keep/drop decision: a record-only observer must not change retention.
    observer_results: dict[str, Any] = dict(traj_set.meta.pop("observers", {}))
    n_scan = 1 if scan_tile is None else int(scan_tile.count)
    if n_scan < 1:
        n_scan = 1
    if n_scan > 1 and observer_results:
        if int(engine.config.n_traj) % n_scan:
            raise ValueError(
                f"Observer ensemble has {engine.config.n_traj} trajectories, "
                f"not divisible by scan count {n_scan}"
            )
        group_size = int(engine.config.n_traj) // n_scan
        observer_instances = observers or engine._normalised_observers()
        observer_results = {
            name: observer_instances[name].split_payload(
                payload, [group_size] * n_scan
            )
            for name, payload in observer_results.items()
        }
    if reporter is not None:
        reporter.update(
            completed=(progress_offset + progress_scale * time_cfg["steps"]),
            total=effective_progress_total,
            unit="trajectory-step",
            stage="sampling",
            message=(
                f"{progress_label} complete"
                if progress_label
                else "Sampling complete"
            ),
        )

    # Ensure model parameters are in metadata. The model's params mapping is
    # live: scan adapters mutate it in place and restore it on exit, so
    # snapshot it here instead of keeping a reference that would revert.
    meta: dict[str, Any] = {}
    if hasattr(model, "config") and hasattr(model.config, "model_dump"):
        meta["params"] = model.config.model_dump()
    elif hasattr(model, "parameters"):
        meta["params"] = model.parameters
    elif hasattr(model, "params"):
        meta["params"] = dict(model.params)

    return SimulatedRun(
        trajectory=traj_set,
        meta=meta,
        observer_payloads=observer_results,
    )
