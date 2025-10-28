from __future__ import annotations

from typing import Dict, Optional, Callable, List, Any
import time as _time

from .protocols import (
    StateBase as StateLike,
    TrajectorySetBase as TrajectorySetLike,
    SDEModel,
    NoiseSpec,
)
from ..noise_models.factory import make_noise_model
from ..states.factory import make_state, make_trajectory_set
from ..core.registry import registry
from .errors import BackendError, IntegratorError

def run(model: SDEModel,
    ic,
    time: Dict,
    n_traj: int,
    solver: str,
    backend,
    noise_spec: NoiseSpec,
    seed: Optional[int] = None,
    master_seed: Optional[int] = None,
    per_traj_seeds: Optional[List[int]] = None,
    save_every: int = 1,
    *,
    # Progress reporting (lightweight, optional)
    progress_cb: Optional[Callable[[int, int, float, int, int], None]] = None,
    # callback signature: (step_done, steps_total, eta_seconds, ic_index, ic_total)
    progress_interval_seconds: float = 1.0,
    ic_index: int = 0,
    ic_total: int = 1,
    warmup_min_steps: int = 0,
    warmup_min_seconds: float = 0.0) -> TrajectorySetLike:
    """Run a multi-trajectory SDE simulation.

    Parameters
    ----------
    model : SDEModel
        Model providing drift/diffusion and metadata (n_modes, noise basis).
    ic : np.ndarray
        Initial condition(s): shape (n_modes,) or (n_traj, n_modes), complex128.
    time : Dict
        Time spec with keys: t0 (optional), dt (float), steps (int).
    n_traj : int
        Number of trajectories to simulate.
    solver : str
        Solver ID (euler|milstein). Milstein falls back to Euler in v0.1.1.
    backend : str
        Backend hint (currently 'numpy').
    noise_spec : NoiseSpec
        Real-noise channel specification (independent|correlated).
    seed : Optional[int]
        RNG seed for reproducibility.
    save_every : int
        Decimation factor for stored time points.

    Returns
    -------
    TrajectorySet
        Container with data of shape (n_traj, n_keep, n_modes) and time grid.
    """
    t0 = float(time.get('t0', 0.0))
    dt = float(time['dt'])
    steps = int(time['steps'])

    # Backend selection via registry or accept instance
    if isinstance(backend, str):
        key = backend if ":" in backend else f"backend:{backend}"
        try:
            be = registry.create(key)
        except Exception:
            raise BackendError(f"[200] Unsupported backend '{backend}'. Ensure it is registered.")
    else:
        be = backend
    # RNG strategy: if per_traj_seeds provided, spawn per-trajectory RNGs; else use single rng
    rng: Any
    if per_traj_seeds is not None and len(per_traj_seeds) == n_traj:
        # derive RNGs from provided integer seeds using backend rng()
        rng = [be.rng(int(s)) for s in per_traj_seeds]
    elif master_seed is not None:
        try:
            rng = be.spawn_rngs(int(master_seed), n_traj)
        except Exception:
            # fallback to single rng
            rng = be.rng(int(master_seed))
    else:
        rng = be.rng(seed)
    sampler = make_noise_model(noise_spec, be)

    # Initialize state; broadcast single-vector IC to all trajectories using backend
    y0 = be.asarray(ic)
    if getattr(y0, 'ndim', 1) == 1:
        n_modes = int(y0.shape[0])
        y_full = be.zeros((n_traj, n_modes), dtype=complex)
        for i in range(n_traj):
            y_full[i, :] = y0
        y0 = y_full
    state = make_state(be, y=y0, t=t0, attrs={"backend": getattr(be, 'name', 'backend'), "interpretation": "ito"})

    # storage
    n_keep = (steps // save_every) + 1
    out = be.empty((n_traj, n_keep, model.n_modes), dtype=complex)
    out[:, 0, :] = state.y
    keep_counter = 1

    t = t0
    # Progress tracking state
    last_report_step = 0
    last_report_time = None
    start_time = _time.monotonic()
    next_report_time = start_time + max(0.1, float(progress_interval_seconds))
    s_ema = None  # seconds per step (EMA)
    alpha = 0.2
    # compute thresholds
    warmup_steps_thr = max(0, int(warmup_min_steps))
    warmup_time_thr = max(0.0, float(warmup_min_seconds))
    for k in range(1, steps+1):
        # sample noise increment
        dW = sampler.sample(rng, state.n_traj, dt)  # (n_traj, M)

        # Select integrator via unified registry (milstein falls back to EM in v0.1.1)
        key = solver if ":" in solver else f"integrator:{solver}"
        try:
            integrator = registry.create(key)
        except Exception as e:
            raise IntegratorError(f"[300] Solver '{solver}' not registered")
        dy = integrator.step(state.y, t, dt, model, dW, be)

        state = make_state(be, y=state.y + dy, t=t + dt, attrs=state.attrs)
        t += dt

        if (k % save_every) == 0:
            out[:, keep_counter, :] = state.y
            keep_counter += 1

        # Lightweight periodic progress callback
        if progress_cb is not None:
            now = _time.monotonic()
            if now >= next_report_time:
                # Update rate using steps progressed since last report
                steps_delta = k - last_report_step
                if steps_delta > 0:
                    dt_wall = now - (last_report_time if last_report_time is not None else start_time)
                    s_inst = dt_wall / steps_delta
                    if s_ema is None:
                        s_ema = s_inst
                    else:
                        s_ema = alpha * s_inst + (1.0 - alpha) * s_ema
                last_report_step = k
                last_report_time = now
                next_report_time = now + max(0.1, float(progress_interval_seconds))

                # Respect warm-up thresholds before estimating ETA
                eta = float('nan')
                elapsed = now - start_time
                if s_ema is not None and k >= warmup_steps_thr and elapsed >= warmup_time_thr:
                    remaining = max(0, steps - k)
                    eta = remaining * float(s_ema)
                try:
                    progress_cb(k, steps, eta, ic_index, ic_total)
                except Exception:
                    # Never let progress reporting break the simulation
                    pass

    return make_trajectory_set(be, data=out, t0=t0, dt=dt*save_every, meta={})
