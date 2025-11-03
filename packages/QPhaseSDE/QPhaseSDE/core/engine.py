"""
QPhaseSDE: Simulation Engine
----------------------------
Backend-agnostic driver that advances multi-trajectory SDEs using pluggable
integrators, noise models, and compute backends. It exposes a single
high-level entry point `run(...)` and keeps policy minimal to preserve stable
contracts.

Behavior
--------
- (*RNG Strategy*) The engine supports per-trajectory random streams derived
  from explicit seeds or a master seed, as well as a batched mode. Backends
  provide RNG handles; reproducibility depends on a consistent strategy and
  fixed ordering.
- (*Progress Reporting*) An optional callback can be invoked periodically with
  progress and ETA. The ETA is estimated after a short warm‑up using an
  exponential moving average of time per step, and any exception inside the
  callback is contained and does not affect the run.
- (*Interoperability*) State and trajectory containers are selected by a
  backend-aware factory so that storage matches the chosen compute backend.
  Integrators and noise models are resolved from the central registry for
  decoupled composition.

Notes
-----
- The solver key 'milstein' currently aliases to Euler–Maruyama in this
  release.
- Passing a backend name uses the registry; passing an instance bypasses
  lookup.
"""

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
from .errors import QPSBackendError, QPSRegistryError, get_logger

__all__ = [
    "run",
]

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
    return_stride: int = 1,
    rng_stream: str = "per_trajectory",
    *,
    # Progress reporting (lightweight, optional)
    progress_cb: Optional[Callable[[int, int, float, int, int], None]] = None,
    # callback signature: (step_done, steps_total, eta_seconds, ic_index, ic_total)
    progress_interval_seconds: float = 1.0,
    ic_index: int = 0,
    ic_total: int = 1,
    warmup_min_steps: int = 0,
    warmup_min_seconds: float = 0.0) -> TrajectorySetLike:
    """Run a multi-trajectory SDE simulation and return sampled trajectories.

    Advance the system defined by ``model`` for ``n_traj`` trajectories over the
    time grid specified by ``time`` using the selected ``solver``, compute
    ``backend``, and ``noise_spec``. Supports two RNG strategies: stable
    per-trajectory streams (default) and a batched stream for speed. Initial
    conditions broadcast from shape ``(n_modes,)`` to ``(n_traj, n_modes)`` when
    needed. A lightweight progress callback can be invoked periodically with an
    ETA estimate after a configurable warm-up.

    Parameters
    ----------
    model : SDEModel
        Provides drift/diffusion and metadata (n_modes, noise_basis, noise_dim).
    ic : array-like
        Initial conditions; shape ``(n_modes,)`` or ``(n_traj, n_modes)``, complex-like.
    time : dict
        Time spec with keys: ``t0`` (float, optional), ``dt`` (float), ``steps`` (int).
    n_traj : int
        Number of trajectories to simulate.
    solver : str
        Solver ID (e.g., ``'euler'``, ``'milstein'``). In this version ``'milstein'``
        aliases to Euler–Maruyama.
    backend : str or BackendBase
        Backend name (resolved via the registry) or a concrete backend instance.
    noise_spec : NoiseSpec
        Real-noise channel specification (``'independent'`` | ``'correlated'``).
    seed : int, optional
        Single-stream RNG seed (used when ``master_seed``/``per_traj_seeds`` are not provided).
    master_seed : int, optional
        Master seed to spawn per-trajectory RNG streams via the backend.
    per_traj_seeds : list[int], optional
        Explicit per-trajectory seeds; takes precedence over ``master_seed``.
    save_every : int
        Decimation factor for persisted time-series (handled by callers/IO layer).
    return_stride : int
        Decimation factor for the returned TrajectorySet (analysis/plotting path).
    rng_stream : str
        RNG strategy: ``'per_trajectory'`` (stable per-trajectory streams; default)
        or ``'batched'`` (single RNG for vectorized sampling; faster, but per-trajectory
        invariance is not guaranteed if count/order changes).
    progress_cb : callable, optional
        Optional callback: ``(step_done, steps_total, eta_seconds, ic_index, ic_total)``.
    progress_interval_seconds : float
        Minimum wall-time between progress reports.
    ic_index : int
        Index of the current IC batch (for multi-IC runs) used in progress reports.
    ic_total : int
        Total number of IC batches for progress reports.
    warmup_min_steps : int
        Minimum steps before ETA estimation.
    warmup_min_seconds : float
        Minimum wall-time before ETA estimation.

    Returns
    -------
    TrajectorySetLike
        Backend-aware trajectory container with data shaped ``(n_traj, n_keep, n_modes)``
        where ``n_keep = steps // return_stride + 1`` and time grid ``(t0, dt * return_stride)``.

    Examples
    --------
    >>> from QPhaseSDE.core.engine import run
    >>> # model, noise_spec assumed constructed elsewhere
    >>> ts = run(model, ic=[0+0j, 0+0j], time={"t0": 0.0, "dt": 1e-3, "steps": 10},
    ...          n_traj=4, solver="euler", backend="numpy", noise_spec=noise_spec,
    ...          return_stride=2)
    >>> ts.data.shape  # doctest: +SKIP
    (4, 6, model.n_modes)
    """
    # Time grid
    t0 = float(time.get('t0', 0.0))
    dt = float(time['dt'])
    steps = int(time['steps'])

    # Backend selection via registry or accept instance
    if isinstance(backend, str):
        key = backend if ":" in backend else f"backend:{backend}"
        try:
            be = registry.create(key)
        except Exception:
            raise QPSBackendError(f"[404] Unsupported backend '{backend}'. Ensure it is registered.")
    else:
        be = backend

    # Pre-check state support for CuPy backends to provide clear error messages
    try:
        from ..states.factory import get_state_classes as _get_sc
        StateCls, _ = _get_sc(be)
        is_cupy = False
        try:
            is_cupy = str(be.backend_name()).lower() in ("cupy", "cp")
        except Exception:
            is_cupy = False
        if is_cupy and not (getattr(StateCls, "__module__", "").endswith("cupy_state")):
            raise QPSBackendError("[201] CuPy backend selected but GPU state containers are unavailable. Install cupy or switch to backend 'numba'/'numpy'.")
    except QPSBackendError:
        raise
    except Exception:
        # Non-fatal: proceed; errors will surface later if unsupported
        pass
    # RNG strategy: if per_traj_seeds provided, spawn per-trajectory RNGs; else based on rng_stream
    rng: Any
    if per_traj_seeds is not None and len(per_traj_seeds) == n_traj:
        # derive RNGs from provided integer seeds using backend rng()
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
    sampler = make_noise_model(noise_spec, be)

    # Initialize state; broadcast single-vector IC to all trajectories using backend
    y0 = be.asarray(ic)
    if getattr(y0, 'ndim', 1) == 1:
        n_modes = int(y0.shape[0])
        # Vectorized broadcast instead of Python loop
        y_full = be.zeros((n_traj, n_modes), dtype=complex)
        try:
            y_full[:] = y0  # broadcast along first axis
        except Exception:
            # Fallback to explicit tile if backend does not support broadcasting assignment
            import numpy as _np
            y_full = be.asarray(_np.tile(be.asarray(y0), (n_traj, 1)))
        y0 = y_full
    state = make_state(be, y=y0, t=t0, attrs={"backend": getattr(be, 'backend_name', lambda: 'backend')(), "interpretation": "ito"})

    # storage for returned data (analysis/plotting)
    rs = max(1, int(return_stride))
    n_keep = (steps // rs) + 1
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
    # Resolve integrator once per run (avoid per-step registry lookup)
    key = solver if ":" in solver else f"integrator:{solver}"
    # Temporary deprecation notice for milstein alias in v0.1.2
    try:
        if str(solver).lower() == "milstein":
            get_logger().warning("[994] 'milstein' currently aliases to Euler–Maruyama in v0.1.2; true Milstein will arrive in a future release.")
    except Exception:
        pass
    integrator = registry.create(key)

    for k in range(1, steps+1):
        # sample noise increment
        dW = sampler.sample(rng, state.n_traj, dt)  # (n_traj, M)

        # Integrate one step using resolved integrator (Milstein currently maps to EM)
        dy = integrator.step(state.y, t, dt, model, dW, be)

        state = make_state(be, y=state.y + dy, t=t + dt, attrs=state.attrs)
        t += dt

        if (k % rs) == 0:
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

    return make_trajectory_set(be, data=out, t0=t0, dt=dt*rs, meta={})