"""Engine Class (v0.2 Architecture)
----------------------------------

Object-oriented wrapper around the core simulation logic that supports
dependency injection of backend and integrator via constructor.

The Engine class now contains the full simulation logic, making the
functional run() interface a simple wrapper for backward compatibility.
"""

import time as _time
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field
from qphase.backend.base import BackendBase
from qphase.core.protocols import EngineBase, EngineManifest, ResultProtocol

from qphase_sde.integrator.base import Integrator
from qphase_sde.model import NoiseSpec, SDEModel
from qphase_sde.result import AnalysisResult, SDEResult
from qphase_sde.state import State, TrajectorySet

__all__ = ["Engine", "EngineConfig"]


class EngineConfig(BaseModel):
    """Configuration for the SDE Engine."""

    dt: float = Field(
        1e-3,
        description="Time step size",
        json_schema_extra={"scanable": True},
    )
    t0: float = Field(
        0.0,
        description="Start time",
        json_schema_extra={"scanable": True},
    )
    t1: float = Field(
        10.0,
        description="End time",
        json_schema_extra={"scanable": True},
    )
    t_end: float | None = Field(
        None,
        description="End time (alias for t1)",
        json_schema_extra={"scanable": True},
    )
    n_traj: int = Field(
        1,
        description="Number of trajectories",
        json_schema_extra={"scanable": True},
    )
    seed: int | None = Field(
        None,
        description="Random seed",
        json_schema_extra={"scanable": True},
    )
    ic: Any | None = Field(
        None,
        description="Initial conditions",
    )

    class ConfigSchema:
        """Pydantic model configuration for SDE engine.

        Allows extra fields for forward compatibility.
        """

        extra = "allow"


class EngineContext:
    """Engine runtime context for dependency injection."""

    def __init__(self):
        self.backend: BackendBase | None = None
        self.integrator: Integrator | None = None

    def set_backend(self, backend: BackendBase) -> None:
        self.backend = backend

    def set_integrator(self, integrator: Integrator) -> None:
        self.integrator = integrator

    def get_backend(self) -> BackendBase:
        if self.backend is None:
            raise RuntimeError(
                "Backend not set. Use set_backend() or pass backend to engine."
            )
        return self.backend

    def get_integrator(self) -> Integrator:
        if self.integrator is None:
            raise RuntimeError(
                "Integrator not set. Use set_integrator() or pass integrator to engine."
            )
        return self.integrator


_context = EngineContext()


def set_backend(backend: BackendBase) -> None:
    """Set global backend for dependency injection."""
    _context.set_backend(backend)


def set_integrator(integrator: Integrator) -> None:
    """Set global integrator for dependency injection."""
    _context.set_integrator(integrator)


def get_backend() -> BackendBase:
    """Get global backend from dependency injection."""
    return _context.get_backend()


def get_integrator() -> Integrator:
    """Get global integrator from dependency injection."""
    return _context.get_integrator()


# -----------------------------------------------------------------------------
# Engine Class
# -----------------------------------------------------------------------------


class Engine(EngineBase):
    """SDE simulation engine with dependency injection support.

    The Engine class provides both high-level simulation methods and
    dependency injection capabilities. All simulation logic is implemented
    in this class for better maintainability and testability.

    Parameters
    ----------
    config : EngineConfig, optional
        Configuration object.
    plugins : dict, optional
        Plugin dictionary.

    """

    name: ClassVar[str] = "sde"
    description: ClassVar[str] = "Stochastic Differential Equation Simulation Engine"
    config_schema: ClassVar[type[EngineConfig]] = EngineConfig
    manifest: ClassVar[EngineManifest] = EngineManifest(
        required_plugins={"backend", "model"},
        optional_plugins={"integrator", "analyser"},
        defaults={"integrator": "euler_maruyama"},
    )

    def __init__(
        self,
        config: EngineConfig | None = None,
        plugins: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize Engine with optional default backend and integrator.

        Parameters
        ----------
        config : EngineConfig, optional
            Configuration object (injected by Registry)
        plugins : dict, optional
            Plugin dictionary (injected by Registry)
        **kwargs : Any
            Additional arguments (e.g. 'backend', 'integrator' for legacy support)

        """
        self.config = config
        self.plugins = plugins or {}

        # Legacy support for direct injection via kwargs
        backend = kwargs.get("backend")
        integrator = kwargs.get("integrator")

        self._default_backend = self.plugins.get("backend", backend)
        self._default_integrator = self.plugins.get("integrator", integrator)

    def run(
        self,
        data: Any | None = None,
        *,
        progress_cb: Callable[[float | None, float | None, str, str | None], None]
        | None = None,
    ) -> ResultProtocol:
        """Execute the engine (Plugin Protocol)."""
        if not self.config:
            raise RuntimeError("Engine not configured.")

        model = self.plugins.get("model")
        if not model:
            raise RuntimeError("Engine requires 'model' plugin.")

        time_cfg = {
            "t0": self.config.t0,
            "dt": self.config.dt,
            "steps": int((self.config.t1 - self.config.t0) / self.config.dt),
        }

        ic = self.config.ic
        if ic is None:
            if hasattr(model, "default_ic"):
                ic = model.default_ic
            elif data is not None:
                ic = data
            else:
                raise RuntimeError("No IC provided.")

        # Adapter for progress callback
        # SDE engine uses: (k, steps, eta, ic_index, ic_total)
        # Protocol expects: (percent, total_duration_estimate, message, stage)
        sde_progress_cb = None
        if progress_cb is not None:

            def _sde_cb(
                k: int, steps: int, eta: float, ic_index: int, ic_total: int
            ) -> None:
                percent = k / steps if steps > 0 else 0.0

                # Calculate total duration estimate from ETA
                # ETA = Remaining = Total * (1 - p)
                # Total = ETA / (1 - p)
                total_est = None
                if eta is not None and not np.isnan(eta) and percent < 1.0:
                    try:
                        total_est = eta / (1.0 - percent)
                    except ZeroDivisionError:
                        pass

                msg = f"Traj {ic_index + 1}/{ic_total} | Step {k}/{steps}"
                progress_cb(percent, total_est, msg, "sampling")

            sde_progress_cb = _sde_cb

        traj_set = self.run_sde(
            model=model,
            ic=ic,
            time=time_cfg,
            n_traj=self.config.n_traj,
            seed=self.config.seed,
            progress_cb=sde_progress_cb,
        )

        # Check for analysers
        analysers = self.plugins.get("analyser")
        if not analysers:
            return SDEResult(trajectory=traj_set, kind="trajectory")

        # If analysers is a single instance (backward compat), wrap it
        if not isinstance(analysers, dict):
            analysers = {"default": analysers}

        results = {}
        for name, analyser in analysers.items():
            # analyser.analyze(data, backend)
            # Note: analyser.analyze returns a ResultProtocol
            results[name] = analyser.analyze(traj_set, self._default_backend)

        return AnalysisResult(results=results, meta={})

    def run_sde(
        self,
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
        progress_cb: Callable[[int, int, float, int, int], None] | None = None,
        progress_interval_seconds: float = 1.0,
        ic_index: int = 0,
        ic_total: int = 1,
        warmup_min_steps: int = 0,
        warmup_min_seconds: float = 0.0,
        rng: Any | None = None,
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
        model : SDEModel
            SDE model providing drift/diffusion and metadata
        ic : array-like
            Initial conditions
        time : dict
            Time spec with keys: t0 (optional), dt, steps
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
        progress_cb : callable, optional
            Progress callback function
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

        Returns
        -------
        TrajectorySet
            The simulation result.

        """
        # Resolve dependencies
        be = backend or self._default_backend
        if be is None:
            # Fallback to global default if available
            try:
                be = get_backend()
            except RuntimeError as err:
                raise RuntimeError("No backend provided or configured.") from err

        integrator = solver or self._default_integrator
        if integrator is None:
            try:
                integrator = get_integrator()
            except RuntimeError as err:
                raise RuntimeError("No integrator provided or configured.") from err

        # Parse time config
        t0 = float(time.get("t0", 0.0))
        dt = float(time["dt"])
        steps = int(time["steps"])

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

            y0 = be.asarray(ic)

        # Broadcast IC if necessary to match n_traj
        # Expected shape: (n_traj, n_modes)
        if y0.ndim == 1:
            y0 = be.expand_dims(y0, 0)

        if y0.shape[0] != n_traj:
            if y0.shape[0] == 1:
                # Broadcast
                y0 = be.repeat(y0, n_traj, axis=0)

        state = State(
            data=y0,
            t=t0,
            meta={
                "back": getattr(be, "backend_name", lambda: "backend")(),
                "interpretation": "ito",
            },
        )

        # Setup RNG if not pre-configured
        if rng is None:
            try:
                if per_traj_seeds is not None and len(per_traj_seeds) == n_traj:
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
        n_keep = (steps // rs) + 1
        out = be.empty((n_traj, n_keep, model.n_modes), dtype=complex)
        out[:, 0, :] = state.data
        keep_counter = 1

        t = t0
        # Progress tracking
        last_report_step = 0
        last_report_time = None
        start_time = _time.monotonic()
        next_report_time = start_time + max(0.1, float(progress_interval_seconds))
        s_ema = None
        alpha = 0.2
        warmup_steps_thr = max(0, int(warmup_min_steps))
        warmup_time_thr = max(0.0, float(warmup_min_seconds))

        # Main simulation loop
        for k in range(1, steps + 1):
            # Sample noise increment: dW ~ N(0, dt)
            # rng must be set by this point
            assert rng is not None, "RNG not initialized"
            dW = be.randn(rng, (state.n_traj, model.noise_dim), dtype=float) * (dt**0.5)

            # Integrate one step
            dy = integrator.step(state.data, t, dt, model, dW, be)

            state = State(data=state.data + dy, t=t + dt, meta=state.meta)
            t += dt

            # Save data at stride intervals
            if (k % rs) == 0:
                out[:, keep_counter, :] = state.data
                keep_counter += 1

            # Progress reporting
            if progress_cb is not None:
                now = _time.monotonic()
                if now >= next_report_time:
                    steps_delta = k - last_report_step
                    if steps_delta > 0:
                        dt_wall = now - (
                            last_report_time
                            if last_report_time is not None
                            else start_time
                        )
                        s_inst = dt_wall / steps_delta
                        if s_ema is None:
                            s_ema = s_inst
                        else:
                            s_ema = alpha * s_inst + (1.0 - alpha) * s_ema
                    last_report_step = k
                    last_report_time = now
                    next_report_time = now + max(0.1, float(progress_interval_seconds))

                    # ETA estimation with warm-up
                    eta = float("nan")
                    elapsed = now - start_time
                    if (
                        s_ema is not None
                        and k >= warmup_steps_thr
                        and elapsed >= warmup_time_thr
                    ):
                        remaining = max(0, steps - k)
                        eta = remaining * float(s_ema)

                    try:
                        progress_cb(k, steps, eta, ic_index, ic_total)
                    except Exception:
                        # Never let progress reporting break the simulation
                        pass

        return TrajectorySet(data=out, t0=t0, dt=dt * rs, meta={})
