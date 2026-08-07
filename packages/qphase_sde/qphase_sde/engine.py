"""qphase_sde: SDE Engine
---------------------------------------------------------
Object-oriented wrapper around the core simulation logic that supports
dependency injection of backend and integrator via constructor.

The Engine class now contains the full simulation logic, making the
functional run() interface a simple wrapper for backward compatibility.

Public API
----------
``Engine`` : Main simulation engine class.
``EngineConfig`` : Configuration model for the engine.
"""

import logging
import time as _time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy, get_xp
from qphase.core.protocols import EngineBase, EngineManifest, ResultProtocol

from qphase_sde.buffers import SDEBufferCache
from qphase_sde.integrator.base import Integrator
from qphase_sde.model import NoiseSpec, SDEModel
from qphase_sde.planning import SDEExecutionPlan, build_execution_plan
from qphase_sde.result import SDEResult
from qphase_sde.state import TrajectorySet

__all__ = ["Engine", "EngineConfig", "TrajectoryDivergenceError"]

log = logging.getLogger(__name__)


class TrajectoryDivergenceError(RuntimeError):
    """Raised when an SDE trajectory leaves a configured state-norm bound."""


@dataclass(frozen=True)
class _GroupedRNG:
    """Independent RNG handles grouped by scan point inside one fused tile."""

    handles: tuple[Any, ...]
    group_size: int


class EngineConfig(BaseModel):
    """Configuration for the SDE Engine.

    Organized into logical groups:
    1. Time Domain: t0, t1, dt
    2. Ensemble: n_traj, seed, ic
    3. Adaptive Stepping: adaptive, atol, rtol, min_dt, max_dt
    4. Output Control: save_stride
    """

    model_config = ConfigDict(extra="allow")

    # --- Time Domain ---
    t0: float = Field(
        0.0,
        description="Start time",
        json_schema_extra={"scanable": False},
    )
    t1: float = Field(
        10.0,
        description="End time",
        json_schema_extra={"scanable": False},
    )
    dt: float = Field(
        1e-3,
        description="Time step size (initial step for adaptive)",
        json_schema_extra={"scanable": False},
    )

    # --- Ensemble ---
    n_traj: int = Field(
        1,
        description="Number of trajectories",
        json_schema_extra={"scanable": False},
    )
    trajectory_batching: Literal["auto", "off", "required"] = Field(
        "auto",
        description="Memory-aware batching across independent trajectories",
        json_schema_extra={"scanable": False},
    )
    trajectory_batch_size: int | None = Field(
        None,
        ge=1,
        description="Optional physical trajectory batch-size override",
        json_schema_extra={"scanable": False},
    )
    seed: int | None = Field(
        None,
        description="Random seed",
        json_schema_extra={"scanable": True},
    )
    ic: Any | None = Field(
        None,
        description="Initial conditions (list or array)",
        json_schema_extra={"scanable": True},
    )

    # --- Adaptive Stepping ---
    adaptive: bool = Field(
        False,
        description="Enable adaptive stepping (if supported by integrator)",
        json_schema_extra={"scanable": False},
    )
    atol: float = Field(
        1e-6,
        description="Absolute tolerance for adaptive stepping",
        json_schema_extra={"scanable": False},
    )
    rtol: float = Field(
        1e-3,
        description="Relative tolerance for adaptive stepping",
        json_schema_extra={"scanable": False},
    )
    min_dt: float = Field(
        1e-9,
        description="Minimum time step for adaptive stepping",
        json_schema_extra={"scanable": False},
    )
    max_dt: float = Field(
        10.0,
        description="Maximum time step for adaptive stepping",
        json_schema_extra={"scanable": False},
    )

    # --- Output Control ---
    save_stride: int = Field(
        1,
        ge=1,  # Must be at least 1
        description="Save every N-th step to the result trajectory",
        json_schema_extra={"scanable": False},
    )

    keep_traj: bool | None = Field(
        None,
        description=(
            "Force keeping trajectory data. None=Auto (drop if analyzed), "
            "True=Keep, False=Drop"
        ),
        json_schema_extra={"scanable": False},
    )

    record_modes: list[int] | None = Field(
        None,
        min_length=1,
        description="Physical mode indices to retain; None records every mode",
        json_schema_extra={"scanable": False},
    )

    max_state_norm: float | None = Field(
        None,
        gt=0.0,
        description=(
            "Optional Euclidean-norm guard across all modes. The simulation "
            "fails before analysis when any trajectory exceeds this value."
        ),
        json_schema_extra={"scanable": False},
    )

    state_check_interval_steps: int = Field(
        1024,
        ge=1,
        description=(
            "Integration-step interval for max_state_norm checks. Smaller "
            "values detect escape sooner but synchronize accelerators more often."
        ),
        json_schema_extra={"scanable": False},
    )

    mode: Literal["simulate", "analyze"] = Field(
        "simulate",
        description=(
            "Engine execution mode. 'simulate' runs the SDE; "
            "'analyze' runs analysers on upstream input data."
        ),
        json_schema_extra={"scanable": False},
    )


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

    name: ClassVar[str] = "SDE"
    description: ClassVar[str] = "Stochastic Differential Equation Simulation Engine"
    config_schema: ClassVar[type[EngineConfig]] = EngineConfig
    manifest: ClassVar[EngineManifest] = EngineManifest(
        required_plugins={"backend", "model", "integrator"},
        optional_plugins={"analyser"},
        defaults={"integrator": "euler_maruyama"},
        input_plugins={"analyser"},
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

        # Resolve Backend
        # Scheduler might pass a dict of backends if multiple are available
        p_backend = self.plugins.get("backend", backend)
        if isinstance(p_backend, dict):
            # If multiple available, check defaults or pick first
            default_name = self.manifest.defaults.get("backend")
            if default_name and default_name in p_backend:
                self._default_backend = p_backend[default_name]
            elif p_backend:  # Ensure not empty
                # Pick arbitrary first one if no preference
                self._default_backend = next(iter(p_backend.values()))
            else:
                self._default_backend = None
        else:
            self._default_backend = p_backend

        # Resolve Integrator
        # Scheduler might pass a dict of integrators if multiple are available
        p_integrator = self.plugins.get("integrator", integrator)
        if isinstance(p_integrator, dict):
            # If multiple available, check defaults or pick first
            default_name = self.manifest.defaults.get("integrator")
            if default_name and default_name in p_integrator:
                self._default_integrator = p_integrator[default_name]
            elif p_integrator:  # Ensure not empty
                # Pick arbitrary first one if no preference
                self._default_integrator = next(iter(p_integrator.values()))
            else:
                self._default_integrator = None
        else:
            self._default_integrator = p_integrator

    def run(
        self,
        data: Any | None = None,
        *,
        context: Any | None = None,
        progress_cb: (
            Callable[[float | None, float | None, str, str | None], None] | None
        ) = None,
    ) -> ResultProtocol:
        """Execute the engine (Plugin Protocol)."""
        if not self.config:
            raise RuntimeError("Engine not configured.")

        reporter = context.progress if context is not None else None
        if reporter is None and progress_cb is not None:
            from qphase.core.progress import ProgressReporter

            reporter = ProgressReporter.wrap_legacy(progress_cb)

        if getattr(self.config, "mode", "simulate") == "analyze":
            return self._run_analyze(data)
        grid = context.parameter_grid if context is not None else None
        model = self._required_model()
        analysers = self._normalised_analysers()
        backend = self._required_backend()
        integrator = self._required_integrator()
        plan = build_execution_plan(
            config=self.config,
            grid=grid,
            model=model,
            backend=backend,
            integrator=integrator,
            analysers=analysers,
            resources=getattr(context, "resources", None),
        )
        plan_payload = plan.to_dict()
        log.info("SDE execution plan: %s", plan_payload)
        if context is not None and isinstance(getattr(context, "metadata", None), dict):
            context.metadata["execution_plan"] = plan_payload
        if reporter is not None and (
            plan.tile_count > 1 or plan.trajectory_batch_count > 1
        ):
            reporter.status(
                self._execution_plan_summary(plan),
                stage="planning",
                metadata={"execution_plan": plan_payload},
                importance="normal",
            )
        if grid is not None:
            from qphase_sde.scan import SDEParameterGridAdapter, SDEScanResult

            adapter = SDEParameterGridAdapter(self, grid)
            if plan.stream_analysis and (
                plan.tile_count > 1 or plan.trajectory_batch_count > 1
            ):
                combined: ResultProtocol = self._run_scan_tiled(
                    data,
                    adapter=adapter,
                    plan=plan,
                    reporter=reporter,
                    context=context,
                )
            else:
                with adapter:
                    combined = self._run_simulate(
                        data, reporter=reporter, context=context
                    )
            if not isinstance(combined, SDEResult):
                raise TypeError("SDE simulation did not return an SDEResult")
            combined.meta.update(
                {
                    "scan_shape": grid.shape,
                    "scan_combine": grid.combine,
                    "execution_plan": plan.to_dict(),
                }
            )
            return SDEScanResult(
                combined,
                grid,
                adapter.base_params,
                adapter.base_n_traj,
            )
        if plan.trajectory_batch_count > 1:
            result: ResultProtocol = self._run_trajectory_batched(
                data,
                plan=plan,
                reporter=reporter,
                context=context,
                point_index=0,
                master_seed=self.config.seed,
            )
        else:
            result = self._run_simulate(data, reporter=reporter, context=context)
        if isinstance(result, SDEResult):
            result.meta.setdefault("execution_plan", plan.to_dict())
        return result

    @staticmethod
    def _execution_plan_summary(plan: SDEExecutionPlan) -> str:
        parts = [
            f"SDE plan: {plan.tile_count} scan tile(s)",
            f"{plan.trajectory_batch_size} trajectories/batch",
        ]
        if plan.trajectory_batch_count > 1:
            parts.append(f"{plan.trajectory_batch_count} batches/point")
        if plan.budget_bytes is not None:
            parts.append(f"{plan.budget_bytes / (1024**3):.2f} GiB budget")
        return "; ".join(parts)

    def _required_model(self) -> Any:
        model = self.plugins.get("model")
        if model is None or isinstance(model, dict):
            raise RuntimeError("SDE engine requires exactly one model plugin")
        return model

    def _required_backend(self) -> BackendBase:
        if self._default_backend is None:
            raise RuntimeError("SDE engine requires a backend plugin")
        return self._default_backend

    def _required_integrator(self) -> Integrator:
        if self._default_integrator is None:
            raise RuntimeError("SDE engine requires an integrator plugin")
        return self._default_integrator

    def _normalised_analysers(self) -> dict[str, Any]:
        analysers = self.plugins.get("analyser")
        if not analysers:
            return {}
        if isinstance(analysers, dict):
            return dict(analysers)
        return {getattr(analysers, "name", "analyser"): analysers}

    def _release_backend_pool(self) -> None:
        backend = self._default_backend
        release = getattr(backend, "free_all_blocks", None)
        if callable(release):
            release()

    @classmethod
    def _analysis_to_host(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: cls._analysis_to_host(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._analysis_to_host(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._analysis_to_host(item) for item in value)
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return convert_to_numpy(value)
        return value

    def _run_scan_tiled(
        self,
        data: Any | None,
        *,
        adapter: Any,
        plan: SDEExecutionPlan,
        reporter: Any | None,
        context: Any | None,
    ) -> SDEResult:
        """Integrate and analyze scan tiles without retaining the full trajectory."""
        assert self.config is not None
        analysers = self._normalised_analysers()
        accumulated: dict[str, list[Any]] = {name: [] for name in analysers}
        result_meta: dict[str, Any] = {}
        total_work = plan.scan_size * plan.n_traj_per_point * plan.steps

        for tile_index, start in enumerate(
            range(0, plan.scan_size, plan.scan_tile_size)
        ):
            stop = min(start + plan.scan_tile_size, plan.scan_size)
            point_count = stop - start
            cancellation = getattr(context, "cancellation", None)
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            if reporter is not None:
                reporter.status(
                    f"SDE scan tile {tile_index + 1}/{plan.tile_count} "
                    f"(points {start + 1}-{stop})",
                    stage="sampling",
                    metadata={"tile_index": tile_index, "tile_count": plan.tile_count},
                )

            with adapter.tile(start, stop):
                if plan.trajectory_batch_count > 1:
                    if point_count != 1:
                        raise RuntimeError(
                            "trajectory-batched SDE scans require one point per tile"
                        )
                    tile_result: ResultProtocol = self._run_trajectory_batched(
                        data,
                        plan=plan,
                        reporter=reporter,
                        context=context,
                        point_index=start,
                        master_seed=adapter.master_seed,
                    )
                else:
                    tile_result = self._run_simulate(
                        data,
                        reporter=reporter,
                        context=context,
                        rng_group_seeds=adapter.point_seeds(start, stop),
                        rng_group_size=adapter.base_n_traj,
                        progress_offset=(start * plan.n_traj_per_point * plan.steps),
                        progress_scale=point_count * plan.n_traj_per_point,
                        progress_total=total_work,
                        progress_label=f"tile {tile_index + 1}/{plan.tile_count}",
                    )
            if not isinstance(tile_result, SDEResult):
                raise TypeError("SDE tile did not return an SDEResult")
            if tile_result.trajectory is not None:
                raise RuntimeError(
                    "resource-aware scan tiling requires the tile trajectory to "
                    "be released after analysis"
                )
            for name in analysers:
                value = self._analysis_to_host(tile_result.analysis.get(name))
                if point_count == 1:
                    accumulated[name].append(value)
                elif isinstance(value, list) and len(value) == point_count:
                    accumulated[name].extend(value)
                else:
                    raise TypeError(
                        f"analyser {name!r} did not return {point_count} "
                        "point results for an SDE scan tile"
                    )
            if not result_meta:
                result_meta.update(tile_result.meta)
            self._release_backend_pool()
            if reporter is not None:
                reporter.update(
                    completed=stop * plan.n_traj_per_point * plan.steps,
                    total=total_work,
                    unit="trajectory-step",
                    stage="sampling",
                    message=(
                        f"Completed SDE scan tile {tile_index + 1}/{plan.tile_count}"
                    ),
                    metadata={"tile_index": tile_index, "tile_count": plan.tile_count},
                )

        result_meta.update(
            {
                "params": dict(adapter.base_params),
                "rng_master_seed": adapter.master_seed,
                "rng_strategy": plan.rng_strategy,
                "drop_trajectory_reason": "analyzed_by_scan_tile",
                "execution_plan": plan.to_dict(),
            }
        )
        return SDEResult(trajectory=None, analysis=accumulated, meta=result_meta)

    def _run_trajectory_batched(
        self,
        data: Any | None,
        *,
        plan: SDEExecutionPlan,
        reporter: Any | None,
        context: Any | None,
        point_index: int,
        master_seed: int | None,
    ) -> SDEResult:
        """Integrate one parameter point in bounded trajectory batches."""
        assert self.config is not None
        analysers = self._normalised_analysers()
        accumulators = {
            name: analyser.create_result_accumulator()
            for name, analyser in analysers.items()
        }
        if len(accumulators) != len(analysers):
            raise RuntimeError(
                "trajectory batching requires an accumulator for every analyser"
            )

        if master_seed is None:
            master_seed = int(
                np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0]
            )
        master_seed = int(master_seed) % (1 << 64)
        total_trajectories = plan.n_traj_per_point
        base_params = dict(self._required_model().params)
        result_meta: dict[str, Any] = {}

        for batch_index, start in enumerate(
            range(0, total_trajectories, plan.trajectory_batch_size)
        ):
            stop = min(start + plan.trajectory_batch_size, total_trajectories)
            count = stop - start
            cancellation = getattr(context, "cancellation", None)
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            group_size = plan.logical_rng_group_size
            if count % group_size:
                if stop != total_trajectories or count >= group_size:
                    raise RuntimeError(
                        "trajectory batch boundaries must align with logical RNG groups"
                    )
                group_size = count
            first_group = start // plan.logical_rng_group_size
            group_count = count // group_size
            seeds = tuple(
                int(
                    np.random.SeedSequence(
                        [master_seed, point_index, first_group + offset]
                    ).generate_state(1, dtype=np.uint64)[0]
                )
                for offset in range(group_count)
            )
            if reporter is not None:
                reporter.status(
                    f"SDE trajectory batch {batch_index + 1}/"
                    f"{plan.trajectory_batch_count}",
                    stage="sampling",
                    metadata={
                        "trajectory_batch_index": batch_index,
                        "trajectory_batch_count": plan.trajectory_batch_count,
                    },
                )
            with self._trajectory_batch_scope(start, stop, total_trajectories):
                partial = self._run_simulate(
                    data,
                    reporter=reporter,
                    context=context,
                    rng_group_seeds=seeds,
                    rng_group_size=group_size,
                    progress_offset=(point_index * total_trajectories + start)
                    * plan.steps,
                    progress_scale=count,
                    progress_total=(plan.scan_size * total_trajectories * plan.steps),
                    progress_label=(
                        f"trajectory batch {batch_index + 1}/"
                        f"{plan.trajectory_batch_count}"
                    ),
                )
            if not isinstance(partial, SDEResult):
                raise TypeError("SDE trajectory batch did not return an SDEResult")
            if partial.trajectory is not None:
                raise RuntimeError(
                    "trajectory batching requires keep_traj=false after analysis"
                )
            for name, accumulator in accumulators.items():
                accumulator.update(self._analysis_to_host(partial.analysis[name]))
            if not result_meta:
                result_meta.update(partial.meta)
            self._release_backend_pool()

        result_meta.update(
            {
                "params": base_params,
                "rng_master_seed": master_seed,
                "rng_strategy": plan.rng_strategy,
                "drop_trajectory_reason": "online_trajectory_aggregation",
                "trajectory_batch_size": plan.trajectory_batch_size,
                "trajectory_batch_count": plan.trajectory_batch_count,
                "execution_plan": plan.to_dict(),
            }
        )
        return SDEResult(
            trajectory=None,
            analysis={
                name: accumulator.finalize()
                for name, accumulator in accumulators.items()
            },
            meta=result_meta,
        )

    @contextmanager
    def _trajectory_batch_scope(self, start: int, stop: int, total_trajectories: int):
        """Temporarily slice trajectory-shaped configuration and model values."""
        assert self.config is not None
        model = self._required_model()
        original_n_traj = int(self.config.n_traj)
        original_ic = self.config.ic
        original_params = dict(model.params)
        sliced_params = {
            name: self._slice_trajectory_value(value, start, stop, total_trajectories)
            for name, value in original_params.items()
        }
        self.config.n_traj = stop - start
        self.config.ic = self._slice_trajectory_value(
            original_ic, start, stop, total_trajectories
        )
        self._replace_model_params(model, sliced_params)
        try:
            yield
        finally:
            self.config.n_traj = original_n_traj
            self.config.ic = original_ic
            self._replace_model_params(model, original_params)

    @staticmethod
    def _slice_trajectory_value(
        value: Any, start: int, stop: int, total_trajectories: int
    ) -> Any:
        if value is None:
            return None
        try:
            array = np.asarray(value)
        except Exception:
            return value
        if array.ndim > 0 and array.shape[0] == total_trajectories:
            return value[start:stop]
        return value

    @staticmethod
    def _replace_model_params(model: Any, params: dict[str, Any]) -> None:
        if hasattr(model, "_params"):
            model._params = params
            return
        current = getattr(model, "params", None)
        if isinstance(current, dict):
            current.clear()
            current.update(params)
            return
        raise TypeError(f"model {model.name!r} does not expose mutable parameters")

    def _run_analyze(self, data: Any | None) -> ResultProtocol:
        """Run analysers on upstream input data without performing a simulation.

        This mode is used for cross-job postprocessing: the scheduler passes an
        ``AggregateResult``, a directory ``Path``, or a single ``SDEResult`` as
        ``data``, and the configured analysers produce derived outputs.
        """
        assert self.config is not None

        from qphase.backend.numpy_backend import NumpyBackend

        analysis_results: dict[str, Any] = {}
        meta: dict[str, Any] = {"mode": "analyze"}

        analysers = self.plugins.get("analyser")
        if not analysers:
            return SDEResult(trajectory=None, meta=meta, analysis=analysis_results)

        if not isinstance(analysers, dict):
            analysers = {getattr(analysers, "name", "analyser"): analysers}

        be = NumpyBackend()
        output_dir = getattr(self.config, "output_dir", None)
        for name, analyzer in analysers.items():
            if hasattr(analyzer, "analyze"):
                # Provide a place for analysers to write aggregated outputs.
                if output_dir is not None:
                    analyzer.output_dir = output_dir
                res = analyzer.analyze(data, be)
                analysis_results[name] = res.data_dict

        return SDEResult(trajectory=None, meta=meta, analysis=analysis_results)

    def _run_simulate(
        self,
        data: Any | None,
        *,
        reporter: Any | None = None,
        context: Any | None = None,
        rng_group_seeds: tuple[int, ...] | None = None,
        rng_group_size: int | None = None,
        progress_offset: int = 0,
        progress_scale: int | None = None,
        progress_total: int | None = None,
        progress_label: str | None = None,
    ) -> ResultProtocol:
        """Execute SDE simulation and optional per-job analysis."""
        assert self.config is not None

        model = self.plugins.get("model")
        if not model:
            raise RuntimeError("Engine requires 'model' plugin.")

        time_cfg = {
            "t0": self.config.t0,
            "dt": self.config.dt,
            "steps": int((self.config.t1 - self.config.t0) / self.config.dt),
        }
        if progress_scale is None:
            progress_scale = int(self.config.n_traj)
        effective_progress_total = (
            progress_total
            if progress_total is not None
            else progress_offset + progress_scale * time_cfg["steps"]
        )

        ic = self.config.ic
        if ic is None:
            if hasattr(model, "default_ic"):
                ic = model.default_ic
            elif data is not None:
                ic = data
            else:
                raise RuntimeError("No IC provided.")

        # Adapt the integration loop's counters to structured, monotonic work.
        sde_progress_cb = None
        if reporter is not None:

            def _sde_cb(
                k: int, steps: int, eta: float, ic_index: int, ic_total: int
            ) -> None:
                del eta
                prefix = f"{progress_label} | " if progress_label else ""
                msg = f"{prefix}Traj {ic_index + 1}/{ic_total} | Step {k}/{steps}"
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
                    stage="sampling",
                    message=msg,
                    metadata=metadata,
                )

            sde_progress_cb = _sde_cb

        traj_set: TrajectorySet | None = self.run_sde(
            model=model,
            ic=ic,
            time=time_cfg,
            n_traj=self.config.n_traj,
            seed=self.config.seed,
            return_stride=self.config.save_stride,
            progress_cb=sde_progress_cb,
            rng_group_seeds=rng_group_seeds,
            rng_group_size=rng_group_size,
        )
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

        # Determine the backend name used for simulation to decide whether the
        # trajectory can be retained. GPU backends (especially cupy) should not
        # keep the full raw trajectory by default, because downstream plotting
        # expects numpy arrays and retaining it would cause large D2H transfers.
        backend_name = ""
        if self._default_backend is not None:
            try:
                backend_name = self._default_backend.backend_name()
            except Exception:
                backend_name = ""

        # Run analyzers if configured
        analysis_results: dict[str, Any] = {}
        meta: dict[str, Any] = {}

        analysers = self.plugins.get("analyser")
        if analysers:
            # Normalize to dict of name -> instance
            if not isinstance(analysers, dict):
                # Single instance
                analysers = {getattr(analysers, "name", "analyser"): analysers}

            if reporter is not None:
                analyser_names = tuple(str(name) for name in analysers)
                stage = (
                    "spectral-analysis"
                    if any(name.lower() == "psd" for name in analyser_names)
                    else "analysis"
                )
                reporter.status(
                    f"Analyzing {self.config.n_traj} trajectories",
                    stage=stage,
                    metadata={"analysers": analyser_names},
                )

            # Get backend used for simulation
            be = self._default_backend
            if be is None:
                try:
                    be = get_backend()
                except RuntimeError:
                    # Should not happen if run_sde succeeded, but for safety
                    from qphase.backend.numpy_backend import NumpyBackend

                    be = NumpyBackend()

            # Detect batch mode: the scheduler fuses scan points into one ensemble.
            n_scan = getattr(self.config, "_batch_scan_count", 1)
            if not isinstance(n_scan, int) or n_scan < 1:
                n_scan = 1

            if n_scan > 1 and traj_set is not None:
                # Split the combined trajectory into per-scan-point slices and run
                # each analyzer on each slice so that PSD/dist/etc. remain per-point.
                data = traj_set.data
                n_total_traj = data.shape[0]
                if n_total_traj % n_scan != 0:
                    raise ValueError(
                        f"Batch trajectory has {n_total_traj} trajectories, "
                        f"not divisible by scan count {n_scan}"
                    )
                n_traj_per_scan = n_total_traj // n_scan

                per_scan_results: list[dict[str, Any]] = []
                for i in range(n_scan):
                    start = i * n_traj_per_scan
                    end = start + n_traj_per_scan
                    sub_traj = TrajectorySet(
                        data=data[start:end],
                        t0=traj_set.t0,
                        dt=traj_set.dt,
                        meta=dict(traj_set.meta),
                    )
                    scan_res: dict[str, Any] = {}
                    for name, analyzer in analysers.items():
                        if hasattr(analyzer, "analyze"):
                            res = analyzer.analyze(sub_traj, be)
                            scan_res[name] = res.data_dict
                    per_scan_results.append(scan_res)

                # Store as lists keyed by analyzer name; the result splitter will
                # distribute the i-th element to the i-th original job.
                for name in analysers.keys():
                    analysis_results[name] = [r.get(name) for r in per_scan_results]
            else:
                for name, analyzer in analysers.items():
                    if hasattr(analyzer, "analyze"):
                        res = analyzer.analyze(traj_set, be)
                        # Analyzer names come from the plugin configuration keys.
                        # But name comes from config key
                        analysis_results[name] = res.data_dict

            # Decide whether to keep trajectory based on config
            should_keep = self.config.keep_traj

            # GPU backends force trajectory drop to avoid large device-to-host
            # transfers and GPU memory bloat. Users who need raw trajectories for
            # plotting should run a separate job with the numpy backend.
            if should_keep and backend_name == "cupy":
                import logging

                logging.getLogger(__name__).warning(
                    "CuPy backend forces keep_traj=False to avoid large "
                    "device-to-host transfers. Use the numpy backend if you "
                    "need to keep raw trajectories."
                )
                should_keep = False

            if should_keep is None:
                # Default behavior: drop if analyzed to save resources
                should_keep = not bool(analysis_results)

            if not should_keep and traj_set is not None:
                # Preserve metadata before dropping trajectory
                meta["t0"] = traj_set.t0
                meta["dt"] = traj_set.dt
                meta.update(traj_set.meta)
                # Record reason for dropping
                meta["drop_trajectory_reason"] = (
                    "analyzed" if analysis_results else "user_requested"
                )
                # Drop trajectory to save memory
                traj_set = None

        # Ensure model parameters are in metadata
        if model:
            if hasattr(model, "config") and hasattr(model.config, "model_dump"):
                meta["params"] = model.config.model_dump()
            elif hasattr(model, "parameters"):
                meta["params"] = model.parameters
            elif hasattr(model, "params"):
                meta["params"] = model.params

        return SDEResult(trajectory=traj_set, meta=meta, analysis=analysis_results)

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
        rng_group_seeds: tuple[int, ...] | None = None,
        rng_group_size: int | None = None,
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
        rng_group_seeds : tuple[int, ...], optional
            Stable seeds for scan-point groups inside a fused tile.
        rng_group_size : int, optional
            Number of adjacent trajectories driven by each grouped seed.

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
            if self.config is None or self.config.max_state_norm is None
            else float(self.config.max_state_norm)
        )
        state_check_interval = (
            1 if self.config is None else int(self.config.state_check_interval_steps)
        )
        if state_norm_limit is not None:
            self._check_state_norm(y, t, state_norm_limit)
        next_state_check_step = state_check_interval

        # Setup RNG if not pre-configured
        if rng is None:
            try:
                if rng_group_seeds is not None:
                    group_size = int(rng_group_size or 0)
                    if group_size < 1 or len(rng_group_seeds) * group_size != n_traj:
                        raise ValueError(
                            "rng_group_seeds and rng_group_size do not cover n_traj"
                        )
                    handles = tuple(be.rng(int(value)) for value in rng_group_seeds)
                    rng = (
                        handles[0]
                        if len(handles) == 1
                        else _GroupedRNG(handles, group_size)
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
        if self.config is not None and self.config.record_modes is not None:
            record_modes = tuple(int(mode) for mode in self.config.record_modes)
            if len(set(record_modes)) != len(record_modes):
                raise ValueError("record_modes must not contain duplicates")
            if any(mode < 0 or mode >= model.n_modes for mode in record_modes):
                raise ValueError(f"record_modes must be within 0..{model.n_modes - 1}")
        else:
            record_modes = tuple(range(model.n_modes))
        n_keep = (steps // rs) + 1
        out = be.empty((n_traj, n_keep, len(record_modes)), dtype=y.dtype)
        out[:, 0, :] = y[:, record_modes]
        keep_counter = 1

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
        next_save_time = t0 + save_dt

        # Adaptive stepping setup
        use_adaptive = False
        noise_spec = None
        # Ensure config is not None (use default if None)
        config = self.config if self.config is not None else EngineConfig()
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

        while t < t_end - 1e-12:
            if use_chunked:
                assert rng is not None, "RNG not initialized"
                assert callable(chunk_step)
                n_chunk = min(requested_chunk_steps, steps - k)
                noise_dtype = y.real.dtype if hasattr(y, "real") else y.dtype
                noise_shape = (n_chunk, n_traj, model.noise_dim)
                raw_noise = buf_cache.get(noise_shape, noise_dtype)
                try:
                    self._draw_standard_normal_into(be, rng, raw_noise)
                    dt_sqrt = be.asarray(dt**0.5, dtype=raw_noise.dtype)
                    raw_noise *= dt_sqrt
                    save_offsets = tuple(
                        offset
                        for offset in range(1, n_chunk + 1)
                        if (k + offset) % rs == 0
                    )
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
                    self._check_state_norm(y, next_t, state_norm_limit)
                    next_state_check_step = (
                        next_k // state_check_interval + 1
                    ) * state_check_interval
                n_saved = len(save_offsets)
                if n_saved:
                    end = keep_counter + n_saved
                    out[:, keep_counter:end, :] = result.saved_states
                    keep_counter = end
                    next_save_time += n_saved * save_dt
                k += n_chunk
                t = t0 + k * dt
            else:
                k += 1
                y_prev = y
                t_prev = t

                if use_adaptive:
                    assert noise_spec is not None
                    y_next, t_next, next_dt, error = integrator.step_adaptive(
                        y, t, current_dt, tol, model, noise_spec, be, rng
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
                        self._draw_standard_normal_into(be, rng, raw_noise)
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
                    self._check_state_norm(y, t, state_norm_limit)
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
                        next_save_time += save_dt
                    finally:
                        buf_cache.put(y_interp)

            # Progress reporting
            if progress_cb is not None:
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
                        progress_cb(
                            int(progress * steps), steps, eta, ic_index, ic_total
                        )
                    except Exception:
                        pass

        return TrajectorySet(
            data=out,
            t0=t0,
            dt=dt * rs,
            meta={"mode_indices": list(record_modes)},
        )

    @staticmethod
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
                "No PSD was produced because the trajectory ensemble is not "
                "stationary within the requested observation window."
            )

    @staticmethod
    def _draw_standard_normal_into(backend: BackendBase, rng: Any, out: Any) -> Any:
        """Fill a cached noise array, including stable grouped RNG slices."""
        if not isinstance(rng, _GroupedRNG):
            fill = getattr(backend, "randn_into", None)
            if callable(fill):
                return fill(rng, out)
            out[...] = backend.randn(rng, out.shape, dtype=out.dtype)
            return out

        trajectory_axis = 1 if out.ndim == 3 else 0
        expected = len(rng.handles) * rng.group_size
        if int(out.shape[trajectory_axis]) != expected:
            raise ValueError("grouped RNG shape does not match the fused scan tile")
        fill = getattr(backend, "randn_into", None)
        for index, handle in enumerate(rng.handles):
            selection = [slice(None)] * out.ndim
            start = index * rng.group_size
            selection[trajectory_axis] = slice(start, start + rng.group_size)
            target = out[tuple(selection)]
            if callable(fill):
                fill(handle, target)
            else:
                target[...] = backend.randn(handle, target.shape, dtype=target.dtype)
        return out
