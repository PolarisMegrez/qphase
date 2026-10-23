"""qphase_sde: SDE Engine
---------------------------------------------------------
The SDE engine integrates trajectories and executes the compiler-frozen
ProductGraph over typed data products. Simulate jobs always produce a
``trajectories`` time-series product; analyser outputs are typed Datasets
assembled into an :class:`~qphase_sde.result.SDEDataBundle`.

Public API
----------
``Engine`` : Main simulation engine class.
``EngineConfig`` : Configuration model for the engine.
"""

import logging
from contextlib import contextmanager
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy
from qphase.core.dataset import DatasetResultProtocol
from qphase.core.protocols import EngineBase, EngineManifest, ResultProtocol

from qphase_sde.contracts.bundle import SDEProvenance
from qphase_sde.integrator.base import Integrator
from qphase_sde.model import model_params
from qphase_sde.planning import (
    SDEExecutionPlan,
    build_execution_plan,
    resolve_time_grid,
)
from qphase_sde.products import add_scan_parameter_coordinates
from qphase_sde.result import SDEDataBundle, fingerprint_text, trajectory_product
from qphase_sde.runtime import simulation_loop
from qphase_sde.runtime.rng import _fused_rng_groups
from qphase_sde.runtime.typed_execution import (
    _merge_simulation_trajectory,
    _run_scan_tiled_typed,
    _run_trajectory_batched_typed,
    _run_typed_analyze,
)
from qphase_sde.state import TrajectorySet

__all__ = ["Engine", "EngineConfig", "TrajectoryDivergenceError"]

log = logging.getLogger(__name__)


TrajectoryDivergenceError = simulation_loop.TrajectoryDivergenceError


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
        ge=0.0,
        description=(
            "Observation start time. Integration always starts at physical time 0; "
            "samples before t0 are warm-up and are not retained."
        ),
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
    rng_group_size: int = Field(
        64,
        ge=1,
        description=(
            "Logical RNG group size: one random stream serves this many "
            "consecutive trajectories per scan point. Numerical provenance — "
            "physical trajectory batches must be whole multiples of it, so "
            "results are independent of the physical batch size. Different "
            "group sizes draw different but distribution-equivalent "
            "realizations."
        ),
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
    # Conditional task profiles are the single asset-requirement declaration:
    # the compiler selects simulate/analyze/simulate_analyze from the job
    # config instead of consulting legacy required/input plugin sets.
    manifest: ClassVar[EngineManifest] = EngineManifest(
        defaults={"integrator": "euler_maruyama"},
        task_profiles="qphase_sde.contracts.tasks:SDE_TASK_PROFILES",
        profile_selector="qphase_sde.contracts.tasks:select_sde_task_profile",
        product_graph_builder="qphase_sde.graph:compile_product_graph",
        analyser_config_expander="qphase_sde.analyser_config:expand_analyser_config",
    )

    def __init__(
        self,
        config: EngineConfig | None = None,
        plugins: dict[str, Any] | None = None,
        *,
        product_graph: Any | None = None,
    ):
        """Initialize Engine with scheduler-resolved plugin instances.

        Parameters
        ----------
        config : EngineConfig, optional
            Configuration object (injected by Registry)
        plugins : dict, optional
            Plugin instances keyed by plugin class (injected by Registry).
            ``backend`` and ``integrator`` must each resolve to a single
            instance; ``analyser``/``observer`` may be name→instance dicts.
        product_graph : ProductGraph, optional
            Compiler-frozen analyser product graph of this job.

        """
        self.config = config
        self.plugins = plugins or {}
        self._product_graph = product_graph

        # The scheduler resolves the task profile's required backend and
        # integrator to exactly one instance each; an ambiguous mapping is a
        # wiring error, not something to resolve by preference.
        backend = self.plugins.get("backend")
        if isinstance(backend, dict):
            raise RuntimeError(
                "SDE engine requires exactly one backend plugin instance, got "
                f"a mapping with {sorted(backend)}"
            )
        self._default_backend = backend

        integrator = self.plugins.get("integrator")
        if isinstance(integrator, dict):
            raise RuntimeError(
                "SDE engine requires exactly one integrator plugin instance, "
                f"got a mapping with {sorted(integrator)}"
            )
        self._default_integrator = integrator

    def run(
        self, data: Any | None = None, *, context: Any | None = None
    ) -> ResultProtocol:
        """Execute the engine (Plugin Protocol)."""
        if not self.config:
            raise RuntimeError("Engine not configured.")

        reporter = context.progress if context is not None else None
        graph = getattr(context, "product_graph", None) if context is not None else None
        if graph is None:
            graph = self._product_graph
        analysers = self._normalised_analysers()
        if graph is None and analysers:
            raise RuntimeError(
                "SDE job configures analysers but has no compiler-frozen "
                "ProductGraph; run the job through the qphase compiler/"
                "scheduler so the analyser graph is validated up front"
            )

        if getattr(self.config, "mode", "simulate") == "analyze":
            if not isinstance(data, DatasetResultProtocol):
                raise TypeError(
                    "analyze mode requires a typed DatasetResultProtocol input"
                )
            return _run_typed_analyze(self, data, context=context, graph=graph)

        grid = context.parameter_grid if context is not None else None
        model = self._required_model()
        observers = self._normalised_observers()
        backend = self._required_backend()
        integrator = self._required_integrator()
        plan = build_execution_plan(
            config=self.config,
            grid=grid,
            model=model,
            backend=backend,
            integrator=integrator,
            analysers=analysers,
            observers=observers,
            resources=getattr(context, "resources", None),
            product_graph=graph,
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
            from qphase_sde.runtime.scan import SDEParameterGridAdapter

            adapter = SDEParameterGridAdapter(self, grid)
            if plan.stream_analysis and (
                plan.tile_count > 1 or plan.trajectory_batch_count > 1
            ):
                return _run_scan_tiled_typed(
                    self,
                    data,
                    adapter=adapter,
                    plan=plan,
                    reporter=reporter,
                    context=context,
                )
            group_seeds, group_counts = _fused_rng_groups(
                adapter.master_seed,
                0,
                adapter.grid.size,
                plan.logical_rng_group_size,
                adapter.base_n_traj,
            )
            with adapter:
                outcome = self._run_simulate(
                    data,
                    reporter=reporter,
                    context=context,
                    scan_tile=adapter.scan_tile,
                    rng_group_seeds=group_seeds,
                    rng_group_size=plan.logical_rng_group_size,
                    rng_group_counts=group_counts,
                )
            meta = dict(outcome.meta)
            meta.update(
                {
                    "scan_shape": grid.shape,
                    "scan_combine": grid.combine,
                    "execution_plan": plan_payload,
                }
            )
            if outcome.observer_payloads:
                meta["observer_payloads"] = outcome.observer_payloads
            source = SDEDataBundle(
                {
                    "trajectories": add_scan_parameter_coordinates(
                        trajectory_product(
                            outcome.trajectory,
                            scan_size=grid.size,
                            n_traj_per_point=adapter.base_n_traj,
                        ),
                        grid.parameter_arrays(flatten=True),
                    )
                },
                self._job_provenance(grid=grid),
                scan_axes=dict(grid.axes),
                scan_shape=grid.shape,
                meta=meta,
                n_traj_per_point=adapter.base_n_traj,
            )
            if graph is None:
                return source
            typed = _run_typed_analyze(self, source, context=context, graph=graph)
            return _merge_simulation_trajectory(self, source, typed)
        if plan.trajectory_batch_count > 1:
            return _run_trajectory_batched_typed(
                self,
                data,
                plan=plan,
                reporter=reporter,
                context=context,
                master_seed=self.config.seed,
                observers=observers,
                graph=graph,
            )
        master_seed = self.config.seed
        if master_seed is None:
            master_seed = int(
                np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0]
            )
        group_seeds, group_counts = _fused_rng_groups(
            int(master_seed) % (1 << 64),
            0,
            1,
            plan.logical_rng_group_size,
            plan.n_traj_per_point,
        )
        outcome = self._run_simulate(
            data,
            reporter=reporter,
            context=context,
            rng_group_seeds=group_seeds,
            rng_group_size=plan.logical_rng_group_size,
            rng_group_counts=group_counts,
            observers=observers,
        )
        meta = dict(outcome.meta)
        meta.setdefault("execution_plan", plan_payload)
        if outcome.observer_payloads:
            meta["observer_payloads"] = outcome.observer_payloads
        source = SDEDataBundle(
            {
                "trajectories": trajectory_product(
                    outcome.trajectory,
                    scan_size=None,
                    n_traj_per_point=plan.n_traj_per_point,
                )
            },
            self._job_provenance(),
            meta=meta,
        )
        if graph is None:
            return source
        typed = _run_typed_analyze(self, source, context=context, graph=graph)
        return _merge_simulation_trajectory(self, source, typed)

    def _job_provenance(self, *, grid: Any | None = None) -> SDEProvenance:
        """Build the job-level provenance record from config and plugins."""
        config = self.config
        scan_grid: dict[str, list[float]] = {}
        if grid is not None:
            for name, values in grid.axes.items():
                try:
                    scan_grid[str(name)] = [float(value) for value in values]
                except (TypeError, ValueError):
                    continue
        warmup_samples = 0
        if config is not None:
            try:
                time_grid = resolve_time_grid(t0=config.t0, t1=config.t1, dt=config.dt)
                warmup_samples = int(time_grid.warmup_steps)
            except Exception:  # noqa: BLE001 - provenance must not break runs
                warmup_samples = 0
        return SDEProvenance(
            t0=float(getattr(config, "t0", 0.0)),
            dt=getattr(config, "dt", None),
            warmup_samples=warmup_samples,
            master_seed=getattr(config, "seed", None),
            model_fingerprint=fingerprint_text(self.plugins.get("model")),
            integrator_fingerprint=fingerprint_text(self._default_integrator),
            scan_grid=scan_grid,
        )

    @staticmethod
    def _execution_plan_summary(plan: SDEExecutionPlan) -> str:
        parts = [
            f"SDE plan: {plan.tile_count} scan tile(s)",
            f"{plan.trajectory_batch_size} trajectories/batch",
        ]
        if plan.trajectory_batch_count > 1:
            parts.append(f"{plan.trajectory_batch_count} batches/point")
        if plan.deferred_analysers:
            parts.append(
                "deferred to post-merge execution: "
                + ", ".join(plan.deferred_analysers)
            )
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

    def _normalised_observers(self) -> dict[str, Any]:
        observers = self.plugins.get("observer")
        if not observers:
            return {}
        if isinstance(observers, dict):
            return dict(observers)
        return {getattr(observers, "name", "observer"): observers}

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
        try:
            with model_params(model, sliced_params):
                yield
        finally:
            self.config.n_traj = original_n_traj
            self.config.ic = original_ic

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


    def _run_simulate(
        self, *args: Any, **kwargs: Any
    ) -> simulation_loop.SimulatedRun:
        """Run one integration pass; see runtime.simulation_loop."""
        return simulation_loop.run_simulate(self, *args, **kwargs)

    def run_sde(self, *args: Any, **kwargs: Any) -> TrajectorySet:
        """Run one integration pass; see runtime.simulation_loop."""
        return simulation_loop.run_sde(self, *args, **kwargs)


