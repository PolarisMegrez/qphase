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
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy
from qphase.core.dataset import DatasetResultProtocol
from qphase.core.protocols import EngineBase, EngineManifest, ResultProtocol
from qphase.data import concat_datasets, stack_datasets

from qphase_sde.contracts.bundle import SDEProvenance
from qphase_sde.integrator.base import Integrator
from qphase_sde.model import model_params
from qphase_sde.planning import (
    SDEExecutionPlan,
    build_execution_plan,
    resolve_time_grid,
    typed_batch_scope,
    typed_scan_scope,
)
from qphase_sde.products import add_scan_parameter_coordinates
from qphase_sde.result import SDEDataBundle, fingerprint_text, trajectory_product
from qphase_sde.runtime import simulation_loop
from qphase_sde.runtime.graph_surgery import (
    _batch_scope_sub_graph,
    _node_source_products,
    _post_batch_graph,
    _post_tile_graph,
    _tile_boundary_outputs,
    _tile_scope_sub_graph,
)
from qphase_sde.runtime.rng import (
    _fused_rng_groups,
    _rng_group_seed,
)
from qphase_sde.state import TrajectorySet

__all__ = ["Engine", "EngineConfig", "TrajectoryDivergenceError"]

log = logging.getLogger(__name__)


TrajectoryDivergenceError = simulation_loop.TrajectoryDivergenceError










@dataclass(frozen=True)
class _SimulatedRun:
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
            return self._run_typed_analyze(data, context=context, graph=graph)

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
                return self._run_scan_tiled_typed(
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
            typed = self._run_typed_analyze(source, context=context, graph=graph)
            return self._merge_simulation_trajectory(source, typed)
        if plan.trajectory_batch_count > 1:
            return self._run_trajectory_batched_typed(
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
        typed = self._run_typed_analyze(source, context=context, graph=graph)
        return self._merge_simulation_trajectory(source, typed)

    def _run_typed_analyze(
        self,
        source: DatasetResultProtocol,
        *,
        context: Any | None = None,
        completed_nodes: Mapping[str, Mapping[str, Any]] | None = None,
        graph: Any | None = None,
    ) -> SDEDataBundle:
        """Run the analyser set through the compiler-frozen ProductGraph.

        ``graph`` overrides the compiler-frozen graph for tile-phase
        execution of a scan: the engine runs the induced tile-scope
        sub-graph per tile and defers final-scope nodes to one post-merge
        execution of the full graph.
        """
        analysers = self._normalised_analysers()
        if not analysers:
            raise RuntimeError("typed analyser execution requires analysers")
        from qphase.core import ProductGraphExecutor

        source_products = getattr(source, "products", None)
        if not isinstance(source_products, dict):
            source_products = dict(source.data)
        if not source.shape:
            trajectories = source_products.get("trajectories")
            if (
                trajectories is not None
                and "scan" in {axis.name for axis in trajectories.axes}
                and trajectories.axis("scan").size == 1
            ):
                source_products = dict(source_products)
                source_products["trajectories"] = trajectories.point_view(scan=0)
        if graph is None:
            graph = (
                getattr(context, "product_graph", None)
                if context is not None
                else self._product_graph
            )
        if graph is None:
            raise RuntimeError(
                "typed analyser execution requires the compiler-resolved "
                "ProductGraph in ExecutionContext"
            )
        node_names = {
            node.node_id: node.node_id.removeprefix("analyser.") for node in graph.nodes
        }
        for name in node_names.values():
            analyser = analysers.get(name)
            if analyser is None:
                raise RuntimeError(
                    f"compiled ProductGraph references unknown analyser {name!r}"
                )
        backend = (
            self._required_backend() if self._default_backend is not None else None
        )
        if backend is None:
            from qphase.backend.numpy_backend import NumpyBackend

            backend = NumpyBackend()
        source_metadata = getattr(source, "metadata", None)
        execution_metadata = (
            dict(source_metadata) if isinstance(source_metadata, dict) else {}
        )
        source_provenance = getattr(source, "provenance", None)
        if isinstance(source_provenance, Mapping):
            provenance_meta = source_provenance.get("meta")
            if isinstance(provenance_meta, Mapping):
                execution_metadata.update(provenance_meta)
        raw_scan_axes = dict(source.axes)
        combine = execution_metadata.get("scan_combine")
        if combine == "zipped":
            scan_axes = {
                name: np.asarray(values).reshape(-1)
                for name, values in raw_scan_axes.items()
            }
        elif raw_scan_axes:
            meshes = np.meshgrid(*raw_scan_axes.values(), indexing="ij")
            scan_axes = {
                name: values.reshape(-1)
                for name, values in zip(raw_scan_axes, meshes, strict=True)
            }
        else:
            scan_axes = {}
        execution_metadata["scan_axes"] = scan_axes
        execution_metadata["scan_shape"] = list(source.shape)
        scan_coordinates = {
            name: np.asarray(values).reshape(-1) for name, values in scan_axes.items()
        }

        def make_runner(node_id: str, analyser: Any, name: str) -> Any:
            source_products = _node_source_products(graph, node_id)
            preserves_scan = bool(source.shape) and any(
                binding.consumer == node_id and binding.product == "trajectories"
                for binding in graph.external_inputs
            )

            def runner(node: Any, inputs: Any) -> Any:
                del node
                node_metadata = dict(execution_metadata)
                if source_products:
                    node_metadata["source_products"] = dict(source_products)
                outputs = analyser.execute(
                    inputs,
                    backend=backend,
                    label=str(name),
                    metadata=node_metadata,
                    context=context,
                )
                cancellation = getattr(context, "cancellation", None)
                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                # Downstream consumers may read the scan parameter coordinates
                # (e.g. allan_scaling reads parameter.<scan_param>), so attach
                # them at production time, not only to the terminal products.
                normalized = {}
                for product_name, product in outputs.items():
                    if preserves_scan and "scan" not in {
                        axis.name for axis in product.axes
                    }:
                        product = stack_datasets([product], axis="scan")
                    normalized[product_name] = add_scan_parameter_coordinates(
                        product, scan_coordinates
                    )
                return normalized

            return runner

        runners = {
            node_id: make_runner(node_id, analysers[name], name)
            for node_id, name in node_names.items()
        }
        node_products = ProductGraphExecutor(graph, runners).run(
            {"input": source_products},
            context=context,
            completed_nodes=completed_nodes,
        )
        products = {
            product_name: product
            for node_products_for_node in node_products.values()
            for product_name, product in node_products_for_node.items()
        }
        products = {
            name: add_scan_parameter_coordinates(product, scan_coordinates)
            for name, product in products.items()
        }
        result_meta = dict(execution_metadata)
        result_meta.pop("scan_axes", None)
        result_meta.pop("scan_shape", None)
        result_meta.update({"mode": "analyze", "execution": "product_graph"})
        return SDEDataBundle(
            products,
            self._job_provenance(),
            scan_axes=dict(source.axes),
            scan_shape=source.shape,
            meta=result_meta,
        )

    def _run_scan_tiled_typed(
        self,
        data: Any | None,
        *,
        adapter: Any,
        plan: SDEExecutionPlan,
        reporter: Any | None,
        context: Any,
    ) -> SDEDataBundle:
        """Run a resource-bounded scan through the frozen typed graph.

        The graph splits into tile-scope nodes (executed per scan tile) and
        final-scope nodes (deferred to one execution on the merged full-scan
        products), so cross-scan analysers never execute on partial tiles.
        """
        graph = getattr(context, "product_graph", None)
        if graph is None:
            graph = self._product_graph
        if graph is None:
            raise RuntimeError("typed scan tiling requires a ProductGraph")
        tile_scope, final_scope = typed_scan_scope(graph, self._normalised_analysers())
        tile_graph = _tile_scope_sub_graph(graph, tile_scope) if final_scope else graph
        assert context.parameter_grid is not None
        grid = context.parameter_grid
        flat_coordinates = grid.parameter_arrays(flatten=True)

        if plan.trajectory_batch_count > 1:
            tile_bundles: list[SDEDataBundle] = []
            for tile_index, start in enumerate(
                range(0, plan.scan_size, plan.scan_tile_size)
            ):
                stop = min(start + plan.scan_tile_size, plan.scan_size)
                if reporter is not None:
                    reporter.status(
                        f"SDE typed scan tile {tile_index + 1}/{plan.tile_count} "
                        f"(points {start + 1}-{stop})",
                        stage="sampling",
                    )
                tile_bundles.append(
                    self._run_trajectory_batched_typed(
                        data,
                        adapter=adapter,
                        plan=plan,
                        reporter=reporter,
                        context=context,
                        scan_start=start,
                        scan_stop=stop,
                        master_seed=adapter.master_seed,
                        observers=self._normalised_observers(),
                        graph=tile_graph,
                    )
                )
            products = self._concat_scan_products(
                [bundle.products for bundle in tile_bundles]
            )
            observer_payloads: dict[str, list[Any]] = {}
            for bundle in tile_bundles:
                payloads = bundle.metadata.get("observer_payloads", {})
                if not isinstance(payloads, dict):
                    continue
                for name, value in payloads.items():
                    values = value if isinstance(value, list) else [value]
                    observer_payloads.setdefault(name, []).extend(values)
            products = {
                name: add_scan_parameter_coordinates(product, flat_coordinates)
                for name, product in products.items()
            }
            if final_scope:
                products = self._run_scan_final_phase(
                    graph=graph,
                    context=context,
                    tile_scope=tile_scope,
                    merged_products=products,
                    grid=grid,
                )
            return SDEDataBundle(
                products,
                self._job_provenance(grid=grid),
                scan_axes=dict(grid.axes),
                scan_shape=grid.shape,
                meta={
                    "mode": "simulate_analyze",
                    "execution": "product_graph",
                    "execution_plan": plan.to_dict(),
                    "observer_payloads": observer_payloads,
                },
            )
        tile_products: list[dict[str, Any]] = []
        tile_trajectories: list[Any] = []
        streamed_observer_payloads: dict[str, list[Any]] = {}
        total_work = plan.scan_size * plan.n_traj_per_point * plan.steps
        for tile_index, start in enumerate(
            range(0, plan.scan_size, plan.scan_tile_size)
        ):
            stop = min(start + plan.scan_tile_size, plan.scan_size)
            point_count = stop - start
            if reporter is not None:
                reporter.status(
                    f"SDE typed scan tile {tile_index + 1}/{plan.tile_count} "
                    f"(points {start + 1}-{stop})",
                    stage="sampling",
                )
            group_seeds, group_counts = _fused_rng_groups(
                adapter.master_seed,
                start,
                stop,
                plan.logical_rng_group_size,
                adapter.base_n_traj,
            )
            with adapter.tile(start, stop):
                tile_outcome = self._run_simulate(
                    data,
                    reporter=reporter,
                    context=context,
                    scan_tile=adapter.scan_tile,
                    rng_group_seeds=group_seeds,
                    rng_group_size=plan.logical_rng_group_size,
                    rng_group_counts=group_counts,
                    progress_offset=start * plan.n_traj_per_point * plan.steps,
                    progress_scale=point_count * plan.n_traj_per_point,
                    progress_total=total_work,
                    progress_label=f"tile {tile_index + 1}/{plan.tile_count}",
                )
            for name, payload in tile_outcome.observer_payloads.items():
                values = payload if isinstance(payload, list) else [payload]
                streamed_observer_payloads.setdefault(name, []).extend(values)
            tile_scan_axes = {
                name: np.asarray(values)[start:stop]
                for name, values in flat_coordinates.items()
            }
            tile_source = SDEDataBundle(
                {
                    "trajectories": trajectory_product(
                        tile_outcome.trajectory,
                        scan_size=point_count,
                        n_traj_per_point=plan.n_traj_per_point,
                    )
                },
                self._job_provenance(grid=grid),
                scan_axes=tile_scan_axes,
                scan_shape=(point_count,),
                meta={"scan_combine": "zipped"},
            )
            analysed_tile = self._run_typed_analyze(
                tile_source, context=context, graph=tile_graph
            )
            tile_products.append(analysed_tile.products)
            if self.config is not None and self.config.keep_traj:
                tile_trajectories.append(tile_source.products["trajectories"])
            self._release_backend_pool()

        products = self._concat_scan_products(tile_products)
        products = {
            name: add_scan_parameter_coordinates(product, flat_coordinates)
            for name, product in products.items()
        }
        if final_scope:
            products = self._run_scan_final_phase(
                graph=graph,
                context=context,
                tile_scope=tile_scope,
                merged_products=products,
                grid=grid,
            )
        if tile_trajectories:
            products["trajectories"] = concat_datasets(tile_trajectories, axis="scan")
        return SDEDataBundle(
            products,
            self._job_provenance(grid=grid),
            scan_axes=dict(grid.axes),
            scan_shape=grid.shape,
            meta={
                "mode": "simulate_analyze",
                "execution": "product_graph",
                "execution_plan": plan.to_dict(),
                "observer_payloads": streamed_observer_payloads,
            },
        )

    def _run_scan_final_phase(
        self,
        *,
        graph: Any,
        context: Any,
        tile_scope: frozenset[str],
        merged_products: dict[str, Any],
        grid: Any,
    ) -> dict[str, Any]:
        """Run final-scope nodes once on the merged full-scan products.

        The tile-scope nodes re-enter the full graph as completed nodes; the
        executor then runs every final-scope node a single time with complete
        inputs and returns the graph's published terminal products.
        """
        boundary = _tile_boundary_outputs(graph, tile_scope)
        completed_nodes = {
            node_id: {name: merged_products[name] for name in names}
            for node_id, names in boundary.items()
        }
        final_graph = _post_tile_graph(graph, tile_scope, boundary)
        final_source = SDEDataBundle(
            dict(merged_products),
            self._job_provenance(grid=grid),
            scan_axes=dict(grid.axes),
            scan_shape=grid.shape,
            meta={"scan_combine": grid.combine},
        )
        analysed = self._run_typed_analyze(
            final_source,
            context=context,
            completed_nodes=completed_nodes,
            graph=final_graph,
        )
        return analysed.products

    @staticmethod
    def _concat_scan_products(
        groups: list[dict[str, Any]],
        *,
        stack: bool = False,
    ) -> dict[str, Any]:
        if not groups:
            raise ValueError("cannot concatenate an empty product collection")
        if len(groups) == 1:
            if not stack:
                return dict(groups[0])
            return {
                name: stack_datasets([product], axis="scan")
                for name, product in groups[0].items()
            }
        names = set(groups[0])
        if any(set(group) != names for group in groups[1:]):
            raise ValueError("scan tiles produced different product sets")
        return {
            name: (
                stack_datasets([group[name] for group in groups], axis="scan")
                if stack
                else concat_datasets([group[name] for group in groups], axis="scan")
            )
            for name in names
        }

    def _run_trajectory_batched_typed(
        self,
        data: Any | None,
        *,
        adapter: Any | None = None,
        plan: SDEExecutionPlan,
        reporter: Any | None,
        context: Any,
        scan_start: int = 0,
        scan_stop: int = 1,
        master_seed: int | None,
        observers: dict[str, Any],
        graph: Any | None = None,
    ) -> SDEDataBundle:
        """Run fused scan-tile trajectory batches through the frozen graph."""
        assert self.config is not None
        point_count = scan_stop - scan_start
        if point_count < 1:
            raise ValueError("typed trajectory batch requires a non-empty scan tile")
        analysers = self._normalised_analysers()
        if graph is None:
            graph = getattr(context, "product_graph", None)
        if graph is None:
            raise RuntimeError("typed trajectory batching requires a ProductGraph")
        graph_nodes = {node.node_id: node for node in graph.nodes}
        node_ids = {
            node.node_id.removeprefix("analyser."): node.node_id for node in graph.nodes
        }
        batch_local, reducer_frontier = typed_batch_scope(
            graph, analysers, validate=True
        )
        transient_nodes = batch_local - reducer_frontier
        batch_graph = _batch_scope_sub_graph(graph, batch_local)
        post_graph = _post_batch_graph(graph, transient_nodes, reducer_frontier)
        reducer_names = {
            node_id.removeprefix("analyser.") for node_id in reducer_frontier
        }
        reducer_sets = []
        for _ in range(point_count):
            reducers: dict[str, Any] = {}
            for name in reducer_names:
                analyser = analysers[name]
                reducer = analyser.reducer()
                assert reducer is not None
                node = graph_nodes[node_ids[name]]
                labels = [output.name for output in node.outputs]
                if len(labels) != 1:
                    raise RuntimeError(
                        f"typed reducer analyser {name!r} must declare exactly "
                        "one output Dataset"
                    )
                if hasattr(reducer, "label"):
                    reducer.label = labels[0]
                reducers[name] = reducer
            reducer_sets.append(reducers)
        if master_seed is None:
            master_seed = int(
                np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0]
            )
        master_seed = int(master_seed) % (1 << 64)
        observer_batches: dict[str, list[list[Any]]] = {
            name: [[] for _ in range(point_count)] for name in observers
        }
        result_meta: dict[str, Any] = {}
        reducer_states = [
            {
                name: reducer.prepare(None, context)
                for name, reducer in reducers.items()
                if reducer is not None
            }
            for reducers in reducer_sets
        ]

        for batch_index, start in enumerate(
            range(0, plan.n_traj_per_point, plan.trajectory_batch_size)
        ):
            stop = min(start + plan.trajectory_batch_size, plan.n_traj_per_point)
            count = stop - start
            cancellation = getattr(context, "cancellation", None)
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            group_size = plan.logical_rng_group_size
            if start % group_size:
                raise RuntimeError(
                    "trajectory batch boundaries must align with logical RNG groups"
                )
            first_group = start // group_size
            last_group = (stop - 1) // group_size
            batch_group_counts = tuple(
                min(stop, (group + 1) * group_size) - group * group_size
                for group in range(first_group, last_group + 1)
            )
            seeds = tuple(
                _rng_group_seed(master_seed, point, group)
                for point in range(scan_start, scan_stop)
                for group in range(first_group, last_group + 1)
            )
            group_counts = batch_group_counts * point_count
            if reporter is not None:
                reporter.status(
                    f"SDE typed trajectory batch {batch_index + 1}/"
                    f"{plan.trajectory_batch_count}",
                    stage="sampling",
                )
            scan_tile = None
            if adapter is None:
                batch_scope = self._trajectory_batch_scope(
                    start, stop, plan.n_traj_per_point
                )
            else:
                batch_scope = adapter.trajectory_tile(
                    scan_start, scan_stop, start, stop
                )
            with batch_scope:
                if adapter is not None:
                    scan_tile = adapter.scan_tile
                partial = self._run_simulate(
                    data,
                    reporter=reporter,
                    context=context,
                    scan_tile=scan_tile,
                    rng_group_seeds=seeds,
                    rng_group_size=group_size,
                    rng_group_counts=group_counts,
                    observers=observers,
                    progress_offset=(
                        scan_start * plan.n_traj_per_point + start * point_count
                    )
                    * plan.steps,
                    progress_scale=count * point_count,
                    progress_total=plan.scan_size * plan.n_traj_per_point * plan.steps,
                    progress_label=f"typed trajectory batch {batch_index + 1}/"
                    f"{plan.trajectory_batch_count}",
                )
            source = SDEDataBundle(
                {
                    "trajectories": trajectory_product(
                        partial.trajectory,
                        scan_size=point_count,
                        n_traj_per_point=count,
                    )
                },
                self._job_provenance(),
                scan_axes={"scan": np.arange(point_count)},
                scan_shape=(point_count,),
                meta={"scan_combine": "zipped"},
            )
            trajectory = source.products["trajectories"]
            from qphase.core import ProductGraphExecutor

            def make_batch_runner(node_id: str) -> Any:
                name = node_id.removeprefix("analyser.")
                source_products = _node_source_products(batch_graph, node_id)

                def runner(node: Any, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
                    del node
                    metadata: dict[str, Any] = {"scan_shape": [point_count]}
                    if source_products:
                        metadata["source_products"] = dict(source_products)
                    return analysers[name].execute(
                        inputs,
                        backend=self._required_backend(),
                        label=name,
                        metadata=metadata,
                        context=context,
                    )

                return runner

            batch_outputs = ProductGraphExecutor(
                batch_graph,
                {
                    node.node_id: make_batch_runner(node.node_id)
                    for node in batch_graph.nodes
                },
            ).run({"input": {"trajectories": trajectory}}, context=context)
            for point_offset, reducers in enumerate(reducer_sets):
                for owner, reducer in reducers.items():
                    products = batch_outputs[node_ids[owner]]
                    for product in products.values():
                        point_product = (
                            product.point_view(scan=point_offset)
                            if "scan" in {axis.name for axis in product.axes}
                            else product
                        )
                        reducer.consume(
                            point_product,
                            reducer_states[point_offset][owner],
                            reporter,
                        )
            for name in observers:
                payload = self._analysis_to_host(partial.observer_payloads.get(name))
                if payload is None:
                    raise RuntimeError(
                        f"observer {name!r} returned no typed batch payload"
                    )
                point_payloads = payload if isinstance(payload, list) else [payload]
                if len(point_payloads) != point_count:
                    raise RuntimeError(
                        f"observer {name!r} returned {len(point_payloads)} scan "
                        f"payloads for a {point_count}-point tile"
                    )
                for point_offset, point_payload in enumerate(point_payloads):
                    observer_batches[name][point_offset].append(point_payload)
            if not result_meta:
                result_meta.update(partial.meta)
            del partial, source, batch_outputs
            self._release_backend_pool()

        result_meta.update(
            {
                "params": dict(self._required_model().params),
                "rng_master_seed": master_seed,
                "rng_strategy": plan.rng_strategy,
                "drop_trajectory_reason": "online_trajectory_aggregation",
                "trajectory_batch_size": plan.trajectory_batch_size,
                "trajectory_batch_count": plan.trajectory_batch_count,
                "execution_plan": plan.to_dict(),
            }
        )
        point_products: list[dict[str, Any]] = []
        for reducers, states in zip(reducer_sets, reducer_states, strict=True):
            finalized: dict[str, Any] = {}
            for name, reducer in reducers.items():
                if reducer is None:
                    raise RuntimeError(f"typed analyser {name!r} has no reducer")
                finalized.update(reducer.finalize(states[name]))
            point_products.append(finalized)

        is_scan = adapter is not None
        completed_products = (
            self._concat_scan_products(point_products, stack=True)
            if is_scan or point_count > 1
            else point_products[0]
        )
        tile_coordinates: dict[str, np.ndarray] = {}
        if is_scan:
            grid = getattr(context, "parameter_grid", None)
            if grid is not None:
                # Reducer-completed frontier products must carry the real
                # flat scan coordinates of this tile before downstream graph
                # nodes consume them.
                tile_coordinates = {
                    name: np.asarray(values)[scan_start:scan_stop]
                    for name, values in grid.parameter_arrays(flatten=True).items()
                }
                completed_products = {
                    name: add_scan_parameter_coordinates(product, tile_coordinates)
                    for name, product in completed_products.items()
                }
        completed_nodes = {
            node_ids[name]: {
                product_name: completed_products[product_name]
                for product_name in (
                    output.name for output in graph_nodes[node_ids[name]].outputs
                )
            }
            for name in reducer_names
        }
        result_meta["observer_payloads"] = {
            name: [
                observers[name].merge_payloads(observer_batches[name][point_offset])
                for point_offset in range(point_count)
            ]
            for name in observers
        }
        if not is_scan and point_count == 1:
            result_meta["observer_payloads"] = {
                name: payloads[0]
                for name, payloads in result_meta["observer_payloads"].items()
            }
        source = SDEDataBundle(
            completed_products,
            self._job_provenance(),
            scan_axes=tile_coordinates,
            scan_shape=(point_count,) if is_scan else (),
            meta={**result_meta, "scan_combine": "zipped"},
        )
        analysed = self._run_typed_analyze(
            source,
            context=context,
            completed_nodes=completed_nodes,
            graph=post_graph,
        )
        return analysed

    def _merge_simulation_trajectory(
        self, source: SDEDataBundle, analysed: ResultProtocol
    ) -> SDEDataBundle:
        """Add an explicitly retained trajectory to typed analyser products."""
        products = dict(getattr(analysed, "products", {}))
        if self.config is not None and self.config.keep_traj:
            products["trajectories"] = source.products["trajectories"]
        result_meta = dict(source.metadata)
        result_meta.pop("provenance", None)
        result_meta.update({"execution": "product_graph", "mode": "simulate_analyze"})
        return SDEDataBundle(
            products,
            source.provenance,
            scan_axes=source.axes,
            scan_shape=source.shape,
            meta=result_meta,
        )

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
        self,
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
    ) -> _SimulatedRun:
        """Execute one SDE integration pass; analysers never run here.

        The trajectory always comes back retained: whether it is published
        as a product is decided by the typed graph/merge layer above, never
        inside the integration pass.
        """
        assert self.config is not None

        if observers is None:
            observers = self._normalised_observers()

        model = self._required_model()

        time_grid = resolve_time_grid(
            t0=self.config.t0,
            t1=self.config.t1,
            dt=self.config.dt,
        )
        time_cfg = {
            "t0": time_grid.integration_t0,
            "observation_start": time_grid.observation_t0,
            "dt": self.config.dt,
            "steps": time_grid.total_steps,
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

        traj_set: TrajectorySet = self.run_sde(
            model=model,
            ic=ic,
            time=time_cfg,
            n_traj=self.config.n_traj,
            seed=self.config.seed,
            return_stride=self.config.save_stride,
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
            if int(self.config.n_traj) % n_scan:
                raise ValueError(
                    f"Observer ensemble has {self.config.n_traj} trajectories, "
                    f"not divisible by scan count {n_scan}"
                )
            group_size = int(self.config.n_traj) // n_scan
            observer_instances = observers or self._normalised_observers()
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

        # Ensure model parameters are in metadata
        meta: dict[str, Any] = {}
        if hasattr(model, "config") and hasattr(model.config, "model_dump"):
            meta["params"] = model.config.model_dump()
        elif hasattr(model, "parameters"):
            meta["params"] = model.parameters
        elif hasattr(model, "params"):
            meta["params"] = model.params

        return _SimulatedRun(
            trajectory=traj_set,
            meta=meta,
            observer_payloads=observer_results,
        )

    def run_sde(self, *args: Any, **kwargs: Any) -> TrajectorySet:
        """Run one integration pass; see runtime.simulation_loop."""
        return simulation_loop.run_sde(self, *args, **kwargs)




