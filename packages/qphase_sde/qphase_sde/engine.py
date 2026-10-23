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
import time as _time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy, get_xp
from qphase.core.dataset import DatasetResultProtocol
from qphase.core.protocols import EngineBase, EngineManifest, ResultProtocol
from qphase.data import concat_datasets, stack_datasets

from qphase_sde.contracts.bundle import SDEProvenance
from qphase_sde.integrator.base import Integrator
from qphase_sde.model import NoiseSpec, SDEModel
from qphase_sde.observer.base import (
    ObserverContext,
    ObserverDecision,
    ObserverTriggeredError,
)
from qphase_sde.planning import (
    SDEExecutionPlan,
    _typed_batch_scope,
    _typed_scan_scope,
    build_execution_plan,
    resolve_time_grid,
)
from qphase_sde.products import add_scan_parameter_coordinates
from qphase_sde.result import SDEDataBundle, _fingerprint_text, _trajectory_product
from qphase_sde.runtime.buffers import SDEBufferCache
from qphase_sde.state import TrajectorySet

__all__ = ["Engine", "EngineConfig", "TrajectoryDivergenceError"]

log = logging.getLogger(__name__)


class TrajectoryDivergenceError(RuntimeError):
    """Raised when an SDE trajectory leaves a configured state-norm bound."""


@dataclass(frozen=True)
class _GroupedRNG:
    """Independent RNG handles, each driving a fixed run of trajectories.

    ``counts`` gives the per-handle trajectory counts when groups are not
    uniform (a short tail group ends a scan point's ensemble); ``None``
    means every handle drives ``group_size`` trajectories.
    """

    handles: tuple[Any, ...]
    group_size: int
    counts: tuple[int, ...] | None = None


def _rng_group_seed(master_seed: int, point_index: int, group_index: int) -> int:
    """Derive one logical-group stream seed from (master, scan point, group)."""
    sequence = np.random.SeedSequence(
        [int(master_seed), int(point_index), int(group_index)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _rng_group_counts(group_size: int, n_traj: int) -> tuple[int, ...]:
    """Per-group trajectory counts covering ``n_traj`` rows, tail group last."""
    group_count = -(-n_traj // group_size)
    return (group_size,) * (group_count - 1) + (
        n_traj - (group_count - 1) * group_size,
    )


def _fused_rng_groups(
    master_seed: int,
    point_start: int,
    point_stop: int,
    group_size: int,
    n_traj_per_point: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Point-major (seeds, counts) for a fused span of scan points.

    Every trajectory's random stream derives from (master seed, absolute
    scan point, logical group index), so fused, tiled, and
    trajectory-batched executions of the same configuration draw identical
    streams.
    """
    point_counts = _rng_group_counts(group_size, n_traj_per_point)
    seeds: list[int] = []
    counts: list[int] = []
    for point in range(point_start, point_stop):
        seeds.extend(
            _rng_group_seed(master_seed, point, group)
            for group in range(len(point_counts))
        )
        counts.extend(point_counts)
    return tuple(seeds), tuple(counts)


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


def _tile_scope_sub_graph(graph: Any, tile_scope: frozenset[str]) -> Any:
    """Build the induced tile-phase sub-graph over the tile-scope nodes.

    Tile-scope is ancestor-closed, so every edge into a tile-scope consumer
    stays internal and the remaining external bindings keep validating.
    Outputs crossing the tile boundary become terminal in this induced graph;
    tile-internal transient outputs keep their original liveness.
    """
    from qphase.data import ProductGraph

    nodes = [node for node in graph.nodes if node.node_id in tile_scope]
    edges = [edge for edge in graph.edges if edge.consumer in tile_scope]
    external = [
        binding for binding in graph.external_inputs if binding.consumer in tile_scope
    ]
    return ProductGraph(nodes=nodes, edges=edges, external_inputs=external)


def _tile_boundary_outputs(
    graph: Any, tile_scope: frozenset[str]
) -> dict[str, set[str]]:
    """Return products that must cross the tile merge boundary."""
    boundary = graph.terminal_outputs()
    for node in graph.nodes:
        boundary[node.node_id].update(node.publish_outputs)
    for edge in graph.edges:
        if edge.producer in tile_scope and edge.consumer not in tile_scope:
            boundary[edge.producer].add(edge.product)
    return {
        node_id: names
        for node_id, names in boundary.items()
        if node_id in tile_scope and names
    }


def _post_tile_graph(
    graph: Any,
    tile_scope: frozenset[str],
    boundary: Mapping[str, set[str]],
) -> Any:
    """Build the final graph with merged tile-boundary nodes completed."""
    from qphase.data import ProductGraph

    retained = set(boundary)
    nodes = []
    for node in graph.nodes:
        if node.node_id not in tile_scope:
            nodes.append(node)
            continue
        if node.node_id in retained:
            names = boundary[node.node_id]
            nodes.append(
                node.model_copy(
                    update={
                        "requirements": [],
                        "outputs": [
                            output for output in node.outputs if output.name in names
                        ],
                        "publish_outputs": [
                            name for name in node.publish_outputs if name in names
                        ],
                    }
                )
            )
    return ProductGraph(
        nodes=nodes,
        edges=[
            edge
            for edge in graph.edges
            if edge.producer in retained and edge.consumer not in tile_scope
            or edge.producer not in tile_scope and edge.consumer not in tile_scope
        ],
        external_inputs=[
            binding
            for binding in graph.external_inputs
            if binding.consumer not in tile_scope
        ],
    )


def _batch_scope_sub_graph(graph: Any, batch_scope: frozenset[str]) -> Any:
    """Build the executable per-batch graph over transient and frontier nodes."""
    from qphase.data import ProductGraph

    return ProductGraph(
        nodes=[node for node in graph.nodes if node.node_id in batch_scope],
        edges=[
            edge
            for edge in graph.edges
            if edge.producer in batch_scope and edge.consumer in batch_scope
        ],
        external_inputs=[
            binding
            for binding in graph.external_inputs
            if binding.consumer in batch_scope
        ],
    )


def _node_source_products(graph: Any, node_id: str) -> dict[str, str]:
    """Resolve every requirement's source identity from the frozen graph.

    External bindings record ``"<source>.<product>"`` (e.g.
    ``input.trajectories``); internal edges carry the upstream product name.
    The mapping is requirement-keyed and name-agnostic, so normal, batched
    and tiled runners hand every typed analyser the same provenance.
    """
    sources: dict[str, str] = {}
    for binding in graph.external_inputs:
        if binding.consumer == node_id:
            sources[binding.requirement] = f"{binding.source}.{binding.product}"
    for edge in graph.edges:
        if edge.consumer == node_id:
            sources[edge.requirement] = edge.product
    return sources


def _post_batch_graph(
    graph: Any,
    transient: frozenset[str],
    frontier: frozenset[str],
) -> Any:
    """Remove transient nodes and close completed reducer requirements."""
    from qphase.data import ProductGraph

    return ProductGraph(
        nodes=[
            node.model_copy(update={"requirements": []})
            if node.node_id in frontier
            else node
            for node in graph.nodes
            if node.node_id not in transient
        ],
        edges=[
            edge
            for edge in graph.edges
            if edge.producer not in transient and edge.consumer not in frontier
        ],
        external_inputs=[
            binding
            for binding in graph.external_inputs
            if binding.consumer not in transient and binding.consumer not in frontier
        ],
    )


class _AdaptiveNoiseRNG:
    """Generator-style facade the engine hands to adaptive integrators.

    Adaptive integrators draw noise internally through the ``normal(size=...)``
    API declared by the integrator contract; this facade routes the draw
    through the engine's grouped-stream machinery so per-group provenance is
    preserved without exposing the private ``_GroupedRNG`` container to
    integrator or backend code.
    """

    def __init__(self, backend: Any, rng: Any) -> None:
        self._backend = backend
        self._rng = rng

    def normal(self, size: Any = None) -> Any:
        shape = tuple(size) if size is not None else ()
        out = self._backend.empty(shape, np.float64)
        Engine._draw_standard_normal_into(self._backend, self._rng, out)
        return out


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
                        _trajectory_product(
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
                "trajectories": _trajectory_product(
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
        tile_scope, final_scope = _typed_scan_scope(graph, self._normalised_analysers())
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
                    "trajectories": _trajectory_product(
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
        batch_local, reducer_frontier = _typed_batch_scope(
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
            if adapter is None:
                batch_scope = self._trajectory_batch_scope(
                    start, stop, plan.n_traj_per_point
                )
            else:
                batch_scope = adapter.trajectory_tile(
                    scan_start, scan_stop, start, stop
                )
            with batch_scope:
                partial = self._run_simulate(
                    data,
                    reporter=reporter,
                    context=context,
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
                    "trajectories": _trajectory_product(
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
            model_fingerprint=_fingerprint_text(self.plugins.get("model")),
            integrator_fingerprint=_fingerprint_text(self._default_integrator),
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

    def _run_simulate(
        self,
        data: Any | None,
        *,
        reporter: Any | None = None,
        context: Any | None = None,
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
        n_scan = getattr(self.config, "_batch_scan_count", 1)
        if not isinstance(n_scan, int) or n_scan < 1:
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
        be = backend or self._default_backend
        if be is None:
            raise RuntimeError("No backend provided or configured.")

        integrator = solver or self._default_integrator
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
        if self.config is not None and self.config.record_modes is not None:
            record_modes = tuple(int(mode) for mode in self.config.record_modes)
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
            stop_observer = self._handle_observer_actions(observer_items, y, t, 0)

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
                    self._draw_standard_normal_into(be, rng, raw_noise)
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
                    self._check_state_norm(y, next_t, state_norm_limit)
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
                stop_observer = self._handle_observer_actions(observer_items, y, t, k)
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
                "No downstream analysis was produced because the trajectory "
                "ensemble is not stationary within the requested observation window."
            )

    def _handle_observer_actions(
        self,
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
            raise self._observer_fail_job_error(failures, t, step)
        stops = [
            name
            for name, decision in decisions
            if decision is not None and decision.action == "stop_batch"
        ]
        return stops[0] if stops else None

    @staticmethod
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
        counts = rng.counts
        if counts is None:
            counts = (rng.group_size,) * len(rng.handles)
        if len(counts) != len(rng.handles):
            raise ValueError("grouped RNG counts do not match the handle count")
        expected = sum(counts)
        if int(out.shape[trajectory_axis]) != expected:
            raise ValueError("grouped RNG shape does not match the fused scan tile")
        fill = getattr(backend, "randn_into", None)
        start = 0
        for handle, count in zip(rng.handles, counts, strict=True):
            selection = [slice(None)] * out.ndim
            selection[trajectory_axis] = slice(start, start + count)
            start += count
            target = out[tuple(selection)]
            if callable(fill):
                fill(handle, target)
            else:
                target[...] = backend.randn(handle, target.shape, dtype=target.dtype)
        return out
