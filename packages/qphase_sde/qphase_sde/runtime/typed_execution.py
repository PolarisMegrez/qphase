"""Typed execution: run the frozen ProductGraph over typed products."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from qphase.core.dataset import DatasetResultProtocol
from qphase.core.protocols import ResultProtocol
from qphase.data import concat_datasets, stack_datasets

from qphase_sde.planning import SDEExecutionPlan, typed_batch_scope, typed_scan_scope
from qphase_sde.products import add_scan_parameter_coordinates
from qphase_sde.result import SDEDataBundle, trajectory_product
from qphase_sde.runtime.graph_surgery import (
    _batch_scope_sub_graph,
    _node_source_products,
    _post_batch_graph,
    _post_tile_graph,
    _tile_boundary_outputs,
    _tile_scope_sub_graph,
)
from qphase_sde.runtime.rng import _fused_rng_groups, _rng_group_seed


def _source_artifact_identity(source: Any, metadata: Mapping[str, Any]) -> str | None:
    """Return the artifact id the analysed source is known by, if any.

    A restored SDE bundle declares the id it was published under; a restored
    generic bundle only states it in its metadata. Either way it names the *input*
    of this run, never the result.
    """
    declared = getattr(source, "declared_artifact_id", None)
    if callable(declared):
        value = declared()
        if value:
            return str(value)
    value = metadata.get("artifact_id")
    return None if value is None else str(value)


def _derived_parents(source: Any, source_artifact_id: str | None) -> tuple[str, ...]:
    """Return the parent chain a derived bundle inherits from its source.

    The source's own ancestors come first (deduplicated, order preserved) and the
    source artifact id is appended, so the derived artifact states the whole chain
    it came from and no ancestor is lost.
    """
    chain: list[str] = []
    for value in getattr(source, "parents", ()) or ():
        text = str(value)
        if text and text not in chain:
            chain.append(text)
    if source_artifact_id and source_artifact_id not in chain:
        chain.append(source_artifact_id)
    return tuple(chain)


def _merge_parent_chains(*sources: Any) -> tuple[str, ...]:
    """Return the ordered union of several bundles' parent chains."""
    chain: list[str] = []
    for source in sources:
        if source is None:
            continue
        metadata = getattr(source, "metadata", None)
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        identity = _source_artifact_identity(source, metadata)
        ancestors = tuple(getattr(source, "parents", ()) or ())
        for value in (*ancestors, identity):
            if value is None:
                continue
            text = str(value)
            if text and text not in chain:
                chain.append(text)
    return tuple(chain)


def _run_typed_analyze(
    engine: Any,
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
    analysers = engine._normalised_analysers()
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
            else engine._product_graph
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
        engine._required_backend() if engine._default_backend is not None else None
    )
    if backend is None:
        from qphase.backend.numpy_backend import NumpyBackend
        backend = NumpyBackend()
    source_metadata = getattr(source, "metadata", None)
    execution_metadata = (
        dict(source_metadata) if isinstance(source_metadata, Mapping) else {}
    )
    # The source artifact is an input of this run: record it as such instead of
    # letting a copied ``artifact_id`` look like this result's own identity.
    source_artifact_id = _source_artifact_identity(source, execution_metadata)
    execution_metadata.pop("artifact_id", None)
    if source_artifact_id is not None:
        execution_metadata["input_artifact_id"] = source_artifact_id
    parents = _derived_parents(source, source_artifact_id)
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
        engine._job_provenance(),
        scan_axes=dict(source.axes),
        scan_shape=source.shape,
        meta=result_meta,
        parents=parents,
    )


def _run_scan_tiled_typed(
    engine: Any,
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
        graph = engine._product_graph
    if graph is None:
        raise RuntimeError("typed scan tiling requires a ProductGraph")
    tile_scope, final_scope = typed_scan_scope(graph, engine._normalised_analysers())
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
                _run_trajectory_batched_typed(engine, 
                    data,
                    adapter=adapter,
                    plan=plan,
                    reporter=reporter,
                    context=context,
                    scan_start=start,
                    scan_stop=stop,
                    master_seed=adapter.master_seed,
                    observers=engine._normalised_observers(),
                    graph=tile_graph,
                )
            )
        products = _concat_scan_products(
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
            products = _run_scan_final_phase(engine, 
                graph=graph,
                context=context,
                tile_scope=tile_scope,
                merged_products=products,
                grid=grid,
            )
        return SDEDataBundle(
            products,
            engine._job_provenance(grid=grid),
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
            tile_outcome = engine._run_simulate(
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
            engine._job_provenance(grid=grid),
            scan_axes=tile_scan_axes,
            scan_shape=(point_count,),
            meta={"scan_combine": "zipped"},
        )
        analysed_tile = _run_typed_analyze(engine, 
            tile_source, context=context, graph=tile_graph
        )
        tile_products.append(analysed_tile.products)
        if engine.config is not None and engine.config.keep_traj:
            tile_trajectories.append(tile_source.products["trajectories"])
        engine._release_backend_pool()
    products = _concat_scan_products(tile_products)
    products = {
        name: add_scan_parameter_coordinates(product, flat_coordinates)
        for name, product in products.items()
    }
    if final_scope:
        products = _run_scan_final_phase(engine, 
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
        engine._job_provenance(grid=grid),
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
    engine: Any,
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
        engine._job_provenance(grid=grid),
        scan_axes=dict(grid.axes),
        scan_shape=grid.shape,
        meta={"scan_combine": grid.combine},
    )
    analysed = _run_typed_analyze(engine, 
        final_source,
        context=context,
        completed_nodes=completed_nodes,
        graph=final_graph,
    )
    return analysed.products


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
    engine: Any,
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
    assert engine.config is not None
    point_count = scan_stop - scan_start
    if point_count < 1:
        raise ValueError("typed trajectory batch requires a non-empty scan tile")
    analysers = engine._normalised_analysers()
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
            batch_scope = engine._trajectory_batch_scope(
                start, stop, plan.n_traj_per_point
            )
        else:
            batch_scope = adapter.trajectory_tile(
                scan_start, scan_stop, start, stop
            )
        with batch_scope:
            if adapter is not None:
                scan_tile = adapter.scan_tile
            partial = engine._run_simulate(
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
            engine._job_provenance(),
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
                    backend=engine._required_backend(),
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
            payload = engine._analysis_to_host(partial.observer_payloads.get(name))
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
        engine._release_backend_pool()
    result_meta.update(
        {
            "params": dict(engine._required_model().params),
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
        _concat_scan_products(point_products, stack=True)
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
        engine._job_provenance(),
        scan_axes=tile_coordinates,
        scan_shape=(point_count,) if is_scan else (),
        meta={**result_meta, "scan_combine": "zipped"},
    )
    analysed = _run_typed_analyze(engine, 
        source,
        context=context,
        completed_nodes=completed_nodes,
        graph=post_graph,
    )
    return analysed


def _merge_simulation_trajectory(
    engine: Any, source: SDEDataBundle, analysed: ResultProtocol
) -> SDEDataBundle:
    """Add an explicitly retained trajectory to typed analyser products."""
    products = dict(getattr(analysed, "products", {}))
    if engine.config is not None and engine.config.keep_traj:
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
        parents=_merge_parent_chains(analysed, source),
    )
