"""Assemble an :class:`~..plans.SDEExecutionPlan` for one SDE run.

``build_execution_plan`` combines the resolved time grid, the analyser
requirements and the memory budget into tiles, trajectory batches and the
deferred-analyser frontier; the scope helpers classify a compiled product
graph for tile-phase and batch-phase execution.
"""

from __future__ import annotations

import math
from typing import Any

from qphase.data import DataKind

from ..contracts.quantities import SDEQuantity
from .memory import (
    _MIB,
    _analyzer_capabilities,
    _analyzer_workspace_bytes,
    _backend_name,
    _checked_product,
    _host_memory_budget,
    _memory_budget,
    _minimum_working_set_message,
    _real_dtype,
    _reserve_bytes,
    _tile_capacity,
    _validate_analyser_sampling,
)
from .plans import (
    SDEExecutionPlan,
    SDEMemoryEstimate,
    SDEMemoryPlanningError,
    _AnalyserWorkspace,
    resolve_time_grid,
)


def _observer_check_interval(observers: dict[str, Any] | None) -> int | None:
    """Smallest configured observer cadence, or None without observers."""
    intervals = []
    for observer in (observers or {}).values():
        interval = getattr(observer, "check_interval_steps", None)
        if interval is None:
            interval = getattr(
                getattr(observer, "config", None), "check_interval_steps", None
            )
        if interval is not None:
            intervals.append(int(interval))
    return min(intervals) if intervals else None

def build_execution_plan(
    *,
    config: Any,
    grid: Any | None,
    model: Any,
    backend: Any,
    integrator: Any,
    analysers: dict[str, Any],
    observers: dict[str, Any] | None = None,
    resources: Any | None,
    device_memory: tuple[int, int] | None = None,
    product_graph: Any | None = None,
) -> SDEExecutionPlan:
    """Validate a job and choose an SDE-internal scan tile size.

    With the compiler-frozen ``product_graph``, a resource-driven multi-tile
    scan splits the graph into tile-scope nodes (executed per tile) and
    final-scope nodes (deferred to one post-merge execution on the full-scan
    products); the plan reports the deferred analysers and accounts their
    full-scan retention honestly.
    """
    time_grid = resolve_time_grid(
        t0=config.t0,
        t1=config.t1,
        dt=config.dt,
    )
    steps = time_grid.total_steps
    _validate_analyser_sampling(config, analysers)

    scan_size = int(getattr(grid, "size", 1)) if grid is not None else 1
    scan_shape = tuple(getattr(grid, "shape", ())) if grid is not None else ()
    n_traj = int(config.n_traj)
    if n_traj < 1:
        raise ValueError("n_traj must be positive")

    record_modes = config.record_modes
    n_record_modes = (
        len(record_modes) if record_modes is not None else int(model.n_modes)
    )
    saved_samples = time_grid.saved_samples(int(config.save_stride))
    real_dtype, real_itemsize = _real_dtype(backend)
    complex_itemsize = real_itemsize * 2
    chunk_steps = max(
        1, int(getattr(getattr(integrator, "config", None), "chunk_steps", 1))
    )
    observer_check_interval_steps = _observer_check_interval(observers)
    effective_chunk_steps: int | None = None
    if observer_check_interval_steps is not None and chunk_steps > 1:
        # Observers only run at chunk boundaries, so the engine clamps fused
        # chunks to the check cadence; the plan reports the fusion cost.
        effective_chunk_steps = max(1, min(chunk_steps, observer_check_interval_steps))

    trajectory_per_point = _checked_product(
        n_traj,
        saved_samples,
        n_record_modes,
        complex_itemsize,
        label="trajectory",
    )
    state_per_point = _checked_product(
        n_traj,
        int(model.n_modes),
        complex_itemsize,
        3,
        label="state workspace",
    )
    noise_per_point = _checked_product(
        n_traj,
        int(model.noise_dim),
        chunk_steps,
        real_itemsize,
        label="noise workspace",
    )
    backend_name = _backend_name(backend)
    trajectory_analysers = _trajectory_analysers(analysers)
    batch_local: frozenset[str] = frozenset()
    batch_frontier: frozenset[str] = frozenset()
    if product_graph is not None:
        batch_local, batch_frontier = typed_batch_scope(
            product_graph, analysers, validate=False
        )
    workspace_analysers = (
        {
            node_id.removeprefix("analyser."): analysers[
                node_id.removeprefix("analyser.")
            ]
            for node_id in batch_local
        }
        if batch_local
        else trajectory_analysers
    )
    budget_kind, budget, available, total, fraction = _memory_budget(
        backend_name,
        resources,
        device_memory=device_memory,
    )
    host_budget, host_available, host_total = _host_memory_budget(resources)
    reserve = _reserve_bytes(budget)
    host_reserve = _reserve_bytes(host_budget)
    device_budget = budget if backend_name == "cupy" else None
    device_reserve = reserve if backend_name == "cupy" else 0
    analysis_enabled = bool(analysers)
    should_keep = config.keep_traj is True or not analysis_enabled
    if product_graph is not None:
        incremental = bool(batch_local and batch_frontier)
    else:
        incremental = bool(trajectory_analysers) and all(
            (
                (
                    callable(getattr(analyser, "reducer", None))
                    and analyser.reducer() is not None
                )
                or callable(getattr(analyser, "create_result_accumulator", None))
            )
            and _analyzer_capabilities(analyser).supports_trajectory_batching
            for analyser in trajectory_analysers.values()
        )
    batching_mode = str(getattr(config, "trajectory_batching", "auto"))
    requested_batch = getattr(config, "trajectory_batch_size", None)
    # The logical RNG group size is numerical provenance: per-point streams
    # are derived per (scan point, absolute group index), so it is fixed by
    # configuration and never by the memory budget. Physical batches must
    # cover whole groups to keep results independent of the batch size.
    logical_rng_group = min(int(config.rng_group_size), n_traj)
    # Adaptive stepping shares one error-controlled time grid across the
    # fused array, so any resource partitioning (trajectory batches or scan
    # tiles) would change the numerical result; the planner rejects it.
    supports_adaptive = getattr(integrator, "supports_adaptive_step", None)
    adaptive_active = bool(
        getattr(config, "adaptive", False)
        and callable(supports_adaptive)
        and supports_adaptive()
    )
    trajectory_batch = n_traj
    per_trajectory_work = (
        trajectory_per_point + state_per_point + noise_per_point
    ) // n_traj
    workspace_cache: dict[int, _AnalyserWorkspace] = {}

    def workspace_for(batch_size: int) -> _AnalyserWorkspace:
        cached = workspace_cache.get(batch_size)
        if cached is not None:
            return cached
        estimate = _analyzer_workspace_bytes(
            max(1, trajectory_per_point * batch_size // n_traj),
            workspace_analysers,
            n_traj=batch_size,
            total_n_traj=n_traj,
            saved_samples=saved_samples,
            n_record_modes=n_record_modes,
            real_itemsize=real_itemsize,
            backend_name=backend_name,
        )
        workspace_cache[batch_size] = estimate
        return estimate

    def one_point_peaks(batch_size: int) -> tuple[int, int]:
        estimate = workspace_for(batch_size)
        device_analyser = estimate.device_bytes + estimate.retained_device_bytes
        host_analyser = estimate.host_bytes + estimate.retained_host_bytes
        work = per_trajectory_work * batch_size
        return (
            device_reserve
            + device_analyser
            + (work if backend_name == "cupy" else 0),
            host_reserve + host_analyser + (work if backend_name != "cupy" else 0),
        )

    def batch_fits(batch_size: int) -> bool:
        device_required, host_required = one_point_peaks(batch_size)
        return (device_budget is None or device_required <= device_budget) and (
            host_budget is None or host_required <= host_budget
        )

    if adaptive_active and batching_mode == "required":
        raise ValueError(
            "adaptive stepping is incompatible with "
            "trajectory_batching='required': the error-controlled time grid "
            "is shared across the fused ensemble, so trajectories cannot be "
            "split into physical batches"
        )
    if requested_batch is not None:
        requested = int(requested_batch)
        if adaptive_active and batching_mode != "off" and requested < n_traj:
            raise ValueError(
                "adaptive stepping is incompatible with trajectory_batch_size "
                f"{requested} < n_traj {n_traj}: the error-controlled time "
                "grid is shared across the fused ensemble, so trajectories "
                "cannot be split into physical batches"
            )
        if (
            batching_mode != "off"
            and requested < n_traj
            and (requested < logical_rng_group or requested % logical_rng_group)
        ):
            raise ValueError(
                f"trajectory_batch_size {requested} is not a whole multiple of "
                f"the logical RNG group size {logical_rng_group}: per-point "
                "random streams are derived per logical group, so physical "
                "batches must cover whole groups. Set trajectory_batch_size to "
                f"a multiple of {logical_rng_group} or set a smaller "
                "rng_group_size."
            )
        trajectory_batch = min(n_traj, requested)
    elif (
        batching_mode != "off"
        and incremental
        and not should_keep
        and (budget is not None or host_budget is not None)
    ):
        candidates = [n_traj]
        candidates.extend(
            range(
                ((n_traj - 1) // logical_rng_group) * logical_rng_group,
                logical_rng_group - 1,
                -logical_rng_group,
            )
        )
        trajectory_batch = next(
            (candidate for candidate in candidates if batch_fits(candidate)), 0
        )
        if trajectory_batch == 0:
            raise SDEMemoryPlanningError(
                "trajectory batching cannot fit one logical RNG group: the "
                f"minimum candidate is {logical_rng_group} trajectories. Raise "
                "the memory budget or set a smaller rng_group_size."
            )
    if batching_mode == "off":
        trajectory_batch = n_traj
    if batching_mode == "required" and trajectory_batch == n_traj and n_traj > 1:
        # Forced batching picks the smallest legal split: one full logical
        # RNG group plus any tail. An ensemble covered by a single group
        # cannot be force-batched at all.
        if logical_rng_group >= n_traj:
            raise SDEMemoryPlanningError(
                f"trajectory_batching='required' cannot split an ensemble of "
                f"{n_traj} trajectories into whole logical RNG groups of "
                f"{logical_rng_group}; set a smaller rng_group_size."
            )
        trajectory_batch = logical_rng_group
    if trajectory_batch < n_traj and product_graph is not None:
        typed_batch_scope(product_graph, analysers, validate=True)
    if trajectory_batch < n_traj and (not incremental or should_keep):
        raise SDEMemoryPlanningError(
            "trajectory batching requires keep_traj=false and every analyser "
            "that directly consumes trajectories to provide a typed reducer"
        )

    if trajectory_batch < n_traj:
        # Budget-derived and required-fallback sizes are estimates and round
        # down to whole logical groups; explicit user requests were already
        # validated as whole multiples above. The resolved batch is recorded
        # in the plan payload.
        trajectory_batch = trajectory_batch // logical_rng_group * logical_rng_group
        if trajectory_batch < logical_rng_group:
            raise SDEMemoryPlanningError(
                "trajectory batching cannot fit one logical RNG group: the "
                f"resolved batch size is below the group size "
                f"{logical_rng_group}. Raise the memory budget or set a "
                "smaller rng_group_size."
            )
    if adaptive_active and trajectory_batch < n_traj:
        raise SDEMemoryPlanningError(
            "adaptive stepping shares one error-controlled time grid across "
            "the whole ensemble, so trajectory batching would change the "
            "numerical result; raise the memory budget or disable adaptive "
            "stepping"
        )
    trajectory_batch_count = math.ceil(n_traj / trajectory_batch)
    stream_analysis = (
        analysis_enabled
        and not should_keep
        and (scan_size > 1 or trajectory_batch_count > 1)
    )

    batch_workspace = workspace_for(trajectory_batch)
    retained_device_per_point = batch_workspace.retained_device_bytes
    retained_host_per_point = batch_workspace.retained_host_bytes
    working_per_point = per_trajectory_work * trajectory_batch
    analyzer_device_per_point = (
        batch_workspace.device_bytes + retained_device_per_point
    )
    analyzer_host_per_point = batch_workspace.host_bytes + retained_host_per_point
    device_per_point = analyzer_device_per_point + (
        working_per_point if backend_name == "cupy" else 0
    )
    host_per_point = analyzer_host_per_point + (
        working_per_point if backend_name != "cupy" else 0
    )
    can_tile = trajectory_batch_count > 1 or stream_analysis

    device_capacity = _tile_capacity(
        device_budget, device_reserve, device_per_point
    )
    host_capacity = _tile_capacity(host_budget, host_reserve, host_per_point)
    capacity = min(scan_size, device_capacity, host_capacity)
    if capacity < 1:
        device_required, host_required = one_point_peaks(trajectory_batch)
        if device_budget is not None and device_required > device_budget:
            raise SDEMemoryPlanningError(
                _minimum_working_set_message(
                    device_budget, device_required, backend_name
                )
            )
        assert host_budget is not None
        raise SDEMemoryPlanningError(
            _minimum_working_set_message(host_budget, host_required, "host")
        )
    if can_tile:
        tile_size = capacity
    else:
        tile_size = scan_size
        required_device = device_reserve + device_per_point * scan_size
        required_host = host_reserve + host_per_point * scan_size
        device_exceeded = (
            device_budget is not None and required_device > device_budget
        )
        host_exceeded = host_budget is not None and required_host > host_budget
        if device_exceeded or host_exceeded:
            required = required_device if device_exceeded else required_host
            limiting_budget = device_budget if device_exceeded else host_budget
            limiting_kind = budget_kind if device_exceeded else "host"
            assert limiting_budget is not None
            raise SDEMemoryPlanningError(
                "The SDE result requires materializing the complete scan trajectory "
                f"({required / _MIB:.1f} MiB estimated) but the {limiting_kind} "
                f"budget is {limiting_budget / _MIB:.1f} MiB. Configure an "
                "analyser with "
                "keep_traj=false so qphase_sde can process scan tiles, reduce the "
                "trajectory size, or increase the resource limit."
            )
    tile_size = int(max(1, tile_size))
    tile_count = math.ceil(scan_size / tile_size)
    if adaptive_active and tile_count > 1:
        raise SDEMemoryPlanningError(
            "adaptive stepping shares one error-controlled time grid across "
            "the fused scan, so scan tiling would change the numerical "
            "result; raise the memory budget, reduce the scan size, or "
            "disable adaptive stepping"
        )
    deferred_analysers: tuple[str, ...] = ()
    deferred_device_bytes = 0
    deferred_host_bytes = 0
    if product_graph is not None and tile_size < scan_size:
        tile_scope, final_scope = typed_scan_scope(product_graph, analysers)
        trajectory_consumers = {
            binding.consumer
            for binding in product_graph.external_inputs
            if binding.product == "trajectories"
        }
        blocked = sorted(trajectory_consumers & final_scope)
        if blocked:
            raise SDEMemoryPlanningError(
                "scan tiling is required by the resource budget, but analysers "
                f"{[node.removeprefix('analyser.') for node in blocked]!r} "
                "consume trajectories without scan-tile support; trajectories "
                "cannot be merged across scan tiles. Reduce the scan size, "
                "raise the memory budget, or use analysers that support scan "
                "tiling."
            )
        published = product_graph.terminal_outputs()
        for node in product_graph.nodes:
            published[node.node_id].update(node.publish_outputs)
        table_products = sorted(
            output.name
            for node in product_graph.nodes
            for output in node.outputs
            if output.table is not None and output.name in published[node.node_id]
        )
        if table_products:
            raise SDEMemoryPlanningError(
                "scan tiling is required by the resource budget, but products "
                f"{table_products!r} are ragged/table terminals; ragged/table "
                "products require a domain-specific reducer (future work) and "
                "cannot be merged across scan tiles. Reduce the scan size or "
                "raise the memory budget so the scan fits one tile."
            )
        if final_scope:
            deferred_analysers = tuple(
                sorted(node.removeprefix("analyser.") for node in final_scope)
            )
            # Deferred execution retains the merged tile-scope products at
            # full scan size until the final phase, whose analysers then run
            # once on full-scan inputs. Merged product sizes are approximated
            # by the accumulator-sized per-point products of the tile-scope
            # trajectory analysers; each downstream tile-scope product is
            # assumed no larger than those (it derives from them). The
            # final-phase workspace is estimated against the merged input.
            downstream_count = sum(
                1
                for node in tile_scope
                if node.removeprefix("analyser.") not in trajectory_analysers
            )
            per_point_product_bytes = (
                batch_workspace.retained_device_bytes
                + batch_workspace.retained_host_bytes
            ) * (1 + downstream_count)
            merged_bytes = scan_size * per_point_product_bytes
            final_workspace = _analyzer_workspace_bytes(
                max(1, merged_bytes),
                {name: analysers[name] for name in deferred_analysers},
                n_traj=n_traj,
                total_n_traj=n_traj,
                saved_samples=saved_samples,
                n_record_modes=n_record_modes,
                real_itemsize=real_itemsize,
                backend_name=backend_name,
            )
            deferred_device_bytes = final_workspace.device_bytes
            deferred_host_bytes = merged_bytes + final_workspace.host_bytes

    retained_host_trajectory = (
        trajectory_per_point * scan_size
        if backend_name == "cupy" and should_keep
        else 0
    )
    device_fixed = device_reserve + deferred_device_bytes
    host_fixed = host_reserve + deferred_host_bytes + retained_host_trajectory
    if can_tile:
        adjusted_capacity = min(
            scan_size,
            _tile_capacity(device_budget, device_fixed, device_per_point),
            _tile_capacity(host_budget, host_fixed, host_per_point),
        )
        if adjusted_capacity < 1:
            if (
                device_budget is not None
                and device_fixed + device_per_point > device_budget
            ):
                raise SDEMemoryPlanningError(
                    _minimum_working_set_message(
                        device_budget,
                        device_fixed + device_per_point,
                        backend_name,
                    )
                )
            assert host_budget is not None
            raise SDEMemoryPlanningError(
                _minimum_working_set_message(
                    host_budget, host_fixed + host_per_point, "host"
                )
            )
        tile_size = min(tile_size, adjusted_capacity)
        tile_count = math.ceil(scan_size / tile_size)

    device_peak = (
        device_fixed + device_per_point * tile_size
        if backend_name == "cupy"
        else 0
    )
    host_peak = host_fixed + host_per_point * tile_size
    peak = device_peak if backend_name == "cupy" else host_peak
    if device_budget is not None and device_peak > device_budget:
        raise SDEMemoryPlanningError(
            _minimum_working_set_message(device_budget, device_peak, backend_name)
        )
    if host_budget is not None and host_peak > host_budget:
        if deferred_analysers:
            raise SDEMemoryPlanningError(
                "scan tiling defers analysers "
                f"{list(deferred_analysers)!r} to post-merge execution: the "
                "merged tile products plus final-phase workspace need an "
                f"estimated {deferred_host_bytes / _MIB:.1f} MiB of host memory "
                f"at full scan size, pushing the host estimate to "
                f"{host_peak / _MIB:.1f} MiB over the "
                f"{host_budget / _MIB:.1f} MiB host budget. Reduce the scan "
                "size, raise the memory budget, or use analysers that support "
                "scan tiling."
            )
        raise SDEMemoryPlanningError(
            "The SDE analyser/retained trajectory requires "
            f"{host_peak / _MIB:.1f} MiB of host memory but the host budget is "
            f"{host_budget / _MIB:.1f} MiB. Reduce the saved trajectory, disable "
            "keep_traj, or use analysers that support trajectory batching."
        )
    memory = SDEMemoryEstimate(
        trajectory_bytes_per_point=trajectory_per_point,
        state_bytes_per_point=state_per_point,
        noise_workspace_bytes_per_point=noise_per_point,
        analyzer_workspace_bytes=(
            analyzer_device_per_point + analyzer_host_per_point
        )
        * tile_size,
        analyzer_device_workspace_bytes=analyzer_device_per_point * tile_size,
        analyzer_host_workspace_bytes=analyzer_host_per_point * tile_size,
        reserve_bytes=reserve,
        host_reserve_bytes=host_reserve,
        full_scan_trajectory_bytes=trajectory_per_point * scan_size,
        estimated_peak_bytes=peak,
        estimated_device_peak_bytes=device_peak,
        estimated_host_peak_bytes=host_peak,
    )
    return SDEExecutionPlan(
        scan_size=scan_size,
        scan_shape=scan_shape,
        n_traj_per_point=n_traj,
        trajectory_batch_size=trajectory_batch,
        trajectory_batch_count=trajectory_batch_count,
        logical_rng_group_size=logical_rng_group,
        steps=steps,
        warmup_steps=time_grid.warmup_steps,
        observation_steps=time_grid.observation_steps,
        saved_samples=saved_samples,
        scan_tile_size=tile_size,
        tile_count=tile_count,
        chunk_steps=chunk_steps,
        real_dtype=real_dtype,
        budget_kind=budget_kind,
        budget_bytes=budget,
        available_device_bytes=available,
        device_total_bytes=total,
        host_budget_bytes=host_budget,
        available_host_bytes=host_available,
        host_total_bytes=host_total,
        gpu_memory_fraction=fraction,
        stream_analysis=stream_analysis,
        # Every execution path derives per-point streams from
        # (master seed, scan point, logical group index): fused, tiled, and
        # trajectory-batched runs of the same configuration draw identical
        # streams, including a short tail group when the group size does not
        # divide n_traj.
        rng_strategy="logical_trajectory_group_seedsequence_v1",
        memory=memory,
        observer_check_interval_steps=observer_check_interval_steps,
        effective_chunk_steps=effective_chunk_steps,
        deferred_analysers=deferred_analysers,
    )

def typed_scan_scope(
    graph: Any, analysers: dict[str, Any]
) -> tuple[frozenset[str], frozenset[str]]:
    """Partition graph nodes into tile-scope and final-scope node ids.

    A node is tile-scope iff its analyser instance declares scan-tiling
    support via ``execution_capabilities().supports_scan_tiling`` and every
    graph ancestor is tile-scope; any other node — and with it all its
    descendants — is final-scope and executes once on the merged full-scan
    products after the tile phase.
    """
    capable: dict[str, bool] = {}
    for node in graph.nodes:
        analyser = analysers.get(node.node_id.removeprefix("analyser."))
        capable[node.node_id] = (
            analyser is not None
            and _analyzer_capabilities(analyser).supports_scan_tiling
        )
    predecessors: dict[str, set[str]] = {node.node_id: set() for node in graph.nodes}
    for edge in graph.edges:
        predecessors[edge.consumer].add(edge.producer)
    tile_scope: set[str] = set()
    for node in graph.topological_order():
        if capable[node.node_id] and predecessors[node.node_id] <= tile_scope:
            tile_scope.add(node.node_id)
    all_nodes = {node.node_id for node in graph.nodes}
    return frozenset(tile_scope), frozenset(all_nodes - tile_scope)

def typed_batch_scope(
    graph: Any,
    analysers: dict[str, Any],
    *,
    validate: bool,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return the transient batch subgraph and its reducer frontier."""
    by_id = {node.node_id: node for node in graph.nodes}
    predecessors: dict[str, set[str]] = {node_id: set() for node_id in by_id}
    consumers: dict[tuple[str, str], set[str]] = {}
    for edge in graph.edges:
        predecessors[edge.consumer].add(edge.producer)
        consumers.setdefault((edge.producer, edge.product), set()).add(edge.consumer)
    trajectory_roots = {
        binding.consumer
        for binding in graph.external_inputs
        if binding.source == "input" and binding.product == "trajectories"
    }
    external_by_node: dict[str, set[str]] = {node_id: set() for node_id in by_id}
    for binding in graph.external_inputs:
        external_by_node[binding.consumer].add(binding.product)

    local: set[str] = set()
    frontier: set[str] = set()
    for node in graph.topological_order():
        label = node.node_id.removeprefix("analyser.")
        analyser = analysers.get(label)
        if analyser is None or not _analyzer_capabilities(
            analyser
        ).supports_trajectory_batching:
            continue
        if external_by_node[node.node_id] - {"trajectories"}:
            continue
        if not predecessors[node.node_id] <= local:
            continue
        if predecessors[node.node_id] & frontier:
            continue
        if node.node_id not in trajectory_roots and not predecessors[node.node_id]:
            continue
        local.add(node.node_id)
        if analyser.reducer() is not None:
            frontier.add(node.node_id)

    if not trajectory_roots or not trajectory_roots <= local:
        return frozenset(), frozenset()

    published = graph.terminal_outputs()
    for node in graph.nodes:
        published[node.node_id].update(node.publish_outputs)
    violations: list[str] = []
    for node_id in sorted(frontier):
        if len(by_id[node_id].outputs) != 1:
            violations.append(
                f"{node_id} has {len(by_id[node_id].outputs)} outputs; "
                "the reducer contract accepts one Dataset"
            )
    for node_id in sorted(local - frontier):
        node = by_id[node_id]
        for output in node.outputs:
            output_consumers = consumers.get((node_id, output.name), set())
            if output.name in published[node_id]:
                violations.append(f"{node_id}:{output.name} is published")
            outside = output_consumers - local
            if outside:
                violations.append(
                    f"{node_id}:{output.name} crosses the batch boundary to "
                    f"{sorted(outside)}"
                )
            if not output_consumers:
                violations.append(f"{node_id}:{output.name} is terminal")
    if violations or not frontier:
        if validate:
            detail = violations or ["the batch-local graph has no reducer frontier"]
            raise SDEMemoryPlanningError(
                "trajectory batching cannot retain a non-reduced batch-local "
                f"output: {'; '.join(detail)}"
            )
        return frozenset(), frozenset()
    return frozenset(local), frozenset(frontier)

def _trajectory_analysers(analysers: dict[str, Any]) -> dict[str, Any]:
    """Select graph-frontier analysers that consume trajectory products."""
    selected: dict[str, Any] = {}
    for name, analyser in analysers.items():
        provider = getattr(analyser, "input_requirements", None)
        if not callable(provider):
            selected[name] = analyser
            continue
        if any(
            requirement.name == "trajectories"
            or (
                requirement.kind is DataKind.TIME_SERIES
                and requirement.quantity == SDEQuantity.FIELD_AMPLITUDE.value
            )
            for requirement in provider()
        ):
            selected[name] = analyser
    return selected
