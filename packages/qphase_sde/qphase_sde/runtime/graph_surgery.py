"""qphase_sde: Product-graph surgery for tiled and batched execution.

The engine freezes one ProductGraph and then runs it in phases: a tile-scope
sub-graph per scan tile, optionally a batch-scope sub-graph per trajectory
batch, and a final graph for the deferred analysers. These helpers build those
sub-graphs from declarations only; they never touch analyser instances.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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
