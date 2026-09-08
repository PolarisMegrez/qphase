"""Control-plane product graph construction for the SDE engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qphase.data import ExternalInputBinding, GraphEdge, ProductGraph, ProductNode

__all__ = ["compile_product_graph"]


def compile_product_graph(
    *,
    plugin_config: Mapping[str, Any],
    registry: Any,
    publish_outputs: Mapping[str, list[str]] | None = None,
) -> ProductGraph | None:
    """Compile the analyser product graph from resolved plugin configuration.

    The compiler may instantiate analyser declarations and their child
    plugins, but never creates a backend or runs numerical code. The runtime
    receives this validated graph as part of ``CompiledJob`` and only supplies
    the already-created execution instances.
    """
    raw_analysers = plugin_config.get("analyser")
    if not isinstance(raw_analysers, Mapping) or not raw_analysers:
        return None

    nodes: list[ProductNode] = []
    output_owners: dict[str, tuple[str, str]] = {}
    for name, raw in raw_analysers.items():
        if not isinstance(raw, Mapping):
            raise TypeError(f"analyser config {name!r} must be a mapping")
        instance = registry.create_plugin_instance(
            "analyser", {"name": str(name), **dict(raw)}
        )
        if not callable(getattr(instance, "input_requirements", None)) or not callable(
            getattr(instance, "output_specs", None)
        ):
            if not nodes:
                return None
            raise TypeError(
                f"analyser {name!r} does not satisfy the typed graph contract"
            )
        node_id = f"analyser.{name}"
        outputs = list(instance.output_specs(label=str(name)))
        for output in outputs:
            if output.name in output_owners:
                raise ValueError(f"duplicate analyser output {output.name!r}")
            output_owners[output.name] = (node_id, output.name)
        nodes.append(
            ProductNode(
                node_id=node_id,
                producer=f"analyser:{name}",
                requirements=list(instance.input_requirements()),
                outputs=outputs,
                publish_outputs=list((publish_outputs or {}).get(str(name), [])),
            )
        )

    edges: list[GraphEdge] = []
    external: list[ExternalInputBinding] = []
    for node in nodes:
        for requirement in node.requirements:
            owner = output_owners.get(requirement.name)
            if owner is not None:
                if owner[0] == node.node_id:
                    raise ValueError(
                        f"analyser {node.node_id!r} consumes its own output "
                        f"{requirement.name!r}"
                    )
                edges.append(
                    GraphEdge(
                        producer=owner[0],
                        product=owner[1],
                        consumer=node.node_id,
                        requirement=requirement.name,
                    )
                )
            else:
                external.append(
                    ExternalInputBinding(
                        consumer=node.node_id,
                        requirement=requirement.name,
                        source="input",
                        product=requirement.name,
                    )
                )
    return ProductGraph(nodes=nodes, edges=edges, external_inputs=external)
