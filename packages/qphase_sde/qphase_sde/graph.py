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
    input_bindings: Mapping[str, Mapping[str, str]] | None = None,
) -> ProductGraph | None:
    """Compile the analyser product graph from resolved plugin configuration.

    The compiler may instantiate analyser declarations and their child
    plugins, but never creates a backend or runs numerical code. The runtime
    receives this validated graph as part of ``CompiledJob`` and only supplies
    the already-created execution instances.

    ``input_bindings`` maps ``{analyser_label: {requirement_name: target}}``
    and replaces same-name inference for the listed requirements only: a
    target of ``"<producer_label>.<product_name>"`` creates an internal edge,
    while ``"input.<product_name>"`` binds the requirement to an external
    job-input product. Requirements without an explicit binding keep the
    same-name inference.
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
        requested_publish = list((publish_outputs or {}).get(str(name), []))
        unknown_publish = set(requested_publish) - {output.name for output in outputs}
        if unknown_publish:
            raise ValueError(
                f"publish_outputs for analyser {name!r} references unknown "
                f"products {sorted(unknown_publish)!r}"
            )
        nodes.append(
            ProductNode(
                node_id=node_id,
                producer=f"analyser:{name}",
                requirements=list(instance.input_requirements()),
                outputs=outputs,
                publish_outputs=requested_publish,
            )
        )

    node_by_name = {node.node_id.removeprefix("analyser."): node for node in nodes}
    for analyser_name, _product_names in (publish_outputs or {}).items():
        node = node_by_name.get(str(analyser_name))
        if node is None:
            raise ValueError(
                f"publish_outputs references unknown analyser {analyser_name!r}"
            )

    explicit = _resolve_input_bindings(input_bindings, node_by_name)

    edges: list[GraphEdge] = []
    external: list[ExternalInputBinding] = []
    for node in nodes:
        for requirement in node.requirements:
            bound = explicit.get((node.node_id, requirement.name))
            if bound is not None:
                if isinstance(bound, GraphEdge):
                    edges.append(bound)
                else:
                    external.append(bound)
                continue
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
            elif requirement.optional:
                continue
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


def _resolve_input_bindings(
    input_bindings: Mapping[str, Mapping[str, str]] | None,
    node_by_name: Mapping[str, ProductNode],
) -> dict[tuple[str, str], GraphEdge | ExternalInputBinding]:
    """Resolve explicit input bindings to graph edges or external bindings.

    Returns the resolved binding per ``(consumer node id, requirement name)``
    slot. Every entry is validated strictly: the analyser label, requirement
    name, target syntax and, for internal targets, the producer label and its
    declared product must all exist; binding a requirement to the analyser's
    own output stays rejected as a self-cycle.
    """
    resolved: dict[tuple[str, str], GraphEdge | ExternalInputBinding] = {}
    for analyser_name, bindings in (input_bindings or {}).items():
        node = node_by_name.get(str(analyser_name))
        if node is None:
            raise ValueError(
                f"input_bindings references unknown analyser {analyser_name!r}"
            )
        if not isinstance(bindings, Mapping):
            raise TypeError(
                f"input_bindings for analyser {analyser_name!r} must be a mapping"
            )
        requirement_names = {req.name for req in node.requirements}
        for requirement_name, target in bindings.items():
            requirement_name = str(requirement_name)
            if requirement_name not in requirement_names:
                raise ValueError(
                    f"input_bindings for analyser {analyser_name!r} references "
                    f"unknown requirement {requirement_name!r}"
                )
            source, separator, product = str(target).partition(".")
            if not separator or not source or not product:
                raise ValueError(
                    f"input_bindings target {target!r} for requirement "
                    f"{requirement_name!r} of analyser {analyser_name!r} must be "
                    "'<producer_label>.<product_name>' or 'input.<product_name>'"
                )
            slot = (node.node_id, requirement_name)
            if source == "input":
                resolved[slot] = ExternalInputBinding(
                    consumer=node.node_id,
                    requirement=requirement_name,
                    source="input",
                    product=product,
                )
                continue
            producer = node_by_name.get(source)
            if producer is None:
                raise ValueError(
                    f"input_bindings target {target!r} for requirement "
                    f"{requirement_name!r} of analyser {analyser_name!r} "
                    f"references unknown producer {source!r}"
                )
            if product not in {output.name for output in producer.outputs}:
                raise ValueError(
                    f"input_bindings target {target!r} for requirement "
                    f"{requirement_name!r} of analyser {analyser_name!r} "
                    f"references undeclared output {product!r} of analyser "
                    f"{source!r}"
                )
            if producer.node_id == node.node_id:
                raise ValueError(
                    f"analyser {node.node_id!r} cannot bind requirement "
                    f"{requirement_name!r} to its own output {product!r}"
                )
            resolved[slot] = GraphEdge(
                producer=producer.node_id,
                product=product,
                consumer=node.node_id,
                requirement=requirement_name,
            )
    return resolved
