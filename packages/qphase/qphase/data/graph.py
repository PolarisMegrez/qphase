"""qphase: Product Graph Contracts
---------------------------------------------------------
Freezes the typed dependency description between data products. Engines compile
a ``ProductGraph`` from plugin declarations; the core validates references,
input-slot binding, edge compatibility and acyclicity, but knows nothing about
domain semantics (PSD, Allan, fits, ...).

Requirement binding discipline: a required requirement (``optional=False``)
must have exactly one binding — an edge into its slot XOR an
``ExternalInputBinding`` naming an external source and product — while an optional
requirement has at most one binding (no fan-in this phase; a future
cardinality mechanism will state it explicitly). External bindings declare
where an unbound-by-edge input comes from (job input, artifact, ...).

Edge compatibility is conflict-based: when both endpoints declare a comparable
facet (``kind``, ``quantity``, ``fields``), the two must agree, while absent
facets never conflict. ``value_domain`` and axis constraints are deliberately
NOT checked here: they require a closed ``ProductSchema`` and are owned by the
materialization boundary (``requirement_matches`` and backing validation),
not by the graph.

One node represents one plugin execution: it consumes its ``requirements`` and
atomically returns the mapping of its declared ``outputs``. ``publish_outputs``
is the compiler-resolved persistence disposition of those outputs (terminal
outputs are published by default; intermediates only when explicitly listed) —
it is execution/persistence metadata, not part of the product type declaration.

Public API
----------
ProductRequirement
    Typed input requirement of one node.
ProductDeclaration
    Declared output product of one node.
ProductNode
    One graph node, identified by a compiler-assigned node id.
GraphEdge
    Directed edge from one declared output to one consumer requirement.
ExternalInputBinding
    Binding of one consumer requirement to an external source.
ProductGraph
    Validated acyclic graph of product nodes.
requirement_matches
    Check a requirement against a concrete product schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .kinds import DataKind
from .schema import ProductSchema

__all__ = [
    "ExternalInputBinding",
    "GraphEdge",
    "ProductDeclaration",
    "ProductGraph",
    "ProductNode",
    "ProductRequirement",
    "requirement_matches",
]


class ProductRequirement(BaseModel):
    """Typed input requirement; consumers select by kind/quantity/fields.

    ``value_domain`` distinguishes complex field trajectories from sampled
    real scalar series: when set together with ``quantity``, the same variable
    must satisfy both.

    ``align_with`` names a sibling requirement of the same node whose
    parameter-axis coordinates this input must match elementwise (values,
    length, uniqueness, sampling basis). Alignment is validated when the
    inputs materialize; compilers reject ``align_with`` references to unknown
    sibling requirements.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Local label of the required input.")
    kind: DataKind | None = None
    quantity: str | None = None
    value_domain: Literal["real", "complex"] | None = None
    required_fields: list[str] = Field(default_factory=list)
    accepted_axes: list[str] = Field(default_factory=list)
    align_with: str | None = None
    optional: bool = False


class ProductDeclaration(BaseModel):
    """Declared output product of one graph node."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: DataKind
    quantity: str | None = None
    fields: list[str] = Field(default_factory=list)


class ProductNode(BaseModel):
    """One graph node with an explicit compiler-assigned identity.

    The node executes once and returns its declared ``outputs`` atomically;
    output and requirement names are unique within the node.
    ``publish_outputs`` names the
    subset of outputs the compiler resolved for persistence; all other
    outputs are graph-local intermediates.
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(
        min_length=1,
        description="Stable id assigned by the workflow/product compiler.",
    )
    producer: str = Field(
        description="Identifier of the producing component (plugin/engine)."
    )
    requirements: list[ProductRequirement] = Field(default_factory=list)
    outputs: list[ProductDeclaration] = Field(min_length=1)
    publish_outputs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_outputs(self) -> ProductNode:
        names = [output.name for output in self.outputs]
        if len(set(names)) != len(names):
            raise ValueError(f"node {self.node_id!r} declares duplicate output names")
        known = set(names)
        for published in self.publish_outputs:
            if published not in known:
                raise ValueError(
                    f"node {self.node_id!r} publishes undeclared output {published!r}"
                )
        if len(set(self.publish_outputs)) != len(self.publish_outputs):
            raise ValueError(f"node {self.node_id!r} lists duplicate publish outputs")
        requirement_names = [req.name for req in self.requirements]
        if len(set(requirement_names)) != len(requirement_names):
            raise ValueError(
                f"node {self.node_id!r} declares duplicate requirement names"
            )
        known_requirements = set(requirement_names)
        for req in self.requirements:
            if req.align_with is not None and req.align_with not in known_requirements:
                raise ValueError(
                    f"node {self.node_id!r} requirement {req.name!r} aligns with "
                    f"unknown sibling requirement {req.align_with!r}"
                )
            if req.align_with == req.name:
                raise ValueError(
                    f"node {self.node_id!r} requirement {req.name!r} aligns with itself"
                )
        return self


class GraphEdge(BaseModel):
    """Directed edge from one declared output to one consumer requirement."""

    model_config = ConfigDict(extra="forbid")

    producer: str = Field(description="Id of the producer node.")
    product: str = Field(
        min_length=1,
        description="Name of the producer node output this edge carries.",
    )
    consumer: str = Field(description="Id of the consumer node.")
    requirement: str = Field(
        min_length=1,
        description="Name of the consumer requirement this edge satisfies.",
    )


class ExternalInputBinding(BaseModel):
    """Binding of one consumer requirement to one external product."""

    model_config = ConfigDict(extra="forbid")

    consumer: str = Field(description="Id of the consumer node.")
    requirement: str = Field(
        description="Name of the consumer requirement this binding satisfies.",
    )
    source: str = Field(
        min_length=1,
        description="External source label (e.g. job input / artifact).",
    )
    product: str = Field(
        min_length=1,
        description="Name of the product in the external source.",
    )


def _check_edge_compatibility(
    edge: GraphEdge,
    declaration: ProductDeclaration,
    requirement: ProductRequirement,
) -> None:
    """Reject an edge whose endpoints declare conflicting facets.

    Conflict-based, not match-based: a facet conflicts only when both sides
    specify it and disagree, so absent facets never conflict. ``value_domain``
    and axis constraints require a closed ``ProductSchema`` and are checked at
    the materialization boundary, not here.
    """
    label = (
        f"edge {edge.producer!r}:{edge.product} -> {edge.consumer!r}:{edge.requirement}"
    )
    if requirement.kind is not None and declaration.kind != requirement.kind:
        raise ValueError(
            f"{label}: product {declaration.name!r} of node {edge.producer!r} "
            f"declares kind {declaration.kind.value!r} but requirement "
            f"{requirement.name!r} of node {edge.consumer!r} requires kind "
            f"{requirement.kind.value!r}"
        )
    if (
        requirement.quantity is not None
        and declaration.quantity is not None
        and declaration.quantity != requirement.quantity
    ):
        raise ValueError(
            f"{label}: product {declaration.name!r} of node {edge.producer!r} "
            f"declares quantity {declaration.quantity!r} but requirement "
            f"{requirement.name!r} of node {edge.consumer!r} requires quantity "
            f"{requirement.quantity!r}"
        )
    if declaration.fields and not set(requirement.required_fields) <= set(
        declaration.fields
    ):
        raise ValueError(
            f"{label}: requirement {requirement.name!r} of node "
            f"{edge.consumer!r} requires fields {requirement.required_fields!r} "
            f"but product {declaration.name!r} of node {edge.producer!r} "
            f"declares fields {declaration.fields!r}"
        )


class ProductGraph(BaseModel):
    """Validated acyclic graph of product nodes.

    Every requirement slot follows a strict binding discipline: a required
    requirement (``optional=False``) has exactly one binding — an edge into
    the slot XOR an ``ExternalInputBinding`` naming an external source —
    while an optional requirement has at most one binding. Every edge is
    checked for facet conflicts between the bound declaration and
    requirement.
    """

    model_config = ConfigDict(extra="forbid")

    nodes: list[ProductNode]
    edges: list[GraphEdge] = Field(default_factory=list)
    external_inputs: list[ExternalInputBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_graph(self) -> ProductGraph:
        by_id = {node.node_id: node for node in self.nodes}
        if len(by_id) != len(self.nodes):
            raise ValueError("product graph nodes must have unique node ids")
        bound_slots: set[tuple[str, str]] = set()
        for edge in self.edges:
            producer = by_id.get(edge.producer)
            if producer is None:
                raise ValueError(f"edge references unknown producer {edge.producer!r}")
            consumer = by_id.get(edge.consumer)
            if consumer is None:
                raise ValueError(f"edge references unknown consumer {edge.consumer!r}")
            if edge.product not in {output.name for output in producer.outputs}:
                raise ValueError(
                    f"edge references undeclared output {edge.product!r} of "
                    f"node {edge.producer!r}"
                )
            if edge.requirement not in {req.name for req in consumer.requirements}:
                raise ValueError(
                    f"edge references undeclared requirement "
                    f"{edge.requirement!r} of node {edge.consumer!r}"
                )
            slot = (edge.consumer, edge.requirement)
            if slot in bound_slots:
                raise ValueError(
                    f"requirement {edge.requirement!r} of node {edge.consumer!r} "
                    f"is bound by more than one edge"
                )
            bound_slots.add(slot)
            declaration = next(
                output for output in producer.outputs if output.name == edge.product
            )
            requirement = next(
                req for req in consumer.requirements if req.name == edge.requirement
            )
            _check_edge_compatibility(edge, declaration, requirement)
        external_counts: dict[tuple[str, str], int] = {}
        for binding in self.external_inputs:
            consumer = by_id.get(binding.consumer)
            if consumer is None:
                raise ValueError(
                    f"external input binding references unknown consumer "
                    f"{binding.consumer!r}"
                )
            if binding.requirement not in {req.name for req in consumer.requirements}:
                raise ValueError(
                    f"external input binding references undeclared requirement "
                    f"{binding.requirement!r} of node {binding.consumer!r}"
                )
            slot = (binding.consumer, binding.requirement)
            external_counts[slot] = external_counts.get(slot, 0) + 1
        for node in self.nodes:
            for requirement in node.requirements:
                slot = (node.node_id, requirement.name)
                bindings = int(slot in bound_slots) + external_counts.get(slot, 0)
                if bindings == 0 and not requirement.optional:
                    raise ValueError(
                        f"required requirement {requirement.name!r} of node "
                        f"{node.node_id!r} has no binding; declare an edge or "
                        f"an external input binding"
                    )
                if bindings > 1:
                    raise ValueError(
                        f"requirement {requirement.name!r} of node "
                        f"{node.node_id!r} has more than one binding; each "
                        f"requirement binds at most one edge or external input"
                    )
        self._check_acyclic(list(by_id))
        return self

    def validate_external_products(
        self,
        products: Mapping[str, Mapping[str, ProductSchema]],
    ) -> None:
        """Validate external bindings against already materialized schemas.

        ``products`` is keyed by the external ``source`` label and then by
        product name. The graph does not load artifacts or infer a product;
        callers provide the metadata they have already resolved at the
        materialization boundary.
        """
        by_id = {node.node_id: node for node in self.nodes}
        for binding in self.external_inputs:
            source_products = products.get(binding.source)
            if source_products is None or binding.product not in source_products:
                raise ValueError(
                    f"external binding {binding.source!r}:{binding.product!r} "
                    "does not resolve to a product schema"
                )
            node = by_id[binding.consumer]
            requirement = next(
                item for item in node.requirements if item.name == binding.requirement
            )
            if not requirement_matches(requirement, source_products[binding.product]):
                raise ValueError(
                    f"external product {binding.source!r}:{binding.product!r} "
                    f"does not satisfy requirement {binding.requirement!r} "
                    f"of node {binding.consumer!r}"
                )

    def _check_acyclic(self, node_ids: list[str]) -> None:
        indegree = dict.fromkeys(node_ids, 0)
        successors: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        for edge in self.edges:
            indegree[edge.consumer] += 1
            successors[edge.producer].append(edge.consumer)

        queue = [node_id for node_id, degree in indegree.items() if degree == 0]
        visited = 0
        while queue:
            current = queue.pop()
            visited += 1
            for nxt in successors[current]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        if visited != len(node_ids):
            raise ValueError("product graph must be acyclic")

    def topological_order(self) -> list[ProductNode]:
        """Return nodes in a valid topological order."""
        by_id = {node.node_id: node for node in self.nodes}
        indegree = dict.fromkeys(by_id, 0)
        successors: dict[str, list[str]] = {node_id: [] for node_id in by_id}
        for edge in self.edges:
            indegree[edge.consumer] += 1
            successors[edge.producer].append(edge.consumer)
        queue = [node_id for node_id, degree in indegree.items() if degree == 0]
        order: list[ProductNode] = []
        while queue:
            current = queue.pop(0)
            order.append(by_id[current])
            for nxt in successors[current]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        return order

    def terminal_outputs(self) -> dict[str, set[str]]:
        """Return the outputs with no consuming edge, keyed by node id.

        Compilers use these as the default publish set: terminal outputs are
        persisted unless the producer says otherwise, while intermediates
        stay graph-local unless explicitly listed in ``publish_outputs``.
        """
        consumed = {(edge.producer, edge.product) for edge in self.edges}
        return {
            node.node_id: {
                output.name
                for output in node.outputs
                if (node.node_id, output.name) not in consumed
            }
            for node in self.nodes
        }

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for CLI/GUI display."""
        return {
            "nodes": [
                {
                    "id": node.node_id,
                    "producer": node.producer,
                    "outputs": [output.name for output in node.outputs],
                    "publish_outputs": list(node.publish_outputs),
                }
                for node in self.topological_order()
            ],
            "edges": [
                {
                    "producer": edge.producer,
                    "product": edge.product,
                    "consumer": edge.consumer,
                    "requirement": edge.requirement,
                }
                for edge in self.edges
            ],
        }


def requirement_matches(requirement: ProductRequirement, schema: ProductSchema) -> bool:
    """Check whether a concrete product schema satisfies a requirement.

    ``quantity`` and ``value_domain`` constrain variables: each, when set,
    must be satisfied by at least one variable, and when both are set the
    same variable must satisfy both. ``required_fields`` must be variable
    names; a non-empty ``accepted_axes`` bounds the schema's axis names.
    """
    if requirement.kind is not None and schema.kind != requirement.kind:
        return False
    if requirement.quantity is not None or requirement.value_domain is not None:
        candidates = list(schema.variables)
        if requirement.quantity is not None:
            candidates = [v for v in candidates if v.quantity == requirement.quantity]
        if requirement.value_domain is not None:
            candidates = [
                v for v in candidates if v.value_domain == requirement.value_domain
            ]
        if not candidates:
            return False
    variable_names = {v.name for v in schema.variables}
    if any(field not in variable_names for field in requirement.required_fields):
        return False
    if requirement.accepted_axes:
        axis_names = {axis.name for axis in schema.axes}
        if not axis_names <= set(requirement.accepted_axes):
            return False
    return True
