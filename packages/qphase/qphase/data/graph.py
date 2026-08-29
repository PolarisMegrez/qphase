"""qphase: Product Graph Contracts
---------------------------------------------------------
Freezes the typed dependency description between data products. Engines compile
a ``ProductGraph`` from plugin declarations; the core validates references and
acyclicity but knows nothing about domain semantics (PSD, Allan, fits, ...).

Public API
----------
ProductRequirement
    Typed input requirement of one node.
ProductDeclaration
    Declared output product of one node.
ProductNode
    One graph node, identified by a compiler-assigned node id.
ProductGraph
    Validated acyclic graph of product nodes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .kinds import DataKind

__all__ = [
    "ProductDeclaration",
    "ProductGraph",
    "ProductNode",
    "ProductRequirement",
]


class ProductRequirement(BaseModel):
    """Typed input requirement; consumers select by kind/quantity/fields."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Local label of the required input.")
    kind: DataKind | None = None
    quantity: str | None = None
    required_fields: list[str] = Field(default_factory=list)
    accepted_axes: list[str] = Field(default_factory=list)
    optional: bool = False


class ProductDeclaration(BaseModel):
    """Declared output product of one graph node."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: DataKind
    quantity: str | None = None
    fields: list[str] = Field(default_factory=list)


class ProductNode(BaseModel):
    """One graph node with an explicit compiler-assigned identity."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(
        min_length=1,
        description="Stable id assigned by the workflow/product compiler.",
    )
    producer: str = Field(
        description="Identifier of the producing component (plugin/engine)."
    )
    declaration: ProductDeclaration
    requirements: list[ProductRequirement] = Field(default_factory=list)


class GraphEdge(BaseModel):
    """Directed edge from a producer node to a consumer node."""

    model_config = ConfigDict(extra="forbid")

    producer: str = Field(description="Id of the producer node.")
    consumer: str = Field(description="Id of the consumer node.")
    requirement: str = Field(
        default="", description="Name of the requirement this edge satisfies."
    )


class ProductGraph(BaseModel):
    """Validated acyclic graph of product nodes."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[ProductNode]
    edges: list[GraphEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_graph(self) -> ProductGraph:
        node_ids = [node.node_id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("product graph nodes must have unique node ids")
        known = set(node_ids)
        for edge in self.edges:
            if edge.producer not in known:
                raise ValueError(f"edge references unknown producer {edge.producer!r}")
            if edge.consumer not in known:
                raise ValueError(f"edge references unknown consumer {edge.consumer!r}")
        self._check_acyclic(node_ids)
        return self

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

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for CLI/GUI display."""
        return {
            "nodes": [
                {
                    "id": node.node_id,
                    "producer": node.producer,
                    "product": node.declaration.name,
                    "kind": str(node.declaration.kind),
                }
                for node in self.topological_order()
            ],
            "edges": [
                {
                    "producer": edge.producer,
                    "consumer": edge.consumer,
                    "requirement": edge.requirement,
                }
                for edge in self.edges
            ],
        }
