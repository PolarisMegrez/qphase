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
    One graph node, identified by its content fingerprint.
ProductGraph
    Validated acyclic graph of product nodes.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.utils import canonical_json
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
    """One node of a product graph, identified by content fingerprint."""

    model_config = ConfigDict(extra="forbid")

    producer: str = Field(
        description="Identifier of the producing component (plugin/engine)."
    )
    declaration: ProductDeclaration
    requirements: list[ProductRequirement] = Field(default_factory=list)

    def fingerprint(self) -> str:
        """Return the content fingerprint identifying this node."""
        payload = self.model_dump(mode="json")
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class GraphEdge(BaseModel):
    """Directed edge from a producer node to a consumer node."""

    model_config = ConfigDict(extra="forbid")

    producer: str = Field(description="Fingerprint of the producer node.")
    consumer: str = Field(description="Fingerprint of the consumer node.")
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
        fingerprints = [node.fingerprint() for node in self.nodes]
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("product graph nodes must have unique fingerprints")
        known = set(fingerprints)
        for edge in self.edges:
            if edge.producer not in known:
                raise ValueError(
                    f"edge references unknown producer {edge.producer!r}"
                )
            if edge.consumer not in known:
                raise ValueError(
                    f"edge references unknown consumer {edge.consumer!r}"
                )
        self._check_acyclic(fingerprints)
        return self

    def _check_acyclic(self, fingerprints: list[str]) -> None:
        indegree = dict.fromkeys(fingerprints, 0)
        successors: dict[str, list[str]] = {fp: [] for fp in fingerprints}
        for edge in self.edges:
            indegree[edge.consumer] += 1
            successors[edge.producer].append(edge.consumer)

        queue = [fp for fp, degree in indegree.items() if degree == 0]
        visited = 0
        while queue:
            current = queue.pop()
            visited += 1
            for nxt in successors[current]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        if visited != len(fingerprints):
            raise ValueError("product graph must be acyclic")

    def topological_order(self) -> list[ProductNode]:
        """Return nodes in a valid topological order."""
        by_fingerprint = {node.fingerprint(): node for node in self.nodes}
        indegree = dict.fromkeys(by_fingerprint, 0)
        successors: dict[str, list[str]] = {fp: [] for fp in by_fingerprint}
        for edge in self.edges:
            indegree[edge.consumer] += 1
            successors[edge.producer].append(edge.consumer)
        queue = [fp for fp, degree in indegree.items() if degree == 0]
        order: list[ProductNode] = []
        while queue:
            current = queue.pop(0)
            order.append(by_fingerprint[current])
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
                    "fingerprint": node.fingerprint(),
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
