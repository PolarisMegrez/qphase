"""Runtime executor for the resource-independent ProductGraph contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..data.datasets import Dataset
from ..data.graph import ProductGraph, ProductNode, requirement_matches

__all__ = ["ProductGraphExecutor"]


NodeRunner = Callable[[ProductNode, Mapping[str, Dataset]], Mapping[str, Dataset]]


class ProductGraphExecutor:
    """Execute a validated product graph through injected node runners.

    The executor owns data-flow only. A runner owns the numerical operation for
    one node and returns all declared products atomically; no resource package
    names or algorithms are interpreted here.
    """

    def __init__(
        self,
        graph: ProductGraph,
        runners: Mapping[str, NodeRunner],
    ) -> None:
        self.graph = graph
        self.runners = dict(runners)
        node_ids = {node.node_id for node in graph.nodes}
        missing = sorted(node_ids - set(self.runners))
        if missing:
            raise ValueError(f"missing runners for product graph nodes: {missing}")

    def run(
        self,
        external_inputs: Mapping[str, Mapping[str, Dataset]],
        *,
        context: Any | None = None,
    ) -> dict[str, dict[str, Dataset]]:
        """Run nodes in order and return terminal or explicitly published products."""
        produced: dict[str, dict[str, Dataset]] = {}
        published = self.graph.terminal_outputs()
        for node in self.graph.nodes:
            published[node.node_id].update(node.publish_outputs)
        remaining = {(edge.producer, edge.product): 0 for edge in self.graph.edges}
        for edge in self.graph.edges:
            remaining[(edge.producer, edge.product)] += 1
        for node in self.graph.topological_order():
            self._check_cancelled(context)
            self._report(context, node, "started")
            try:
                inputs = self._resolve_inputs(node, produced, external_inputs)
            except Exception:
                self._report(context, node, "failed")
                raise
            try:
                outputs = self.runners[node.node_id](node, inputs)
            except Exception:
                self._report(context, node, "failed")
                raise
            if not isinstance(outputs, Mapping):
                raise TypeError(
                    f"runner for node {node.node_id!r} must return a product mapping"
                )
            declared = {output.name for output in node.outputs}
            actual = set(outputs)
            if actual != declared:
                raise ValueError(
                    f"runner for node {node.node_id!r} returned products "
                    f"{sorted(actual)!r}; expected {sorted(declared)!r}"
                )
            for name, product in outputs.items():
                if not isinstance(product, Dataset):
                    raise TypeError(
                        f"runner for node {node.node_id!r} returned non-Dataset "
                        f"output {name!r}"
                    )
                declaration = next(
                    output for output in node.outputs if output.name == name
                )
                if product.kind is not declaration.kind:
                    raise TypeError(
                        f"node {node.node_id!r} output {name!r} has kind "
                        f"{product.kind.value!r}, expected {declaration.kind.value!r}"
                    )
                if declaration.quantity is not None and not any(
                    variable.quantity == declaration.quantity
                    for variable in product.variables
                ):
                    raise TypeError(
                        f"node {node.node_id!r} output {name!r} does not provide "
                        f"declared quantity {declaration.quantity!r}"
                    )
                variable_names = {variable.name for variable in product.variables}
                missing_fields = set(declaration.fields) - variable_names
                if missing_fields:
                    raise TypeError(
                        f"node {node.node_id!r} output {name!r} is missing "
                        f"declared fields {sorted(missing_fields)!r}"
                    )
            produced[node.node_id] = dict(outputs)
            for edge in self.graph.edges:
                if edge.consumer != node.node_id:
                    continue
                key = (edge.producer, edge.product)
                remaining[key] -= 1
                if remaining[key] == 0 and edge.product not in published[edge.producer]:
                    produced[edge.producer].pop(edge.product, None)
                    if not produced[edge.producer]:
                        produced.pop(edge.producer)
            self._report(context, node, "completed")
        return {
            node_id: {
                name: products[name] for name in published[node_id] if name in products
            }
            for node_id, products in produced.items()
            if published[node_id]
        }

    @staticmethod
    def _check_cancelled(context: Any | None) -> None:
        if context is None:
            return
        cancellation = getattr(context, "cancellation", None)
        if cancellation is not None:
            cancellation.raise_if_cancelled()

    @staticmethod
    def _report(context: Any | None, node: ProductNode, state: str) -> None:
        if context is None:
            return
        progress = getattr(context, "progress", None)
        if progress is not None:
            progress.status(
                f"product node {node.node_id} {state}",
                stage=node.node_id,
            )

    def _resolve_inputs(
        self,
        node: ProductNode,
        produced: Mapping[str, Mapping[str, Dataset]],
        external_inputs: Mapping[str, Mapping[str, Dataset]],
    ) -> dict[str, Dataset]:
        inputs: dict[str, Dataset] = {}
        edge_by_slot = {
            (edge.consumer, edge.requirement): edge
            for edge in self.graph.edges
            if edge.consumer == node.node_id
        }
        external_by_slot = {
            (binding.consumer, binding.requirement): binding
            for binding in self.graph.external_inputs
            if binding.consumer == node.node_id
        }
        for requirement in node.requirements:
            slot = (node.node_id, requirement.name)
            edge = edge_by_slot.get(slot)
            binding = external_by_slot.get(slot)
            if edge is not None:
                source = produced[edge.producer].get(edge.product)
                if source is None:
                    raise ValueError(
                        f"producer {edge.producer!r} did not publish output "
                        f"{edge.product!r} required by {node.node_id!r}"
                    )
                product = source
            elif binding is not None:
                external = external_inputs.get(binding.source, {}).get(binding.product)
                if external is None:
                    if requirement.optional:
                        continue
                    raise ValueError(
                        f"external input {binding.source!r}:{binding.product!r} "
                        f"is unavailable for {node.node_id!r}"
                    )
                product = external
            elif requirement.optional:
                continue
            else:
                raise ValueError(
                    f"required input {requirement.name!r} of node "
                    f"{node.node_id!r} is unbound"
                )
            if not requirement_matches(requirement, product.schema):
                raise TypeError(
                    f"product for input {requirement.name!r} of node "
                    f"{node.node_id!r} does not satisfy its schema requirement"
                )
            inputs[requirement.name] = product
        return inputs
