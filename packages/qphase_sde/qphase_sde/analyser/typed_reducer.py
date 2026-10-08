"""Typed product reducers for trajectory-batched analysers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qphase.data import Dataset

from ..contracts.reducer import ReducerProtocol

__all__ = ["TypedProductReducer"]


class TypedProductReducer(ReducerProtocol):
    """Reduce one analyser's typed products and emit typed products.

    The existing numerical accumulators remain analyser-owned. This adapter
    changes only the execution boundary: input is a typed Dataset and output
    is the analyser's declared typed Dataset mapping.
    """

    def __init__(self, analyzer: Any, *, label: str) -> None:
        self.analyzer = analyzer
        self.label = label

    def prepare(self, schema: Any, context: Any) -> Any:
        del schema, context
        return self.analyzer.create_result_accumulator()

    def consume(self, view: Dataset, state: Any, reporter: Any) -> Any:
        del reporter
        if not isinstance(view, Dataset):
            raise TypeError("typed reducer input must be a Dataset")
        state.consume_product(view)
        return state

    def merge(self, left: Any, right: Any) -> Any:
        left.merge(right)
        return left

    def finalize(self, state: Any) -> Mapping[str, Dataset]:
        products = state.finalize_product(self.label)
        if not products:
            raise RuntimeError(f"analyser {self.label!r} produced no reduced product")
        return products

    def checkpoint(self, state: Any) -> dict[str, Any]:
        return state.checkpoint()

    def restore(self, checkpoint: dict[str, Any]) -> Any:
        state = self.analyzer.create_result_accumulator()
        state.restore(checkpoint)
        return state
