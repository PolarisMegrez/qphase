"""Typed product reducers for trajectory-batched analysers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from qphase.data import Dataset

from ..contracts.reducer import ReducerProtocol

__all__ = ["TypedProductReducer"]


def _assign_nested(target: dict[Any, Any], name: str, value: Any) -> None:
    parts = name.split(".")
    current = target
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _payload_from_dataset(product: Dataset) -> dict[str, Any]:
    """Recover one analyser payload without constructing an SDE result view."""
    payload: dict[str, Any] = {}
    for variable in product.variables:
        _assign_nested(
            payload,
            variable.name,
            np.asarray(product.handle(variable.name).materialize()),
        )
    payload_meta = product.attributes.get("payload_meta", {})
    if isinstance(payload_meta, Mapping):
        for name, value in payload_meta.items():
            _assign_nested(payload, str(name), value)
    return payload


class TypedProductReducer(ReducerProtocol):
    """Reduce one analyser's typed products and emit typed products.

    The existing numerical accumulators remain analyser-owned. This adapter
    changes only the execution boundary: input is a typed Dataset and output
    is the analyser's declared typed Dataset mapping. No ``SDEResult`` or
    engine-level legacy view is involved.
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
        state.update(_payload_from_dataset(view))
        return state

    def merge(self, left: Any, right: Any) -> Any:
        del left, right
        raise NotImplementedError(
            "typed analyser reducers are consumed sequentially in this runtime"
        )

    def finalize(self, state: Any) -> Mapping[str, Dataset]:
        products = self.analyzer.build_products(
            state.finalize(), scan_size=None, label=self.label
        )
        if not products:
            raise RuntimeError(f"analyser {self.label!r} produced no reduced product")
        return products

    def checkpoint(self, state: Any) -> dict[str, Any]:
        del state
        raise NotImplementedError(
            "typed analyser reducer checkpoints are not supported"
        )

    def restore(self, checkpoint: dict[str, Any]) -> Any:
        del checkpoint
        raise NotImplementedError(
            "typed analyser reducer checkpoints are not supported"
        )
