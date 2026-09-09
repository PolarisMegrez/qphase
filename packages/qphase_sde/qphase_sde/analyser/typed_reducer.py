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


def _portable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": value.tobytes().hex(),
        }
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_portable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _unportable(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__ndarray__") is True:
            array = np.frombuffer(
                bytes.fromhex(str(value["data"])),
                dtype=np.dtype(str(value["dtype"])),
            ).copy()
            return array.reshape(tuple(int(item) for item in value["shape"]))
        return {key: _unportable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unportable(item) for item in value]
    return value


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
        payloads = getattr(right, "payloads", None)
        if isinstance(payloads, list):
            for payload in payloads:
                left.update(payload)
            return left
        finalize = getattr(right, "finalize", None)
        if not callable(finalize):
            raise TypeError("typed reducer state cannot be merged")
        left.update(finalize())
        return left

    def finalize(self, state: Any) -> Mapping[str, Dataset]:
        products = self.analyzer.build_products(
            state.finalize(), scan_size=None, label=self.label
        )
        if not products:
            raise RuntimeError(f"analyser {self.label!r} produced no reduced product")
        return products

    def checkpoint(self, state: Any) -> dict[str, Any]:
        finalize = getattr(state, "finalize", None)
        if not callable(finalize):
            raise TypeError("typed reducer state cannot be checkpointed")
        return {"payload": _portable(finalize())}

    def restore(self, checkpoint: dict[str, Any]) -> Any:
        payload = checkpoint.get("payload")
        if not isinstance(payload, dict):
            raise TypeError("typed reducer checkpoint must contain a payload")
        state = self.analyzer.create_result_accumulator()
        state.update(_unportable(payload))
        return state
