"""Schema-driven composition of typed datasets.

Composition provenance contract: the composed product records a
``composition`` mapping with the operation parameters (``operation`` and
``axis``; for ``stack`` the created axis and its size follow from the output
schema) plus an ordered ``inputs`` list carrying every source identity. Each
entry holds the source's ``artifact_id``/``product`` provenance keys when it
carries them; sources that are themselves compositions carry their inner
``composition`` record, so nested stack/concat chains keep the full
composition tree; any other source carries the owner marker of its runtime
handles. No source is silently dropped. The first input's own
``artifact_id``/``product`` keys are input identities, so they are not
attributed to the composed product's top-level provenance.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from .datasets import Dataset

__all__ = ["concat_datasets", "stack_datasets"]


def _source_identities(datasets: Sequence[Dataset]) -> list[dict[str, Any]]:
    """Ordered identity of every composition input.

    Sources carrying ``artifact_id``/``product`` provenance keys (artifact
    restore path) are identified by them. Sources that are themselves
    compositions carry their inner ``composition`` record, so nested
    stack/concat chains keep the full composition tree instead of degrading
    to the runtime owner marker. Any other source is identified by the owner
    marker of its runtime handles.
    """
    identities: list[dict[str, Any]] = []
    for dataset in datasets:
        provenance = dataset.provenance
        identity = {
            key: provenance[key]
            for key in ("artifact_id", "product")
            if key in provenance
        }
        if identity:
            identities.append(identity)
            continue
        composition = provenance.get("composition")
        if composition is not None:
            identities.append({"composition": composition})
            continue
        owners = sorted(
            {dataset.handle(variable.name).owner for variable in dataset.variables}
        )
        if len(owners) == 1:
            identities.append({"owner": owners[0]})
        else:
            identities.append({"owners": owners})
    return identities


def _composition_provenance(
    first: Dataset,
    datasets: Sequence[Dataset],
    *,
    operation: str,
    axis: str,
) -> dict[str, Any]:
    provenance = {
        key: value
        for key, value in first.provenance.items()
        if key not in ("artifact_id", "product")
    }
    provenance["composition"] = {
        "operation": operation,
        "axis": axis,
        "inputs": _source_identities(datasets),
    }
    return provenance


def _normalized_schema(dataset: Dataset, axis: str) -> dict[str, Any]:
    schema = dataset.schema.model_dump(mode="json")
    schema["axes"] = [
        {**item, "size": None} if item["name"] == axis else item
        for item in schema["axes"]
    ]
    return schema


def _concatenate(values: list[Any], *, axis: int, device: str) -> Any:
    if device == "cpu":
        return np.concatenate([np.asarray(value) for value in values], axis=axis)
    module = importlib.import_module(type(values[0]).__module__.split(".", 1)[0])
    concatenate = getattr(module, "concatenate", None)
    if not callable(concatenate):
        raise TypeError(f"device array module {module.__name__!r} has no concatenate")
    return concatenate(values, axis=axis)


def _stack(values: list[Any], *, axis: int, device: str) -> Any:
    if device == "cpu":
        return np.stack([np.asarray(value) for value in values], axis=axis)
    module = importlib.import_module(type(values[0]).__module__.split(".", 1)[0])
    stack = getattr(module, "stack", None)
    if not callable(stack):
        raise TypeError(f"device array module {module.__name__!r} has no stack")
    return stack(values, axis=axis)


def concat_datasets(datasets: Sequence[Dataset], *, axis: str) -> Dataset:
    """Concatenate compatible runtime datasets along one existing axis."""
    if not datasets:
        raise ValueError("cannot concatenate an empty Dataset sequence")
    first = datasets[0]
    if first.attributes.get("table") in {
        "peak_candidates",
        "peak_paths",
        "row_table",
    }:
        raise RuntimeError("ragged/table products require a domain-specific reducer")
    if axis not in {item.name for item in first.axes}:
        raise ValueError(f"dataset has no axis {axis!r}")
    if not first.is_runtime_backed:
        raise TypeError("dataset composition requires runtime-backed inputs")
    devices = first.devices
    if len(devices) != 1:
        raise ValueError("dataset composition requires one shared device")
    device = devices[0]
    normalized = _normalized_schema(first, axis)
    for dataset in datasets[1:]:
        if type(dataset) is not type(first):
            raise TypeError("datasets must use the same concrete Dataset class")
        if not dataset.is_runtime_backed or dataset.devices != devices:
            raise ValueError("datasets must be runtime-backed on the same device")
        if dataset.attributes != first.attributes:
            raise ValueError("datasets have inconsistent attributes")
        if _normalized_schema(dataset, axis) != normalized:
            raise ValueError(
                "datasets have incompatible schema, coordinates, attributes, "
                "or sampling bases"
            )

    arrays: dict[str, Any] = {}
    for variable in first.variables:
        values = [dataset.handle(variable.name).materialize() for dataset in datasets]
        if axis in variable.dims:
            arrays[variable.name] = _concatenate(
                values, axis=variable.dims.index(axis), device=device
            )
        else:
            reference = np.asarray(values[0])
            if not all(
                np.array_equal(reference, np.asarray(value)) for value in values[1:]
            ):
                raise ValueError(
                    f"non-concatenated variable {variable.name!r} differs "
                    "between inputs"
                )
            arrays[variable.name] = values[0]

    schema_data = first.schema.model_dump(mode="python")
    axis_sizes = [dataset.axis(axis).size for dataset in datasets]
    if any(size is None for size in axis_sizes):
        raise ValueError(f"cannot concatenate open axis {axis!r}")
    for item in schema_data["axes"]:
        if item["name"] == axis:
            item["size"] = sum(int(size) for size in axis_sizes if size is not None)
    provenance = _composition_provenance(first, datasets, operation="concat", axis=axis)
    return type(first).from_arrays(
        type(first.schema).model_validate(schema_data),
        arrays,
        owner="qphase.data.concat",
        device=device,
        provenance=provenance,
    )


def stack_datasets(
    datasets: Sequence[Dataset],
    *,
    axis: str,
    role: str = "parameter",
) -> Dataset:
    """Stack compatible runtime datasets along one new outer axis."""
    if not datasets:
        raise ValueError("cannot stack an empty Dataset sequence")
    first = datasets[0]
    if axis in {item.name for item in first.axes}:
        raise ValueError(f"dataset already has axis {axis!r}")
    if first.attributes.get("table") in {
        "peak_candidates",
        "peak_paths",
        "row_table",
    }:
        raise RuntimeError("ragged/table products require a domain-specific reducer")
    if not first.is_runtime_backed:
        raise TypeError("dataset composition requires runtime-backed inputs")
    devices = first.devices
    if len(devices) != 1:
        raise ValueError("dataset composition requires one shared device")
    device = devices[0]
    schema = first.schema.model_dump(mode="json")
    for dataset in datasets[1:]:
        if type(dataset) is not type(first):
            raise TypeError("datasets must use the same concrete Dataset class")
        if not dataset.is_runtime_backed or dataset.devices != devices:
            raise ValueError("datasets must be runtime-backed on the same device")
        if dataset.schema.model_dump(mode="json") != schema:
            raise ValueError("datasets have incompatible schemas")
        if dataset.attributes != first.attributes:
            raise ValueError("datasets have inconsistent attributes")

    coordinates = {item.variable: item for item in first.coordinates()}
    arrays: dict[str, Any] = {}
    variables = []
    for variable in first.variables:
        values = [dataset.handle(variable.name).materialize() for dataset in datasets]
        coordinate = coordinates.get(variable.name)
        if coordinate is not None and not (
            coordinate.role == "parameter" and not coordinate.dims
        ):
            reference = np.asarray(values[0])
            if not all(
                np.array_equal(reference, np.asarray(value)) for value in values[1:]
            ):
                raise ValueError(
                    f"coordinate variable {variable.name!r} differs between inputs"
                )
            arrays[variable.name] = values[0]
            variables.append(variable)
            continue
        arrays[variable.name] = _stack(values, axis=0, device=device)
        variables.append(variable.model_copy(update={"dims": (axis, *variable.dims)}))

    schema_data = first.schema.model_dump(mode="python")
    schema_data["axes"] = [
        {"name": axis, "role": role, "size": len(datasets)},
        *schema_data["axes"],
    ]
    schema_data["variables"] = variables
    schema_data["coordinates"] = [
        {
            **coordinate,
            "dims": (axis,),
        }
        if coordinate["role"] == "parameter" and not coordinate["dims"]
        else coordinate
        for coordinate in schema_data["coordinates"]
    ]
    provenance = _composition_provenance(first, datasets, operation="stack", axis=axis)
    return type(first).from_arrays(
        type(first.schema).model_validate(schema_data),
        arrays,
        owner="qphase.data.stack",
        device=device,
        provenance=provenance,
    )
