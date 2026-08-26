"""qphase_sde: Typed Analyser Product Assembly (2.0)
---------------------------------------------------------
Shared machinery turning legacy ``analyze()`` payloads into typed data
products. Analyser-owned builders declare their payload's trailing dims,
quantities, uncertainties and sampling bases; this module handles the
mechanics they all share:

- splitting payloads into numeric leaves / JSON-safe meta / drops;
- stacking per-point leaves over a scan axis (demoting ragged leaves to
  per-point metadata);
- assembling a :class:`~qphase.data.ProductSchema` with named, role-typed
  axes and materializing the dataset.

The legacy bridge (``bridge="legacy_analysis/1"``) uses the same mechanics
with positional axes only, so migrated and graph-ready products round-trip
through the legacy view identically. Builders never pickle payloads and
never copy arrays beyond the scan stacking the engine already performed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Literal

import numpy as np
from qphase.core.utils import canonical_json
from qphase.data import (
    AxisRole,
    AxisSchema,
    CoordinateSchema,
    DataKind,
    Dataset,
    ProductSchema,
    SamplingBasisSchema,
    SpectralDataset,
    StatisticsDataset,
    UncertaintySchema,
    VariableConstraints,
    VariableSchema,
)

__all__ = [
    "StackedLeaves",
    "TypedAxisSpec",
    "assemble_typed_product",
    "json_safe_meta",
    "split_payload_leaves",
    "stack_payload_leaves",
]

_SCAN_AXIS = "scan"


def json_safe_meta(meta: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Split a metadata mapping into JSON-serializable items and dropped keys.

    numpy scalars and arrays are coerced to Python natives first, recursing
    into nested mappings and sequences; values that still fail canonical
    JSON serialization are reported as dropped.
    """

    def _coerce(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): _coerce(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [_coerce(item) for item in value]
        return value

    safe: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in meta.items():
        candidate = _coerce(value)
        try:
            canonical_json(candidate)
        except (TypeError, ValueError):
            dropped.append(str(key))
            continue
        safe[str(key)] = candidate
    return safe, dropped


def split_payload_leaves(
    raw: Any,
    *,
    _prefix: str = "",
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[str]]:
    """Split one raw analyser payload into arrays, JSON-safe meta, and drops.

    Nested dicts (e.g. per-mode result tables) are flattened recursively with
    dotted key paths so their numeric leaves stay typed variables; legacy
    views re-nest them on reconstruction.
    """
    items = raw.items() if isinstance(raw, dict) else [("value", raw)]
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in items:
        key = f"{_prefix}{key}"
        if not key:
            dropped.append("<empty>")
            continue
        array = np.asarray(value)
        if array.dtype.hasobject and isinstance(value, dict):
            sub_arrays, sub_meta, sub_dropped = split_payload_leaves(
                value, _prefix=f"{key}."
            )
            arrays.update(sub_arrays)
            meta.update(sub_meta)
            dropped.extend(sub_dropped)
            continue
        if array.dtype.hasobject or array.dtype.kind in "US":
            # Non-numeric leaves (orientation strings, small config values)
            # are payload metadata, not variables: keep the JSON-safe ones so
            # legacy views can rebuild the original analyser payload.
            safe, _ = json_safe_meta({key: value})
            if key in safe:
                meta[key] = safe[key]
            else:
                dropped.append(key)
            continue
        arrays[key] = array
    return arrays, meta, dropped


@dataclass(frozen=True)
class StackedLeaves:
    """Payload leaves stacked over the scan axis of one logical job."""

    arrays: dict[str, np.ndarray]
    payload_meta: dict[str, Any]
    per_point_meta: list[str]
    dropped: list[str]
    scan_independent: frozenset[str] = frozenset()


def stack_payload_leaves(
    name: str,
    payload: Any,
    *,
    scan_size: int,
) -> StackedLeaves | None:
    """Stack one analyser payload over the scan axis.

    ``payload`` is a single per-point mapping (``scan_size == 1``) or a list
    of ``scan_size`` per-point mappings. Ragged leaves that cannot be stacked
    (e.g. variable-length peak lists) are demoted to per-point metadata;
    missing per-point payloads and inconsistent per-point keys are rejected.
    Returns ``None`` for a ``None`` payload.
    """
    if payload is None:
        return None
    if scan_size <= 1:
        single_arrays, single_meta, single_dropped = split_payload_leaves(payload)
        return StackedLeaves(single_arrays, single_meta, [], single_dropped)
    if not (isinstance(payload, list) and len(payload) == scan_size):
        raise TypeError(
            f"analysis product {name!r}: expected a list of {scan_size} "
            f"per-point payloads, got {type(payload).__name__}"
        )
    missing = [i for i, point in enumerate(payload) if point is None]
    if missing:
        raise TypeError(
            f"analysis product {name!r}: missing payloads for scan points {missing}"
        )
    splits = [split_payload_leaves(point) for point in payload]
    key_sets = [set(split[0]) for split in splits]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise TypeError(
            f"analysis product {name!r}: per-point payload keys differ "
            f"across the scan ({key_sets})"
        )
    arrays: dict[str, np.ndarray] = {}
    payload_meta: dict[str, Any] = {}
    per_point_meta: list[str] = []
    for key in sorted(key_sets[0]):
        try:
            arrays[key] = np.stack([split[0][key] for split in splits])
        except (TypeError, ValueError) as exc:
            # Ragged leaf (e.g. a variable-length peak list): demote to
            # per-point metadata instead of rejecting the whole payload.
            demoted = [split[0][key].tolist() for split in splits]
            safe, _ = json_safe_meta({key: demoted})
            if key not in safe:
                raise TypeError(
                    f"analysis product {name!r}: cannot stack variable "
                    f"{key!r} over scan points and it is not JSON-safe"
                ) from exc
            payload_meta[key] = safe[key]
            per_point_meta.append(key)
    meta_keys = {key for split in splits for key in split[1]}
    for key in sorted(meta_keys):
        payload_meta[key] = [split[1].get(key) for split in splits]
        per_point_meta.append(key)
    dropped = sorted({key for split in splits for key in split[2]})
    return StackedLeaves(arrays, payload_meta, per_point_meta, dropped)


@dataclass(frozen=True)
class TypedAxisSpec:
    """Template of one named trailing axis of a typed analyser product.

    Sizes close at assembly time from the arrays that span the axis. Regular
    coordinates declare ``start``/``step``; explicit coordinates are backed
    by coordinate variables via :class:`~qphase.data.CoordinateSchema`.
    """

    name: str
    role: AxisRole
    units: str = ""
    coordinate: Literal["regular", "explicit"] = "explicit"
    start: float | None = None
    step: float | None = None


@dataclass
class _TrailingAxis:
    """One resolved trailing axis and its closed size."""

    spec: TypedAxisSpec
    size: int


def _resolve_declared_dims(
    key: str,
    declared_dims: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...] | None:
    """Resolve the trailing dims of one leaf by exact name, then by glob."""
    declared = declared_dims.get(key)
    if declared is not None:
        return tuple(declared)
    for pattern, dims in declared_dims.items():
        if any(mark in pattern for mark in "*?[") and fnmatchcase(key, pattern):
            return tuple(dims)
    return None


def assemble_typed_product(
    name: str,
    leaves: StackedLeaves,
    *,
    scan_size: int,
    kind: DataKind,
    declared_dims: Mapping[str, tuple[str, ...]] | None = None,
    axis_specs: Mapping[str, TypedAxisSpec] | None = None,
    quantities: Mapping[str, str] | None = None,
    constraints: Mapping[str, VariableConstraints] | None = None,
    uncertainties: Sequence[UncertaintySchema] = (),
    sampling_bases: Sequence[SamplingBasisSchema] = (),
    coordinates: Sequence[CoordinateSchema] = (),
    attributes: Mapping[str, Any] | None = None,
    owner: str = "engine.sde",
) -> Dataset | None:
    """Assemble one typed product from stacked payload leaves.

    Every numeric leaf becomes a variable: its dims are the leading ``scan``
    axis (when ``scan_size > 1``) plus the trailing dims declared for it
    (exact key or ``fnmatch`` glob, for per-mode dotted keys). Leaves without
    a declaration fall back to positional ``dim<N>`` index axes, matching
    the legacy bridge layout. String and ragged leaves stay in
    ``payload_meta`` so legacy views rebuild the original payload.

    Returns ``None`` when the payload has no numeric leaf (a product schema
    requires at least one variable).
    """
    declared_dims = declared_dims or {}
    axis_specs = axis_specs or {}
    quantities = quantities or {}
    constraints = constraints or {}

    arrays = dict(leaves.arrays)
    coordinate_specs = list(coordinates)
    scan_independent = set(leaves.scan_independent)
    if scan_size > 1:
        for coordinate in list(coordinate_specs):
            array = arrays.get(coordinate.variable)
            if (
                array is not None
                and coordinate.dims[:1] == (_SCAN_AXIS,)
                and array.shape[:1] == (scan_size,)
                and all(
                    np.array_equal(array[0], row, equal_nan=True)
                    for row in array[1:]
                )
            ):
                arrays[coordinate.variable] = array[0]
                scan_independent.add(coordinate.variable)
                coordinate_specs[coordinate_specs.index(coordinate)] = (
                    coordinate.model_copy(
                        update={"dims": coordinate.dims[1:], "role": "dimension"}
                    )
                )
    axes: list[AxisSchema] = []
    if scan_size > 1:
        axes.append(
            AxisSchema(name=_SCAN_AXIS, role=AxisRole.PARAMETER, size=scan_size)
        )
    trailing: dict[str, _TrailingAxis] = {}
    positional: dict[str, AxisSchema] = {}
    variables: list[VariableSchema] = []
    clean_arrays: dict[str, np.ndarray] = {}
    for key, array in arrays.items():
        leading = (
            ()
            if key in scan_independent or scan_size <= 1
            else (_SCAN_AXIS,)
        )
        trailing_ndim = array.ndim - len(leading)
        declared = _resolve_declared_dims(key, declared_dims)
        if declared is None:
            declared_list: list[str] = []
            for position in range(trailing_ndim):
                axis_name = f"dim{position}"
                if axis_name not in positional:
                    positional[axis_name] = AxisSchema(
                        name=axis_name, role=AxisRole.INDEX
                    )
                declared_list.append(axis_name)
            declared = tuple(declared_list)
        elif len(declared) != trailing_ndim:
            raise TypeError(
                f"analysis product {name!r}: variable {key!r} declares dims "
                f"{declared} but its array has {trailing_ndim} trailing "
                f"dimensions"
            )
        for dim, extent in zip(declared, array.shape[len(leading) :], strict=True):
            if dim in positional:
                continue
            spec = axis_specs.get(dim)
            if spec is None:
                raise TypeError(
                    f"analysis product {name!r}: variable {key!r} declares "
                    f"unknown axis {dim!r}"
                )
            resolved = trailing.get(dim)
            if resolved is None:
                trailing[dim] = _TrailingAxis(spec, int(extent))
            elif resolved.size != int(extent):
                raise TypeError(
                    f"analysis product {name!r}: axis {dim!r} has "
                    f"conflicting sizes {resolved.size} and {int(extent)}"
                )
        dtype = np.dtype(array.dtype)
        variables.append(
            VariableSchema(
                name=key,
                dtype=dtype.str,
                value_domain="complex" if dtype.kind == "c" else "real",
                dims=(*leading, *declared),
                quantity=quantities.get(key, ""),
                constraints=constraints.get(key, VariableConstraints()),
            )
        )
        clean_arrays[key] = array
    if not variables:
        # ProductSchema requires at least one variable; payloads without any
        # numeric leaf cannot form a product (their meta is reported through
        # the bundle's ``dropped_products`` metadata by the caller).
        return None

    for resolved in trailing.values():
        axes.append(
            AxisSchema(
                name=resolved.spec.name,
                role=resolved.spec.role,
                size=resolved.size,
                coordinate=resolved.spec.coordinate,
                start=resolved.spec.start,
                step=resolved.spec.step,
                units=resolved.spec.units,
            )
        )
    schema = ProductSchema(
        kind=kind,
        axes=[*axes, *positional.values()],
        sampling_bases=list(sampling_bases),
        variables=variables,
        uncertainties=list(uncertainties),
        coordinates=coordinate_specs,
        attributes={
            "source_analyser": str(name),
            "dropped_keys": leaves.dropped,
            "payload_meta": leaves.payload_meta,
            "per_point_meta": leaves.per_point_meta,
            **dict(attributes or {}),
        },
    )
    dataset_class = SpectralDataset if kind is DataKind.SPECTRAL else StatisticsDataset
    return dataset_class.from_arrays(
        schema,
        clean_arrays,
        owner=owner,
        provenance={"source_analyser": str(name)},
    )


def add_scan_parameter_coordinates(
    dataset: Dataset,
    coordinates: Mapping[str, np.ndarray],
) -> Dataset:
    """Attach flattened scan parameters to one graph-ready product."""
    if not coordinates or "scan" not in {axis.name for axis in dataset.axes}:
        return dataset
    from qphase.data.runtime import DictProductBacking, HostArrayHandle

    handles = {
        variable.name: dataset.handle(variable.name)
        for variable in dataset.variables
    }
    variables = list(dataset.variables)
    coordinate_specs = list(dataset.schema.coordinates)
    coordinate_names = {item.name for item in coordinate_specs}
    scan_size = dataset.axis("scan").size
    assert scan_size is not None
    for name, raw in coordinates.items():
        variable_name = f"parameter.{name}"
        if variable_name in handles:
            raise TypeError(
                f"scan coordinate {variable_name!r} collides with product data"
            )
        values = np.asarray(raw)
        if values.shape != (scan_size,):
            raise TypeError(
                f"scan coordinate {name!r} has shape {values.shape}, "
                f"expected {(scan_size,)}"
            )
        variable = VariableSchema(
                name=variable_name,
                dtype=values.dtype.str,
                value_domain="complex" if values.dtype.kind == "c" else "real",
                dims=("scan",),
                quantity="scan_parameter",
            )
        variables.append(variable)
        handles[variable_name] = HostArrayHandle(
            values, variable, owner="engine.sde"
        )
        coordinate_name = (
            str(name) if str(name) not in coordinate_names else f"parameter.{name}"
        )
        coordinate_names.add(coordinate_name)
        coordinate_specs.append(
            CoordinateSchema(
                name=coordinate_name,
                variable=variable_name,
                dims=("scan",),
                role="parameter",
            )
        )
    schema = ProductSchema.model_validate(
        {
            **dataset.schema.model_dump(mode="python"),
            "variables": variables,
            "coordinates": coordinate_specs,
        }
    )
    return type(dataset)(
        schema,
        DictProductBacking(handles),
        provenance=dataset.provenance,
    )
