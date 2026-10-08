"""qphase_sde: Typed Analyser Product Assembly (2.0)
---------------------------------------------------------
Shared machinery turning analyser payloads into typed data products.
Analyser-owned builders declare their payload's trailing dims,
quantities, uncertainties and sampling bases; this module handles the
mechanics they all share:

- splitting payloads into numeric leaves / JSON-safe meta / drops;
- stacking per-point leaves over a scan axis (demoting ragged leaves to
  per-point metadata);
- assembling a :class:`~qphase.data.ProductSchema` with named, role-typed
  axes and materializing the dataset.

Builders never pickle payloads and never copy arrays beyond the scan stacking
the engine already performed.
"""

from __future__ import annotations

import math
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
    "assemble_row_table_product",
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
        if isinstance(value, float) and not math.isfinite(value):
            return None
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
    scan_size: int | None,
) -> StackedLeaves | None:
    """Stack one analyser payload over the scan axis.

    ``scan_size`` is ``None`` when the job had no parameter grid: the payload
    is the bare per-point mapping and no scan axis exists. Otherwise
    ``scan_size`` is the exact size of a genuine scan — including 1 — and the
    payload is the bare per-point mapping for ``scan_size == 1`` or a list of
    ``scan_size`` per-point mappings for larger scans (list wrapping exists
    iff ``scan_size > 1``). Scan payloads are stacked over a scan axis of
    exactly ``scan_size`` and their meta leaves become per-point lists.
    Ragged leaves that cannot be stacked (e.g. variable-length peak lists)
    are demoted to per-point metadata; missing per-point payloads and
    inconsistent per-point keys are rejected. Returns ``None`` for a ``None``
    payload.
    """
    if payload is None:
        return None
    if scan_size is None:
        single_arrays, single_meta, single_dropped = split_payload_leaves(payload)
        return StackedLeaves(single_arrays, single_meta, [], single_dropped)
    points = [payload] if scan_size == 1 else payload
    if not (isinstance(points, list) and len(points) == scan_size):
        raise TypeError(
            f"analysis product {name!r}: expected a list of {scan_size} "
            f"per-point payloads, got {type(payload).__name__}"
        )
    missing = [i for i, point in enumerate(points) if point is None]
    if missing:
        raise TypeError(
            f"analysis product {name!r}: missing payloads for scan points {missing}"
        )
    splits = [split_payload_leaves(point) for point in points]
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


def assemble_row_table_product(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    scan_size: int | None,
    fields: Sequence[str],
    integer_fields: Sequence[str] = (),
    scan_index_field: str | None = None,
    scan_value_field: str | None = None,
    quantities: Mapping[str, str] | None = None,
    payload_key: str = "rows",
    payload_meta: Mapping[str, Any] | None = None,
    source_scan: bool = False,
    attributes: Mapping[str, Any] | None = None,
    owner: str = "engine.sde",
) -> Dataset:
    """Materialize a small row table as a graph-ready statistics product.

    Row-oriented analysers naturally produce variable-length tables.  The
    table remains rectangular at the product boundary: rows are grouped on
    the scan axis and padded with ``row_valid``.  Original records and
    diagnostic strings stay in product attributes, while declared numeric
    fields are typed variables.  This is intentionally limited to compact
    analyser summaries; large numerical arrays use their analyser-specific
    builders.
    """
    field_names = tuple(fields)
    integer_names = frozenset(integer_fields)
    if not field_names:
        raise ValueError(f"row table {name!r} must declare at least one field")
    if not integer_names <= set(field_names):
        raise ValueError("integer_fields must be a subset of fields")

    groups: list[list[Mapping[str, Any]]]
    if scan_size is None:
        groups = [list(rows)]
    else:
        groups = [[] for _ in range(scan_size)]
        value_to_index: dict[float, int] = {}
        if scan_value_field is not None:
            values = sorted(
                {
                    float(row[scan_value_field])
                    for row in rows
                    if scan_value_field in row
                }
            )
            value_to_index = {value: index for index, value in enumerate(values)}
        for row in rows:
            if scan_index_field is not None and scan_index_field in row:
                index = int(row[scan_index_field])
            elif scan_value_field is not None and scan_value_field in row:
                index = value_to_index[float(row[scan_value_field])]
            else:
                raise ValueError(
                    f"row table {name!r} needs a scan index or scan value field"
                )
            if index < 0 or index >= scan_size:
                raise ValueError(
                    f"row table {name!r} row has scan index {index}, "
                    f"outside scan size {scan_size}"
                )
            groups[index].append(row)

    row_count = max((len(group) for group in groups), default=0)
    row_count = max(row_count, 1)
    arrays: dict[str, np.ndarray] = {}
    for field in field_names:
        dtype = np.int64 if field in integer_names else np.float64
        fill = -1 if field in integer_names else np.nan
        shape = (len(groups), row_count) if scan_size is not None else (row_count,)
        array = np.full(shape, fill, dtype=dtype)
        for group_index, group in enumerate(groups):
            for row_index, row in enumerate(group):
                value = row.get(field, fill)
                if scan_size is None:
                    array[row_index] = value
                else:
                    array[group_index, row_index] = value
        arrays[field] = array
    valid_shape = (len(groups), row_count) if scan_size is not None else (row_count,)
    valid = np.zeros(valid_shape, dtype=np.int8)
    for group_index, group in enumerate(groups):
        if scan_size is None:
            valid[: len(group)] = 1
        else:
            valid[group_index, : len(group)] = 1
    arrays["row_valid"] = valid
    declared_dims = {field: ("row",) for field in (*field_names, "row_valid")}
    meta = {payload_key: list(rows), **dict(payload_meta or {})}
    safe_meta, dropped = json_safe_meta(meta)
    leaves = StackedLeaves(
        arrays=arrays,
        payload_meta=safe_meta,
        per_point_meta=[],
        dropped=dropped,
    )
    product = assemble_typed_product(
        name,
        leaves,
        scan_size=scan_size,
        kind=DataKind.STATISTICS,
        declared_dims=declared_dims,
        axis_specs={"row": TypedAxisSpec("row", AxisRole.INDEX)},
        quantities=quantities,
        attributes={
            "graph_ready": True,
            "table": "row_table",
            "source_scan": source_scan,
            **dict(attributes or {}),
        },
        owner=owner,
    )
    assert product is not None
    return product


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
    scan_size: int | None,
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
    axis (present iff ``scan_size is not None``, of exactly that size — a
    1-point scan keeps a size-1 scan axis) plus the trailing dims declared
    for it (exact key or ``fnmatch`` glob, for per-mode dotted keys). Leaves
    without a declaration fall back to positional ``dim<N>`` index axes,
    matching the legacy bridge layout. String and ragged leaves stay in
    ``payload_meta`` so legacy views rebuild the original payload.

    Returns ``None`` when the payload has no numeric leaf (a product schema
    requires at least one variable); the bundle adapter treats that as a
    producer failure for registered analysers and routes observer/metadata
    payloads into the bundle metadata instead.
    """
    declared_dims = declared_dims or {}
    axis_specs = axis_specs or {}
    quantities = quantities or {}
    constraints = constraints or {}

    arrays = dict(leaves.arrays)
    coordinate_specs = list(coordinates)
    scan_independent = set(leaves.scan_independent)
    if scan_size is not None:
        for coordinate in list(coordinate_specs):
            array = arrays.get(coordinate.variable)
            if (
                array is not None
                and coordinate.dims[:1] == (_SCAN_AXIS,)
                and array.shape[:1] == (scan_size,)
                and all(
                    np.array_equal(array[0], row, equal_nan=True) for row in array[1:]
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
    if scan_size is not None:
        axes.append(
            AxisSchema(name=_SCAN_AXIS, role=AxisRole.PARAMETER, size=scan_size)
        )
    trailing: dict[str, _TrailingAxis] = {}
    positional: dict[str, AxisSchema] = {}
    variables: list[VariableSchema] = []
    clean_arrays: dict[str, np.ndarray] = {}
    for key, array in arrays.items():
        leading = () if key in scan_independent or scan_size is None else (_SCAN_AXIS,)
        trailing_ndim = array.ndim - len(leading)
        declared = _resolve_declared_dims(key, declared_dims)
        if declared is None:
            declared_list: list[str] = []
            for position, extent in enumerate(array.shape[len(leading) :]):
                axis_name = f"{key}.dim{position}"
                if axis_name not in positional:
                    positional[axis_name] = AxisSchema(
                        name=axis_name,
                        role=AxisRole.INDEX,
                        size=int(extent),
                    )
                elif positional[axis_name].size != int(extent):
                    raise TypeError(
                        f"analysis product {name!r}: positional axis "
                        f"{axis_name!r} has conflicting sizes "
                        f"{positional[axis_name].size} and {int(extent)}"
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
        # numeric leaf cannot form a product (the bundle adapter fails the
        # job for registered analysers and routes observer/metadata payloads
        # into the bundle metadata instead).
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
    """Attach flattened scan parameters to one graph-ready product.

    Re-attaching coordinates that are already present with identical values
    is a no-op, so per-tile attachment followed by a full-scan attachment
    after the merge is safe; a genuine name or value collision still fails.
    """
    if not coordinates or "scan" not in {axis.name for axis in dataset.axes}:
        return dataset
    from qphase.data.runtime import DictProductBacking, HostArrayHandle

    handles = {
        variable.name: dataset.handle(variable.name) for variable in dataset.variables
    }
    variables = list(dataset.variables)
    coordinate_specs = list(dataset.schema.coordinates)
    coordinate_names = {item.name for item in coordinate_specs}
    scan_size = dataset.axis("scan").size
    assert scan_size is not None
    for name, raw in coordinates.items():
        variable_name = f"parameter.{name}"
        values = np.asarray(raw)
        if values.shape != (scan_size,):
            raise TypeError(
                f"scan coordinate {name!r} has shape {values.shape}, "
                f"expected {(scan_size,)}"
            )
        if variable_name in handles:
            existing = np.asarray(dataset.handle(variable_name).materialize())
            if existing.shape == values.shape and np.array_equal(
                existing, values, equal_nan=True
            ):
                continue
            raise TypeError(
                f"scan coordinate {variable_name!r} collides with product data"
            )
        variable = VariableSchema(
            name=variable_name,
            dtype=values.dtype.str,
            value_domain="complex" if values.dtype.kind == "c" else "real",
            dims=("scan",),
            quantity="scan_parameter",
        )
        variables.append(variable)
        handles[variable_name] = HostArrayHandle(values, variable, owner="engine.sde")
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
