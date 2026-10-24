"""qphase_cam: shared product helpers
---------------------------------------------------------
Value conversion, schema closing and grid-shaped column helpers shared by the
state and candidate builders, the legacy importer and the export comparisons.

Two rules live here because every caller depends on them:

* A legacy value is converted to JSON *recursively*. Object arrays are handled
  element by element, because the stored legacy diagnostics (for example
  ``scaling_signatures``, a tuple of dicts per candidate) are exactly the
  scientific evidence a typed column must keep. Only a value that still has no
  JSON representation is reported as unmapped.
* A CAM product payload is a typed NumPy array. JSON reaches a product as
  canonical *text* columns, never as an object array, a pickle or a dict
  payload column.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from qphase.core.utils import canonical_json
from qphase.data import AxisSchema, CoordinateSchema, ProductSchema, VariableSchema

__all__ = [
    "canonical_json_text",
    "closed_product",
    "content_digest",
    "coordinate_columns",
    "json_mapping",
    "json_text_column",
    "json_value",
    "string_column",
    "string_widths",
]


# --- value conversion helpers ----------------------------------------------


def json_value(value: Any) -> Any:
    """Return a JSON-serializable equivalent of one legacy value.

    Non-finite floats become ``None`` (JSON has no NaN) and complex values
    become an explicit ``{"real", "imag"}`` pair, so a JSON text column is
    always losslessly parseable.

    Object arrays are converted element by element instead of being refused:
    a legacy artifact stores its diagnostics (e.g. ``scaling_signatures``, one
    tuple of dicts per candidate) and its metadata that way, and that content
    *is* the scientific evidence a classification column must keep. Only a
    value that still has no JSON representation (a set, an arbitrary object)
    raises, and the caller reports the owning key as unmapped.
    """
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return json_value(value.item())
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (complex, np.complexfloating)):
        scalar = complex(value)
        real = scalar.real if np.isfinite(scalar.real) else None
        imag = scalar.imag if np.isfinite(scalar.imag) else None
        return {"real": real, "imag": imag}
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON-serializable")


def json_mapping(mapping: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Convert a mapping for JSON, returning it plus the keys that failed."""
    converted: dict[str, Any] = {}
    unmapped: list[str] = []
    for key, value in mapping.items():
        try:
            converted[str(key)] = json_value(value)
        except (TypeError, ValueError):
            unmapped.append(str(key))
    return converted, sorted(unmapped)


def canonical_json_text(value: Any) -> str:
    """Return canonical JSON text for one column entry."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON column text is not valid JSON: {exc}") from exc
        return canonical_json(parsed)
    return canonical_json(json_value(value))


def string_column(values: Sequence[str]) -> np.ndarray:
    """Return a ``U`` array exactly wide enough for ``values``."""
    items = [str(value) for value in values]
    width = max((len(item) for item in items), default=0)
    return np.asarray(items, dtype=f"U{max(width, 1)}")


def json_text_column(values: Sequence[Any]) -> np.ndarray:
    return string_column([canonical_json_text(value) for value in values])


def string_widths(arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Return the declared ``U`` width of every string column."""
    return {
        name: array.dtype.itemsize // 4
        for name, array in arrays.items()
        if array.dtype.kind == "U"
    }


def closed_product(
    template: ProductSchema,
    *,
    sizes: Mapping[str, int],
    variables: Sequence[VariableSchema],
    coordinates: Sequence[CoordinateSchema],
    attributes: Mapping[str, Any],
    extra_axes: Sequence[AxisSchema] = (),
) -> ProductSchema:
    """Return the template with every axis closed and payloads attached.

    ``extra_axes`` covers products whose axes are data-dependent (an annotation
    product declares one axis per shared field dimension); those axes arrive
    already closed because only the builder knows their size.
    """
    unknown = [axis.name for axis in template.axes if axis.name not in sizes]
    if unknown:
        raise ValueError(f"missing axis sizes for {sorted(unknown)}")
    open_extra = [axis.name for axis in extra_axes if axis.size is None]
    if open_extra:
        raise ValueError(f"extra axes must be closed: {sorted(open_extra)}")
    return ProductSchema(
        kind=template.kind,
        axes=[
            *(
                axis.model_copy(update={"size": int(sizes[axis.name])})
                for axis in template.axes
            ),
            *extra_axes,
        ],
        sampling_bases=list(template.sampling_bases),
        variables=list(variables),
        uncertainties=list(template.uncertainties),
        coordinates=list(coordinates),
        attributes=dict(attributes),
    )


def grid_column(value: Any, *, shape: tuple[int, ...], count: int) -> np.ndarray | None:
    """Flatten one per-point parameter value, or return None when it fits not."""
    array = np.asarray(value)
    if array.ndim == 0:
        column = np.full(count, array)
    elif tuple(array.shape) == shape or (array.ndim == 1 and array.size == count):
        column = array.reshape(count)
    else:
        return None
    dtype = "complex128" if np.iscomplexobj(column) else "float64"
    try:
        return np.asarray(column, dtype=dtype)
    except (TypeError, ValueError):
        return None


def coordinate_columns(
    mapping: Mapping[str, Any] | None,
    *,
    prefix: str,
    shape: tuple[int, ...],
    count: int,
    reserved: set[str] | None = None,
    dims: tuple[str, ...] = ("scan",),
    quantity: str = "scan_parameter",
) -> tuple[
    dict[str, np.ndarray],
    list[VariableSchema],
    list[CoordinateSchema],
    list[str],
]:
    """Turn one parameter mapping into typed columns plus coordinates.

    ``dims`` names the parameter axis the columns span (``scan`` for a state
    scan, ``point`` for a continuation path); the values are one per that axis.
    Values that are not scalar or grid-shaped, values that are not numeric and
    names whose coordinate already exists are returned as unmapped names instead
    of being dropped.
    """
    arrays: dict[str, np.ndarray] = {}
    variables: list[VariableSchema] = []
    coordinates: list[CoordinateSchema] = []
    unmapped: list[str] = []
    taken = reserved if reserved is not None else set()
    for raw_name, value in (mapping or {}).items():
        name = str(raw_name)
        if name in taken:
            unmapped.append(name)
            continue
        column = grid_column(value, shape=shape, count=count)
        if column is None:
            unmapped.append(name)
            continue
        variable_name = f"{prefix}{name}"
        complex_column = column.dtype.kind == "c"
        arrays[variable_name] = column
        variables.append(
            VariableSchema(
                name=variable_name,
                dtype="complex128" if complex_column else "float64",
                value_domain="complex" if complex_column else "real",
                dims=dims,
                quantity=quantity,
            )
        )
        coordinates.append(
            CoordinateSchema(
                name=name,
                variable=variable_name,
                dims=dims,
                role="parameter",
            )
        )
        taken.add(name)
    return arrays, variables, coordinates, sorted(unmapped)


def _digest_value(value: Any) -> Any:
    """Return a JSON view of a value for content hashing, never raising.

    A value with no JSON representation cannot identify an artifact, but it
    must not break the import either: it is replaced by its type name, which is
    deterministic and keeps the digest comparable.
    """
    try:
        return json_value(value)
    except (TypeError, ValueError):
        return {"unhashable_type": type(value).__name__}


def content_digest(payload: Mapping[str, Any], *, length: int = 16) -> str:
    """Return the deterministic content fingerprint of a legacy payload.

    The digest covers the payload itself — arrays, coordinates and metadata —
    and never a path, an ``id()`` or a ``hash()``, so the same scientific
    content always yields the same fingerprint on any machine and a different
    content yields a different one. It identifies an *imported snapshot*: the
    same physics recomputed at a different precision or in a different run is a
    different snapshot, and the solver-owned identity of Phase 10D is what
    survives those.
    """
    normalized = {str(key): _digest_value(item) for key, item in payload.items()}
    text = canonical_json(normalized)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
