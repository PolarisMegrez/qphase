"""qphase_cam: the annotation product builder
---------------------------------------------------------
One annotation product describes *one* kind of annotation over a set of targets:
what was checked (``annotation_kind``), what it was attached to
(``target_identity`` + ``target_kind``), what the verdict was (``status``), how it
was obtained (``method_metadata``) and the typed results themselves (``fields``).

Two deliberate constraints:

* **A field is a real variable.** ``fields`` accepts one typed array per column
  and the builder declares a :class:`~qphase.data.VariableSchema` for it — scalar,
  vector or matrix, with its own dims and dtype. A JSON blob standing in for
  numeric results would defeat the typed product contract, so an object array is
  rejected outright. ``method_metadata`` stays JSON because it records *how* the
  number was produced, not the number.
* **A negative result is a status, not a boolean.** Every row carries one
  :class:`~qphase_cam.data.outcomes.CAMOutcomeStatus`, so "we looked and found
  nothing" (``not_found``) is distinguishable from "we never looked"
  (``not_searched``) and from "it exists but is unstable" (``unstable``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from qphase.data import AxisRole, AxisSchema, Dataset, StatisticsDataset, VariableSchema

from ._common import closed_product, json_text_column, string_column, string_widths
from .declarations import CAM_ANNOTATION_PRODUCT, _annotation_variables
from .outcomes import (
    CAMOutcomeStatus,
    CAMTargetKind,
    coerce_outcome_status,
    coerce_target_kind,
)

__all__ = ["build_annotation_product"]

#: Column and axis names a typed field must not take.
RESERVED_FIELD_NAMES = frozenset(
    {"annotation", "target_identity", "status", "method_metadata"}
)

#: Dtypes a typed field may carry. Object arrays never reach a product.
FIELD_DTYPE_KINDS = frozenset("biufcUS")


def _annotation_targets(values: Sequence[str]) -> list[str]:
    """Return validated, unique, non-empty target identities."""
    targets = [str(value) for value in values]
    for position, value in enumerate(targets):
        if not value.strip():
            raise ValueError(
                f"target_identity[{position}] is empty; every annotation needs "
                "an explicit target"
            )
    seen: dict[str, int] = {}
    for value in targets:
        seen[value] = seen.get(value, 0) + 1
    duplicates = sorted(value for value, count in seen.items() if count > 1)
    if duplicates:
        raise ValueError(
            "target_identity must be unique within one annotation kind; "
            f"duplicated: {duplicates}"
        )
    return targets


def _annotation_statuses(
    status: Sequence[str] | str | CAMOutcomeStatus, n_annotation: int
) -> list[str]:
    """Return one validated outcome status per annotation."""
    if isinstance(status, (str, CAMOutcomeStatus)):
        values: list[Any] = [status] * n_annotation
    else:
        values = list(status)
        if len(values) != n_annotation:
            raise ValueError(
                f"status has {len(values)} entries for {n_annotation} annotations"
            )
    return [coerce_outcome_status(value).value for value in values]


def _field_array(name: str, value: Any, n_annotation: int) -> np.ndarray:
    """Return one typed field column."""
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(
            f"field {name!r} must have one entry per annotation; got a scalar"
        )
    if array.shape[0] != n_annotation:
        raise ValueError(
            f"field {name!r} has {array.shape[0]} rows for {n_annotation} annotations"
        )
    if array.dtype.hasobject:
        raise ValueError(
            f"field {name!r} is an object array; annotation fields must be "
            "typed, so declare the values as a real array instead of JSON"
        )
    if array.dtype.kind not in FIELD_DTYPE_KINDS:
        raise ValueError(
            f"field {name!r} has unsupported dtype {array.dtype}; expected a "
            "numeric, boolean or fixed-width string column"
        )
    return array


def _annotation_field_columns(
    fields: Mapping[str, Any] | None,
    *,
    field_axes: Mapping[str, Sequence[str]] | None,
    declared_axes: Mapping[str, int] | None,
    n_annotation: int,
) -> tuple[
    dict[str, np.ndarray],
    list[VariableSchema],
    list[dict[str, Any]],
    dict[str, int],
]:
    """Return the typed field columns, their variables and their declared axes."""
    arrays: dict[str, np.ndarray] = {}
    variables: list[VariableSchema] = []
    declared: list[dict[str, Any]] = []
    axis_sizes = {str(name): int(size) for name, size in (declared_axes or {}).items()}
    shared = {
        str(name): tuple(str(dim) for dim in dims)
        for name, dims in (field_axes or {}).items()
    }
    used: set[str] = set()
    for raw_name, value in (fields or {}).items():
        name = str(raw_name)
        if not name.isidentifier():
            raise ValueError(
                f"field name {name!r} must be a valid identifier: it becomes a "
                "product variable"
            )
        if name in RESERVED_FIELD_NAMES:
            raise ValueError(f"field name {name!r} is reserved by the product")
        array = _field_array(name, value, n_annotation)
        trailing = shared.get(name)
        if trailing is None:
            trailing = tuple(f"{name}_dim{index}" for index in range(array.ndim - 1))
        elif len(trailing) != array.ndim - 1:
            raise ValueError(
                f"field {name!r} declares {len(trailing)} trailing axes for "
                f"{array.ndim - 1} dimensions"
            )
        if len(set(trailing)) != len(trailing):
            raise ValueError(
                f"field {name!r} repeats a trailing axis name {trailing}; every "
                "dimension needs its own named axis"
            )
        for dim, size in zip(trailing, array.shape[1:], strict=True):
            known = axis_sizes.get(dim)
            if known is not None and known != int(size):
                raise ValueError(
                    f"field {name!r} uses axis {dim!r} with size {int(size)} but "
                    f"it was already declared with size {known}"
                )
            axis_sizes[dim] = int(size)
            used.add(dim)
        dtype = np.dtype(array.dtype)
        arrays[name] = array
        variables.append(
            VariableSchema(
                name=name,
                dtype=dtype.str,
                value_domain="complex" if dtype.kind == "c" else "real",
                dims=("annotation", *trailing),
            )
        )
        declared.append(
            {"name": name, "dtype": dtype.str, "dims": ["annotation", *trailing]}
        )
    unknown = sorted(set(shared) - set(arrays))
    if unknown:
        raise ValueError(f"field_axes names unknown fields: {unknown}")
    unused = sorted(name for name in axis_sizes if name not in used)
    if unused:
        raise ValueError(f"declared axes are not used by any field: {unused}")
    return arrays, variables, declared, axis_sizes


def _annotation_variables_with_widths(
    widths: Mapping[str, int],
) -> list[VariableSchema]:
    """Return the fixed annotation columns with data-dependent string widths."""
    variables: list[VariableSchema] = []
    for variable in _annotation_variables():
        width = widths.get(variable.name)
        if width is None:
            variables.append(variable)
            continue
        variables.append(
            variable.model_copy(update={"dtype": np.dtype(f"U{max(width, 1)}").str})
        )
    return variables


def build_annotation_product(
    *,
    annotation_kind: str,
    target_kind: str | CAMTargetKind,
    target_identity: Sequence[str],
    status: Sequence[str] | str | CAMOutcomeStatus,
    fields: Mapping[str, Any] | None = None,
    field_axes: Mapping[str, Sequence[str]] | None = None,
    axes: Mapping[str, int] | None = None,
    method_metadata: Sequence[Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Dataset:
    """Build one annotation product over a set of targets.

    ``fields`` maps a column name to a typed array whose leading dimension is the
    annotation axis: a 1-D array is a scalar-per-annotation column, a 2-D array a
    vector, a 3-D array a matrix. The builder declares the matching
    :class:`~qphase.data.VariableSchema` (dims ``("annotation", *trailing)``) and
    registers every field in the product attributes.

    Trailing axes are named after the field by default
    (``<field>_dim0``, ...). Pass ``field_axes`` to share a named axis between
    fields and ``axes`` to pin its size up front; the first declaration fixes the
    size and a later disagreement raises.

    ``target_identity`` must be non-empty and unique inside the product, and
    ``status`` must be a :class:`~qphase_cam.data.outcomes.CAMOutcomeStatus`
    value — either one value for every row or one per row.
    """
    kind = str(annotation_kind).strip()
    if not kind:
        raise ValueError("annotation_kind must not be empty")
    target_kind_value = coerce_target_kind(target_kind)
    targets = _annotation_targets(target_identity)
    n_annotation = len(targets)
    statuses = _annotation_statuses(status, n_annotation)
    if method_metadata is None:
        metadata_rows: list[Any] = [{} for _ in range(n_annotation)]
    else:
        metadata_rows = list(method_metadata)
        if len(metadata_rows) != n_annotation:
            raise ValueError(
                f"method_metadata has {len(metadata_rows)} entries for "
                f"{n_annotation} annotations"
            )
    field_arrays, field_variables, declared_fields, axis_sizes = (
        _annotation_field_columns(
            fields,
            field_axes=field_axes,
            declared_axes=axes,
            n_annotation=n_annotation,
        )
    )

    arrays: dict[str, np.ndarray] = {
        "target_identity": string_column(targets),
        "status": string_column(statuses),
        "method_metadata": json_text_column(metadata_rows),
        **field_arrays,
    }
    size_by_axis = {"annotation": n_annotation, **axis_sizes}
    extra_axes = [
        AxisSchema(name=name, role=AxisRole.INDEX, size=size_by_axis[name])
        for name in axis_sizes
    ]
    attributes = {
        **CAM_ANNOTATION_PRODUCT.attributes,
        "annotation_kind": kind,
        "target_kind": target_kind_value.value,
        "annotation_count": n_annotation,
        "field_variables": declared_fields,
        "field_axes": {spec["name"]: spec["dims"][1:] for spec in declared_fields},
        "shared_axes": dict(axis_sizes),
        "status_vocabulary": [member.value for member in CAMOutcomeStatus],
    }
    schema = closed_product(
        CAM_ANNOTATION_PRODUCT,
        sizes={"annotation": n_annotation},
        variables=[
            *_annotation_variables_with_widths(string_widths(arrays)),
            *field_variables,
        ],
        coordinates=[],
        attributes=attributes,
        extra_axes=extra_axes,
    )
    return StatisticsDataset.from_arrays(
        schema,
        arrays,
        owner="engine.cam",
        provenance=dict(provenance or {}),
    )
