"""qphase_cam: the continuation product builder
---------------------------------------------------------
One or more *local* continuation paths as a ragged product: a ``point`` axis over
the flattened path blocks, a ``path``/``path_offset`` pairing and a
``path_offsets`` block of length ``n_path + 1``. An empty path (a continuation
that stopped before its first corrected point) and an empty product are
representable: the offsets say so, so nothing has to be inferred from a zero
length.

``path_identity`` is explicit and required. A path slot is not a branch id: two
paths are not claims about one global branch, and a solver that owns the real
identity supplies it (Phase 10D). The builder refuses to invent one from the slot
position, exactly like the candidate builder refuses to invent a candidate
identity.

The ``point`` axis is the product's parameter axis: along a path every point is a
distinct parameter point, so ``parameter.<name>`` columns and the
``continuation_coordinate`` are parameter coordinates and
``GenericDataBundle.axes``/``point_view`` work without a separate scan axis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from qphase.data import CoordinateSchema, Dataset, StatisticsDataset, VariableSchema

from ._common import (
    closed_product,
    coordinate_columns,
    json_text_column,
    string_column,
    string_widths,
)
from .declarations import CAM_CONTINUATION_PRODUCT, _continuation_variables

__all__ = ["build_continuation_product"]


def _path_states(values: Any, n_point: int) -> np.ndarray:
    """Return continuation states as ``(point, state_row, state_col)``."""
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim == 1 and array.size == 0 and n_point == 0:
        return np.zeros((0, 0, 0), dtype=np.complex128)
    if array.ndim != 3:
        raise ValueError(
            f"states must span (point, state_row, state_col); got shape {array.shape}"
        )
    if array.shape[0] != n_point:
        raise ValueError(f"states has {array.shape[0]} rows for {n_point} points")
    return array


def _point_column(values: Any, n_point: int, name: str, dtype: Any) -> np.ndarray:
    """Return one value per continuation point."""
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1 or array.shape[0] != n_point:
        raise ValueError(
            f"{name} must have one entry per point; got shape {array.shape}"
        )
    return array


def _path_offsets(values: Any | None, n_point: int) -> np.ndarray:
    """Return a validated ``path_offsets`` block.

    ``[0]`` is legal: it describes a product with no path at all, which is how a
    continuation that never started reports itself. ``[0, 0]`` is one path with no
    point.
    """
    if values is None:
        return np.asarray([0, n_point], dtype=np.int64)
    offsets = np.asarray(values, dtype=np.int64)
    if offsets.ndim != 1 or offsets.size < 1:
        raise ValueError(
            "path_offsets must be one-dimensional with n_path + 1 entries; got "
            f"shape {offsets.shape}"
        )
    if int(offsets[0]) != 0:
        raise ValueError("path_offsets must start at 0")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("path_offsets must be monotonic non-decreasing")
    if int(offsets[-1]) != n_point:
        raise ValueError(
            f"path_offsets must end at the point count {n_point}; got "
            f"{int(offsets[-1])}"
        )
    return offsets


def _path_identity(values: Sequence[str]) -> list[str]:
    """Return validated, unique, non-empty path identities."""
    identity = [str(value) for value in values]
    for position, value in enumerate(identity):
        if not value.strip():
            raise ValueError(
                f"path_identity[{position}] is empty; every path needs an "
                "explicit identity"
            )
    seen: dict[str, int] = {}
    for value in identity:
        seen[value] = seen.get(value, 0) + 1
    duplicates = sorted(value for value, count in seen.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"path_identity must be unique across the product; duplicated: {duplicates}"
        )
    return identity


def _drop_coordinate_source(
    mapping: Mapping[str, Any] | None,
    *,
    name: str,
    coordinate: np.ndarray,
    source: str,
) -> tuple[dict[str, Any], str]:
    """Remove the swept parameter from one mapping, checking consistency.

    The swept parameter must not appear twice: the ``continuation_coordinate``
    column already carries it, so the same name is removed from ``parameters``
    and from ``axes``. Three outcomes are distinguished, because "not a per-point
    column" and "a contradictory per-point column" are different facts:

    ``absent``
        The mapping does not carry the swept parameter.
    ``duplicate``
        It carries the matching per-point values; removing them loses nothing.
    ``not_per_point``
        It carries a scalar or a differently shaped value (a legacy result keeps
        the model's configured value there). It cannot become a point column, so
        the caller reports the name as unmapped instead of dropping it silently.
    """
    filtered = {str(key): value for key, value in (mapping or {}).items()}
    if name not in filtered:
        return filtered, "absent"
    values = np.asarray(filtered.pop(name))
    if values.shape != coordinate.shape:
        return filtered, "not_per_point"
    if not np.allclose(values.astype(np.float64), coordinate, equal_nan=True):
        raise ValueError(
            f"{source} entry {name!r} disagrees with continuation_coordinate; the "
            "coordinate already carries the swept parameter"
        )
    return filtered, "duplicate"


def build_continuation_product(
    *,
    path_identity: Sequence[str],
    continuation_coordinate: Any,
    states: Any,
    residuals: Any,
    success: Any,
    path_offsets: Any | None = None,
    continuation_coordinate_name: str = "continuation_coordinate",
    events: Sequence[Any] | None = None,
    parameters: Mapping[str, Any] | None = None,
    axes: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Dataset:
    """Build the ragged continuation product of one or more local paths.

    ``continuation_coordinate`` is the swept parameter value at every point and
    is the anchor of the payload: ``states`` spans
    ``(point, state_row, state_col)`` and ``residuals``/``success``/``events``
    span ``(point,)``.

    ``path_offsets`` defaults to ``[0, n_point]`` (one path). When given it must
    have ``n_path + 1`` entries, start at zero, be monotonic non-decreasing and
    end at the point count, and ``len(path_identity)`` must equal ``n_path``. A
    repeated offset is an empty path, which is legal; a product with no path at
    all is ``path_offsets=[0]`` and ``path_identity=[]``.

    ``parameters`` carries the model parameters per point. The entry named
    ``continuation_coordinate_name`` is excluded (the coordinate already carries
    it) and must agree with ``continuation_coordinate`` when present. ``axes``
    is the legacy solver's axis mapping and is carried the same way.

    Raises
    ------
    ValueError
        If an identity is empty or duplicated, an offset block is
        inconsistent, a column does not cover the point axis, or a parameter
        disagrees with the continuation coordinate.

    """
    coordinate = np.asarray(continuation_coordinate, dtype=np.float64)
    if coordinate.ndim != 1:
        raise ValueError(
            "continuation_coordinate must be one-dimensional over the point "
            f"axis; got shape {coordinate.shape}"
        )
    n_point = int(coordinate.size)
    identity = _path_identity(path_identity)
    offsets = _path_offsets(path_offsets, n_point)
    n_path = int(offsets.size) - 1
    if n_path != len(identity):
        raise ValueError(
            f"path_offsets describes {n_path} paths but path_identity has "
            f"{len(identity)} entries"
        )
    name = str(continuation_coordinate_name).strip()
    if not name:
        raise ValueError("continuation_coordinate_name must not be empty")

    state_array = _path_states(states, n_point)
    residual_array = _point_column(residuals, n_point, "residuals", np.float64)
    success_array = _point_column(success, n_point, "success", bool)
    if events is None:
        event_rows: list[Any] = [{} for _ in range(n_point)]
    else:
        event_rows = list(events)
        if len(event_rows) != n_point:
            raise ValueError(
                f"events has {len(event_rows)} entries for {n_point} points"
            )

    filtered_parameters, from_parameters = _drop_coordinate_source(
        parameters, name=name, coordinate=coordinate, source="parameter"
    )
    filtered_axes, from_axes = _drop_coordinate_source(
        axes, name=name, coordinate=coordinate, source="axis"
    )
    reserved: set[str] = set()
    (
        parameter_arrays,
        parameter_variables,
        parameter_coordinates,
        unmapped_parameters,
    ) = coordinate_columns(
        filtered_parameters,
        prefix="parameter.",
        shape=(n_point,),
        count=n_point,
        reserved=reserved,
        dims=("point",),
        quantity="continuation_parameter",
    )
    axis_arrays, axis_variables, axis_coordinates, unmapped_axes = coordinate_columns(
        filtered_axes,
        prefix="axis.",
        shape=(n_point,),
        count=n_point,
        reserved=reserved,
        dims=("point",),
        quantity="continuation_parameter",
    )
    if from_parameters == "not_per_point":
        unmapped_parameters = sorted([*unmapped_parameters, f"parameter.{name}"])
    if from_axes == "not_per_point":
        unmapped_axes = sorted([*unmapped_axes, f"axis.{name}"])

    arrays: dict[str, np.ndarray] = {
        "path_identity": string_column(identity),
        "path_offsets": offsets,
        "continuation_coordinate": coordinate,
        "states": state_array,
        "residuals": residual_array,
        "success": success_array,
        "events": json_text_column(event_rows),
        **parameter_arrays,
        **axis_arrays,
    }
    # The continuation coordinate is registered last so that it is the parameter
    # coordinate `GenericDataBundle.axes` reports for the point axis.
    coordinates = [
        *parameter_coordinates,
        *axis_coordinates,
        CoordinateSchema(
            name=name,
            variable="continuation_coordinate",
            dims=("point",),
            role="parameter",
        ),
    ]
    variables = [
        *_continuation_variables_with_widths(string_widths(arrays)),
        *parameter_variables,
        *axis_variables,
    ]
    attributes = {
        **CAM_CONTINUATION_PRODUCT.attributes,
        "path_count": n_path,
        "point_count": n_point,
        "state_shape": [int(value) for value in state_array.shape[1:]],
        "continuation_coordinate_name": name,
        "continuation_parameter_from_parameters": from_parameters,
        "continuation_parameter_from_axes": from_axes,
    }
    schema = closed_product(
        CAM_CONTINUATION_PRODUCT,
        sizes={
            "point": n_point,
            "path": n_path,
            "path_offset": n_path + 1,
            "state_row": int(state_array.shape[1]),
            "state_col": int(state_array.shape[2]),
        },
        variables=variables,
        coordinates=coordinates,
        attributes=attributes,
    )
    return StatisticsDataset.from_arrays(
        schema,
        arrays,
        owner="engine.cam",
        provenance={
            **dict(provenance or {}),
            "unmapped_parameter_names": unmapped_parameters,
            "unmapped_axis_names": unmapped_axes,
        },
    )


def _continuation_variables_with_widths(
    widths: Mapping[str, int],
) -> list[VariableSchema]:
    """Return the continuation columns with data-dependent string widths."""
    variables: list[VariableSchema] = []
    for variable in _continuation_variables():
        width = widths.get(variable.name)
        if width is None:
            variables.append(variable)
            continue
        variables.append(
            variable.model_copy(update={"dtype": np.dtype(f"U{max(width, 1)}").str})
        )
    return variables
