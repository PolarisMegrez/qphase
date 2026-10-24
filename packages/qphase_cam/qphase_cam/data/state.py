"""qphase_cam: the state and state-scan product builder
---------------------------------------------------------
One solved parameter point (``state``) or a scan of them (``state_scan``) as a
fixed-capacity product: ``states``/``residuals``/``success``/``valid_mask`` on
``scan`` x ``solution``, plus ``solution_count`` per scan point.

The ``solution`` axis is a capacity, not an identity. Two scan points that both
solve three states share three slots, and nothing here claims those states
continue into each other along the parameter axis — that is the continuation
product of Phase 10C2. The schema states this in its ``solution_semantics``
attribute so a reader of the artifact alone cannot misread it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from qphase.data import Dataset, StatisticsDataset, VariableSchema

from ._common import (
    closed_product,
    content_digest,
    coordinate_columns,
    string_widths,
)
from .declarations import (
    CAM_STATE_PRODUCT,
    CAM_STATE_SCAN_PRODUCT,
    _state_variables,
)

__all__ = ["SOLUTION_IDENTITY_SCHEME", "build_state_product", "solution_identity"]


#: Scheme marker mixed into a solution-identity payload.
SOLUTION_IDENTITY_SCHEME = "qphase_cam.solution_identity/1"


def solution_identity(
    *,
    state_vector: Any,
    parameters: Mapping[str, Any],
) -> str:
    """Return the content identity of one solved state at one parameter point.

    The digest covers the solved state matrix and the parameter values of its
    scan point. The solution slot is *not* part of it: two scan points that both
    solve three states share three slots, and those slots say nothing about which
    state is which.
    """
    payload = {
        "scheme": SOLUTION_IDENTITY_SCHEME,
        "parameters": dict(parameters),
        "state": np.asarray(state_vector, dtype=np.complex128),
    }
    return f"cam:solution:{content_digest(payload)}"


def _with_string_widths(
    variables: Sequence[VariableSchema], widths: Mapping[str, int]
) -> list[VariableSchema]:
    """Return the state columns with data-dependent string widths."""
    resolved: list[VariableSchema] = []
    for variable in variables:
        width = widths.get(variable.name)
        if width is None:
            resolved.append(variable)
            continue
        resolved.append(
            variable.model_copy(update={"dtype": np.dtype(f"U{max(width, 1)}").str})
        )
    return resolved


def build_state_product(
    *,
    states: Any,
    residuals: Any,
    success: Any,
    valid_mask: Any,
    solution_count: Any,
    parameters: Mapping[str, Any] | None = None,
    axes: Mapping[str, Any] | None = None,
    grid_shape: Sequence[int] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Dataset:
    """Build the state product of one parameter point or one scan.

    ``states`` spans ``grid_shape + (solution, state_row, state_col)``,
    ``residuals``/``success``/``valid_mask`` span ``grid_shape + (solution,)``
    and ``solution_count`` spans ``grid_shape``. An empty ``grid_shape`` is the
    single-point ``state`` product; anything else is ``state_scan``. The
    leading grid axes are flattened into the ``scan`` axis and ``grid_shape``
    is recorded as an attribute.

    A multi-dimensional grid is flattened rather than kept, because a product
    has exactly one parameter axis: the case grid is a *layout*, not data.

    Raises
    ------
    ValueError
        If a shape is inconsistent, a count leaves ``[0, capacity]``, or
        ``valid_mask`` does not flag exactly ``solution_count`` slots.

    """
    counts = np.asarray(solution_count, dtype=np.int64)
    shape = tuple(
        int(value) for value in (counts.shape if grid_shape is None else grid_shape)
    )
    if tuple(int(value) for value in counts.shape) != shape:
        raise ValueError(
            f"solution_count shape {tuple(counts.shape)} does not match the "
            f"grid shape {shape}"
        )
    scan_size = int(np.prod(shape, dtype=np.int64))
    residual_array = np.asarray(residuals, dtype=np.float64)
    if residual_array.ndim != len(shape) + 1:
        raise ValueError(
            f"residuals must span grid_shape + (solution,); got shape "
            f"{residual_array.shape} for grid shape {shape}"
        )
    capacity = int(residual_array.shape[-1])
    state_array = np.asarray(states, dtype=np.complex128)
    if state_array.ndim != len(shape) + 3:
        raise ValueError(
            f"states must span grid_shape + (solution, state_row, state_col); "
            f"got shape {state_array.shape} for grid shape {shape}"
        )
    matrix_shape = tuple(int(value) for value in state_array.shape[-2:])
    if state_array.shape != shape + (capacity, *matrix_shape):
        raise ValueError(
            f"states shape {state_array.shape} does not match grid shape "
            f"{shape}, solution capacity {capacity} and matrix shape "
            f"{matrix_shape}"
        )
    boolean_columns: dict[str, np.ndarray] = {}
    for name, value in (("success", success), ("valid_mask", valid_mask)):
        column = np.asarray(value, dtype=bool)
        if column.shape != shape + (capacity,):
            raise ValueError(
                f"{name} must span grid_shape + (solution,); got shape {column.shape}"
            )
        boolean_columns[name] = column
    counts_flat = counts.reshape(scan_size)
    valid_flat = boolean_columns["valid_mask"].reshape(scan_size, capacity)
    if counts_flat.size and (counts_flat.min() < 0 or counts_flat.max() > capacity):
        raise ValueError(f"solution_count must lie in [0, {capacity}]")
    if np.any(valid_flat.sum(axis=-1) != counts_flat):
        raise ValueError(
            "valid_mask must flag exactly solution_count slots at every scan point"
        )

    reserved: set[str] = set()
    parameter_arrays, parameter_variables, coordinates, unmapped_parameters = (
        coordinate_columns(
            parameters,
            prefix="parameter.",
            shape=shape,
            count=scan_size,
            reserved=reserved,
        )
    )
    axis_arrays, axis_variables, axis_coordinates, unmapped_axes = coordinate_columns(
        axes,
        prefix="axis.",
        shape=shape,
        count=scan_size,
        reserved=reserved,
    )
    coordinates = [*coordinates, *axis_coordinates]

    arrays: dict[str, np.ndarray] = {
        "states": state_array.reshape(scan_size, capacity, *matrix_shape),
        "residuals": residual_array.reshape(scan_size, capacity),
        "success": boolean_columns["success"].reshape(scan_size, capacity),
        "valid_mask": valid_flat,
        "solution_count": counts_flat,
        **parameter_arrays,
        **axis_arrays,
    }
    # One identity per filled slot; an empty slot carries an empty string so the
    # column never implies that a slot exists as a solution.
    identity_rows: list[list[str]] = [[""] * capacity for _ in range(scan_size)]
    for point in range(scan_size):
        point_parameters = {
            name.removeprefix("parameter."): float(np.asarray(values[point]).real)
            for name, values in arrays.items()
            if name.startswith("parameter.")
        }
        for slot in range(int(counts_flat[point])):
            identity_rows[point][slot] = solution_identity(
                state_vector=arrays["states"][point, slot],
                parameters=point_parameters,
            )
    # Two slots with the same state *and* the same parameter values are the same
    # solution, so they legitimately share an identity: the digest is content, not
    # position. The only invariant is that a slot holds an identity exactly when
    # it holds a solution.
    if any(
        bool(identity_rows[point][slot]) != bool(valid_flat[point, slot])
        for point in range(scan_size)
        for slot in range(capacity)
    ):
        raise ValueError(
            "solution_identity must be set exactly on valid solution slots"
        )
    identity_width = max(
        (len(value) for row in identity_rows for value in row), default=0
    )
    arrays["solution_identity"] = np.asarray(
        identity_rows, dtype=f"U{max(identity_width, 1)}"
    ).reshape(scan_size, capacity)
    template = CAM_STATE_PRODUCT if not shape else CAM_STATE_SCAN_PRODUCT
    attributes = {
        **template.attributes,
        "grid_shape": [int(value) for value in shape],
        "scan_size": scan_size,
        "solution_capacity": capacity,
        "state_shape": [int(value) for value in matrix_shape],
    }
    schema = closed_product(
        template,
        sizes={
            "scan": scan_size,
            "solution": capacity,
            "state_row": matrix_shape[0],
            "state_col": matrix_shape[1],
        },
        variables=[
            *_with_string_widths(_state_variables(), string_widths(arrays)),
            *parameter_variables,
            *axis_variables,
        ],
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
