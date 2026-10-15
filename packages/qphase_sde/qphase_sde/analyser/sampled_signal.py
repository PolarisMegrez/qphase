"""Typed sampled-signal reader shared by spectral and Allan consumers.

One materialization boundary for the two sampled-signal families of Phase 6:

- complex field trajectories (``alpha``, quantity ``field_amplitude``),
  selected by physical mode index against the ``mode_indices`` attribute;
- real sampled observables (``value``, quantity ``sampled_observable``),
  selected by observable channel name against the ``channel_names``
  attribute.

Both product families formally declare a regular ``time`` coordinate axis
(``units="time"``) and a per-trajectory ``valid_length`` variable (int64,
``quantity="valid_length"``, dims ``(scan?, trajectory)``); the reader enforces
the full contract at this boundary instead of silently defaulting anything.
Consumers never treat padding as observations: a point whose selected
trajectories share one valid length is truncated to it, and a point with mixed
valid lengths fails fast — variable-length records need a dedicated censoring
design that Phase 6 does not introduce. Arrays stay on their backend; only the
small metadata and the ``valid_length`` vector are materialized to the host.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from qphase.backend.xputil import convert_to_numpy
from qphase.data import AxisRole, AxisSchema, Dataset, VariableSchema

from ..contracts.quantities import SDEQuantity
from .base import select_mode_columns
from .observable_transform import validate_mode_indices

__all__ = ["SampledSignal", "read_sampled_signal"]


@dataclass(frozen=True)
class SampledSignal:
    """One point's selected sampled signal plus its coordinate metadata."""

    array: Any
    t0: float
    dt: float
    labels: tuple[int | str, ...]
    columns: tuple[int, ...]
    variable_units: str


def _axis(point: Dataset, name: str) -> AxisSchema:
    """Return the named axis or fail: dims alone never imply axis metadata."""
    for axis in point.axes:
        if axis.name == name:
            return axis
    raise TypeError(f"sampled-signal product is missing the {name} axis")


def _check_axis_role_size(
    point: Dataset, name: str, role: AxisRole, size: int
) -> AxisSchema:
    axis = _axis(point, name)
    if axis.role is not role:
        raise TypeError(
            f"sampled-signal {name} axis role must be {role.value!r}, "
            f"got {axis.role.value!r}"
        )
    if axis.size is not None and axis.size != size:
        raise ValueError(
            f"sampled-signal {name} axis size {axis.size} does not match "
            f"the payload extent {size}"
        )
    return axis


def _time_coordinate(point: Dataset, n_time: int) -> tuple[float, float]:
    """Return the regular time coordinate; missing or malformed axes fail."""
    axis = _check_axis_role_size(point, "time", AxisRole.COORDINATE, n_time)
    if axis.coordinate != "regular":
        raise TypeError(
            "sampled-signal time axis must be a regular coordinate, got "
            f"{axis.coordinate!r}"
        )
    if axis.units != "time":
        raise TypeError(
            f"sampled-signal time axis units must be 'time', got {axis.units!r}"
        )
    if axis.start is None or not np.isfinite(axis.start):
        raise ValueError("sampled-signal time axis must declare a finite start")
    if axis.step is None or not np.isfinite(axis.step) or axis.step <= 0:
        raise ValueError(
            "sampled-signal time axis must declare a finite, positive step"
        )
    return float(axis.start), float(axis.step)


def _valid_length_variable(
    point: Dataset,
    variables: dict[str, VariableSchema],
    *,
    has_scan: bool,
) -> VariableSchema:
    """Return the validated ``valid_length`` schema of either product family."""
    schema = variables.get("valid_length")
    if schema is None:
        raise TypeError("sampled-signal product is missing the valid_length variable")
    expected_dims = ("scan", "trajectory") if has_scan else ("trajectory",)
    if tuple(schema.dims) != expected_dims:
        raise TypeError(
            f"valid_length dims must be {expected_dims}, got {tuple(schema.dims)}"
        )
    if schema.quantity != "valid_length":
        raise TypeError(
            f"valid_length quantity must be 'valid_length', got "
            f"{schema.quantity!r}"
        )
    if schema.value_domain != "real":
        raise TypeError("valid_length must be real-valued")
    if np.dtype(schema.dtype) != np.dtype(np.int64):
        raise TypeError(
            f"valid_length dtype must be int64, got {schema.dtype!r}"
        )
    return schema


def _uniform_valid_length(point: Dataset, n_traj: int, n_time: int) -> int:
    """Return the shared valid length of one point, truncating padding away.

    Every selected trajectory of the point must report the same valid length;
    mixed lengths are variable-length records, which Phase 6 consumers reject
    instead of silently truncating or zero-filling.
    """
    lengths = convert_to_numpy(point.handle("valid_length").materialize())
    lengths = lengths.reshape(-1)
    if lengths.size != n_traj:
        raise ValueError(
            f"valid_length spans {lengths.size} trajectories, expected {n_traj}"
        )
    unique = np.unique(lengths)
    if unique.size != 1:
        raise ValueError(
            "variable-length records are not supported: selected trajectories "
            f"report different valid_length values {unique.tolist()}"
        )
    length = int(unique[0])
    if length < 1:
        raise ValueError(
            f"valid_length {length} is not a positive sample count"
        )
    if length > n_time:
        raise ValueError(
            f"valid_length {length} exceeds the time axis size {n_time}"
        )
    return length


def _observable_channel_names(point: Dataset, n_channel: int) -> list[str]:
    """Return the validated channel identity of a sampled-observable product."""
    raw_names = point.attributes.get("channel_names")
    if (
        not isinstance(raw_names, list | tuple)
        or not raw_names
        or not all(isinstance(name, str) and name for name in raw_names)
    ):
        raise TypeError(
            "sampled-observable product is missing a non-empty string "
            "channel_names attribute"
        )
    names = [str(name) for name in raw_names]
    if len(names) != n_channel:
        raise ValueError(
            f"channel_names records {len(names)} names but the channel axis "
            f"has size {n_channel}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"channel_names must be unique, got {names}")
    return names


def read_sampled_signal(
    point: Dataset,
    *,
    variable: Literal["alpha", "value"],
    channels: Sequence[int] | Sequence[str],
) -> SampledSignal:
    """Read one point's ``variable`` array resolved to selected channels.

    ``alpha`` selects the complex field by physical mode index, ``value`` the
    real sampled observable by channel name. Both paths require the
    ``valid_length`` variable of the trajectory and sampled-observable
    contracts and a regular ``time`` coordinate axis; missing or malformed
    contract metadata fails fast instead of falling back to defaults.
    """
    if variable == "alpha":
        expected_quantity = SDEQuantity.FIELD_AMPLITUDE.value
        expected_domain = "complex"
    elif variable == "value":
        expected_quantity = SDEQuantity.SAMPLED_OBSERVABLE.value
        expected_domain = "real"
    else:
        raise ValueError(f"unsupported sampled-signal variable: {variable!r}")
    variables = {item.name: item for item in point.variables}
    schema = variables.get(variable)
    if schema is None:
        raise TypeError(f"sampled-signal product is missing the {variable} variable")
    if schema.quantity != expected_quantity:
        raise TypeError(
            f"sampled-signal {variable} quantity must be {expected_quantity!r}, "
            f"got {schema.quantity!r}"
        )
    if schema.value_domain != expected_domain:
        raise TypeError(
            f"sampled-signal {variable} must be {expected_domain}-valued, "
            f"got {schema.value_domain}"
        )
    dtype = np.dtype(schema.dtype)
    if variable == "value" and dtype.kind != "f":
        raise TypeError(
            f"sampled-signal value must use a real floating dtype, got "
            f"{schema.dtype!r}"
        )
    expected_dims = {
        ("trajectory", "time", "channel"),
        ("scan", "trajectory", "time", "channel"),
    }
    if tuple(schema.dims) not in expected_dims:
        raise TypeError(
            f"sampled-signal {variable} dims must be (scan?, trajectory, time, "
            f"channel), got {tuple(schema.dims)}"
        )
    has_scan = "scan" in schema.dims
    _valid_length_variable(point, variables, has_scan=has_scan)

    array = point.handle(variable).materialize()
    if np.dtype(array.dtype) != dtype:
        raise TypeError(
            f"sampled-signal {variable} payload dtype {array.dtype!r} does "
            f"not match the schema dtype {schema.dtype!r}"
        )
    if has_scan:
        if array.ndim != 4:
            raise ValueError(
                f"sampled-signal {variable} dims declare a scan axis but the "
                f"payload has shape {array.shape}"
            )
        scan_axis = _axis(point, "scan")
        if scan_axis.role is not AxisRole.PARAMETER:
            raise TypeError(
                f"sampled-signal scan axis role must be 'parameter', got "
                f"{scan_axis.role.value!r}"
            )
        if scan_axis.size != 1:
            raise ValueError(
                "sampled-signal point input must have a singleton scan axis"
            )
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            f"sampled-signal {variable} must have shape "
            f"(trajectory, time, channel), got {array.shape}"
        )
    n_traj, n_time, n_channel = (int(extent) for extent in array.shape)
    _check_axis_role_size(point, "trajectory", AxisRole.REALIZATION, n_traj)
    _check_axis_role_size(point, "channel", AxisRole.COMPONENT, n_channel)
    t0, dt = _time_coordinate(point, n_time)

    if all(isinstance(channel, str) for channel in channels):
        names = _observable_channel_names(point, n_channel)
        name_map = {name: index for index, name in enumerate(names)}
        missing = [channel for channel in channels if channel not in name_map]
        if missing:
            raise ValueError(
                f"requested channels were not recorded: {missing}; available "
                f"channels are {names}"
            )
        labels: tuple[int | str, ...] = tuple(str(channel) for channel in channels)
        columns = tuple(name_map[str(channel)] for channel in channels)
    elif all(
        isinstance(channel, int | np.integer)
        and not isinstance(channel, bool | np.bool_)
        for channel in channels
    ):
        recorded = validate_mode_indices(
            point.attributes.get("mode_indices"), n_channel
        )
        modes, mode_columns = select_mode_columns(
            recorded, [int(channel) for channel in channels]
        )
        labels = tuple(modes)
        columns = tuple(mode_columns)
    else:
        raise TypeError(
            "sampled-signal channels must be all physical mode indices or all "
            "observable channel names"
        )

    length = _uniform_valid_length(point, n_traj, n_time)
    if length < n_time:
        array = array[:, :length, :]

    return SampledSignal(
        array=array,
        t0=t0,
        dt=dt,
        labels=labels,
        columns=columns,
        variable_units=schema.units,
    )
