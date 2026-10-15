"""Typed sampled-signal reader shared by spectral and Allan consumers.

One materialization boundary for the two sampled-signal families of Phase 6:

- complex field trajectories (``alpha``, quantity ``field_amplitude``),
  selected by physical mode index against the ``mode_indices`` attribute;
- real sampled observables (``value``, quantity ``sampled_observable``),
  selected by observable channel name against the ``channel_names``
  attribute.

Both paths propagate the regular time coordinate and the per-trajectory
``valid_length`` censoring of the source product. Consumers never treat
padding as observations: a point whose selected trajectories share one valid
length is truncated to it, and a point with mixed valid lengths fails fast —
variable-length records need a dedicated censoring design that Phase 6 does
not introduce. Arrays stay on their backend; only the small metadata and the
``valid_length`` vector are materialized to the host.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from qphase.backend.xputil import convert_to_numpy
from qphase.data import Dataset

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


def _time_coordinate(point: Dataset) -> tuple[float, float]:
    axis_names = {axis.name for axis in point.axes}
    time_axis = point.axis("time") if "time" in axis_names else None
    t0 = (
        float(time_axis.start)
        if time_axis is not None and time_axis.start is not None
        else 0.0
    )
    dt = (
        float(time_axis.step)
        if time_axis is not None and time_axis.step is not None
        else 1.0
    )
    return t0, dt


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
    if length > n_time:
        raise ValueError(
            f"valid_length {length} exceeds the time axis size {n_time}"
        )
    return length


def read_sampled_signal(
    point: Dataset,
    *,
    variable: Literal["alpha", "value"],
    channels: Sequence[int] | Sequence[str],
) -> SampledSignal:
    """Read one point's ``variable`` array resolved to selected channels.

    ``alpha`` selects the complex field by physical mode index;
    ``valid_length`` is optional there (its absence means every sample is an
    observation). ``value`` selects the real sampled observable by channel
    name and requires the ``valid_length`` variable of the
    sampled-observable contract.
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
    expected_dims = {
        ("trajectory", "time", "channel"),
        ("scan", "trajectory", "time", "channel"),
    }
    if tuple(schema.dims) not in expected_dims:
        raise TypeError(
            f"sampled-signal {variable} dims must be (scan?, trajectory, time, "
            f"channel), got {tuple(schema.dims)}"
        )
    array = point.handle(variable).materialize()
    if array.ndim == 4:
        if point.axis("scan").size != 1:
            raise ValueError(
                "sampled-signal point input must have a singleton scan axis"
            )
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            f"sampled-signal {variable} must have shape "
            f"(trajectory, time, channel), got {array.shape}"
        )
    t0, dt = _time_coordinate(point)
    n_traj, n_time, n_channel = (int(extent) for extent in array.shape)

    if all(isinstance(channel, str) for channel in channels):
        raw_names = point.attributes.get("channel_names")
        if not isinstance(raw_names, list | tuple) or not all(
            isinstance(name, str) for name in raw_names
        ):
            raise TypeError(
                "sampled-observable product is missing a string channel_names "
                "attribute"
            )
        names = [str(name) for name in raw_names]
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

    if "valid_length" in variables:
        length = _uniform_valid_length(point, n_traj, n_time)
        if length < n_time:
            array = array[:, :length, :]
    elif variable == "value":
        raise TypeError(
            "sampled-observable product is missing the valid_length variable"
        )

    return SampledSignal(
        array=array,
        t0=t0,
        dt=dt,
        labels=labels,
        columns=columns,
        variable_units=schema.units,
    )
