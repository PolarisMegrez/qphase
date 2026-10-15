"""Boundary tests for the shared sampled-signal reader.

The reader is the single materialization boundary of the spectral and Allan
consumers: it resolves physical mode indices (``alpha``) or observable
channel names (``value``), propagates the regular time coordinate, and
enforces the ``valid_length`` censoring contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ProductSchema,
    TimeSeriesDataset,
    VariableSchema,
)
from qphase_sde.analyser.sampled_signal import read_sampled_signal
from qphase_sde.contracts.quantities import SDEQuantity

N_TRAJ = 3
N_TIME = 10
T0 = 1.5
DT = 0.5


def _axes(n_channel: int = 2, scan: int | None = None) -> list[AxisSchema]:
    axes: list[AxisSchema] = []
    if scan is not None:
        axes.append(AxisSchema(name="scan", role=AxisRole.PARAMETER, size=scan))
    axes.extend(
        [
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=N_TRAJ),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=N_TIME,
                coordinate="regular",
                start=T0,
                step=DT,
                units="time",
            ),
            AxisSchema(name="channel", role=AxisRole.COMPONENT, size=n_channel),
        ]
    )
    return axes


def _field_product(
    *,
    n_channel: int = 2,
    scan: int | None = None,
    mode_indices: object = None,
    valid_length: np.ndarray | None = None,
    quantity: str = SDEQuantity.FIELD_AMPLITUDE.value,
    domain: str = "complex",
    dtype: str = "<c16",
    dims: tuple[str, ...] | None = None,
    units: str = "sqrt(inverse_time)",
):
    axes = _axes(n_channel, scan)
    rng = np.random.default_rng(20261213)
    leading = ("scan",) if scan is not None else ()
    alpha_dims = dims or (*leading, "trajectory", "time", "channel")
    shape = tuple(
        {"scan": scan, "trajectory": N_TRAJ, "time": N_TIME, "channel": n_channel}[
            name
        ]
        for name in alpha_dims
    )
    alpha = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    if domain == "real":
        alpha = alpha.real
    variables = [
        VariableSchema(
            name="alpha",
            dtype="<f8" if domain == "real" else dtype,
            value_domain=domain,
            dims=alpha_dims,
            quantity=quantity,
            units=units,
        )
    ]
    arrays: dict[str, np.ndarray] = {"alpha": np.asarray(alpha)}
    if valid_length is not None:
        variables.append(
            VariableSchema(
                name="valid_length",
                dtype="<i8",
                value_domain="real",
                dims=(*leading, "trajectory"),
                quantity="valid_length",
            )
        )
        arrays["valid_length"] = np.asarray(valid_length, dtype=np.int64)
    attributes = {}
    if mode_indices is not None:
        attributes["mode_indices"] = mode_indices
    return TimeSeriesDataset.from_arrays(
        ProductSchema(
            kind=DataKind.TIME_SERIES,
            axes=axes,
            variables=variables,
            attributes=attributes,
        ),
        arrays,
        owner="test",
    )


def _observable_product(
    *,
    n_channel: int = 2,
    channel_names: object = ("n0", "n1"),
    valid_length: np.ndarray | None = None,
    quantity: str = SDEQuantity.SAMPLED_OBSERVABLE.value,
    domain: str = "real",
    units: str = "occupation",
):
    axes = _axes(n_channel)
    rng = np.random.default_rng(20261213)
    value = rng.standard_normal((N_TRAJ, N_TIME, n_channel))
    variables = [
        VariableSchema(
            name="value",
            dtype="<f8",
            value_domain=domain,
            dims=("trajectory", "time", "channel"),
            quantity=quantity,
            units=units,
        )
    ]
    arrays: dict[str, np.ndarray] = {"value": value}
    if valid_length is not None:
        variables.append(
            VariableSchema(
                name="valid_length",
                dtype="<i8",
                value_domain="real",
                dims=("trajectory",),
                quantity="valid_length",
            )
        )
        arrays["valid_length"] = np.asarray(valid_length, dtype=np.int64)
    attributes = {}
    if channel_names is not None:
        attributes["channel_names"] = channel_names
    return TimeSeriesDataset.from_arrays(
        ProductSchema(
            kind=DataKind.TIME_SERIES,
            axes=axes,
            variables=variables,
            attributes=attributes,
        ),
        arrays,
        owner="test",
    )


def test_alpha_resolves_physical_modes_against_recorded_indices():
    product = _field_product(mode_indices=[2, 5])
    signal = read_sampled_signal(product, variable="alpha", channels=[5, 2])
    assert signal.labels == (5, 2)
    assert signal.columns == (1, 0)
    source = np.asarray(product.handle("alpha").materialize())
    np.testing.assert_array_equal(signal.array, source)
    selected = signal.array[:, :, list(signal.columns)]
    np.testing.assert_array_equal(selected, source[:, :, [1, 0]])
    assert signal.t0 == T0
    assert signal.dt == DT
    assert signal.variable_units == "sqrt(inverse_time)"


def test_alpha_defaults_to_identity_mode_indices():
    product = _field_product()
    signal = read_sampled_signal(product, variable="alpha", channels=[1])
    assert signal.labels == (1,)
    assert signal.columns == (1,)


def test_alpha_accepts_numpy_integer_channels():
    product = _field_product()
    signal = read_sampled_signal(product, variable="alpha", channels=[np.int64(1)])
    assert signal.labels == (1,)
    assert isinstance(signal.labels[0], int)


def test_alpha_rejects_missing_variable():
    product = _observable_product()
    with pytest.raises(TypeError, match="missing the alpha variable"):
        read_sampled_signal(product, variable="alpha", channels=[0])


def test_alpha_rejects_wrong_quantity():
    product = _field_product(quantity="wrong_quantity")
    with pytest.raises(TypeError, match="quantity"):
        read_sampled_signal(product, variable="alpha", channels=[0])


def test_alpha_rejects_real_domain():
    product = _field_product(domain="real")
    with pytest.raises(TypeError, match="complex-valued"):
        read_sampled_signal(product, variable="alpha", channels=[0])


def test_alpha_rejects_permuted_dims():
    product = _field_product(dims=("time", "trajectory", "channel"))
    with pytest.raises(TypeError, match="dims"):
        read_sampled_signal(product, variable="alpha", channels=[0])


def test_alpha_rejects_unrecorded_mode():
    product = _field_product(mode_indices=[2, 5])
    with pytest.raises(ValueError, match="requested modes were not recorded"):
        read_sampled_signal(product, variable="alpha", channels=[3])


@pytest.mark.parametrize(
    ("mode_indices", "match"),
    [
        ([-1, 2], "non-negative"),
        ([2, 2], "unique"),
        ([2, 5, 7], "channel axis size"),
        ([True, 5], "list of integers"),
    ],
)
def test_alpha_rejects_invalid_recorded_mode_indices(mode_indices, match):
    product = _field_product(mode_indices=mode_indices, n_channel=2)
    with pytest.raises((TypeError, ValueError), match=match):
        read_sampled_signal(product, variable="alpha", channels=[2])


def test_alpha_rejects_mixed_channel_types():
    product = _field_product()
    with pytest.raises(TypeError, match="all physical mode indices"):
        read_sampled_signal(product, variable="alpha", channels=[0, "n1"])


def test_alpha_rejects_bool_and_float_channels():
    product = _field_product()
    with pytest.raises(TypeError, match="all physical mode indices"):
        read_sampled_signal(product, variable="alpha", channels=[True])
    with pytest.raises(TypeError, match="all physical mode indices"):
        read_sampled_signal(product, variable="alpha", channels=[1.0])


def test_alpha_squeezes_singleton_scan_axis():
    product = _field_product(scan=1)
    signal = read_sampled_signal(product, variable="alpha", channels=[0])
    assert signal.array.shape == (N_TRAJ, N_TIME, 2)


def test_alpha_rejects_non_singleton_scan_axis():
    product = _field_product(scan=2)
    with pytest.raises(ValueError, match="singleton scan"):
        read_sampled_signal(product, variable="alpha", channels=[0])


def test_alpha_without_valid_length_keeps_full_series():
    product = _field_product()
    signal = read_sampled_signal(product, variable="alpha", channels=[0])
    assert signal.array.shape == (N_TRAJ, N_TIME, 2)


def test_valid_length_truncates_padding():
    product = _field_product(valid_length=np.full(N_TRAJ, 6))
    signal = read_sampled_signal(product, variable="alpha", channels=[0])
    source = np.asarray(product.handle("alpha").materialize())
    np.testing.assert_array_equal(signal.array, source[:, :6, :])


def test_valid_length_rejects_mixed_lengths():
    product = _field_product(valid_length=np.array([6, 8, 8]))
    with pytest.raises(ValueError, match="variable-length records"):
        read_sampled_signal(product, variable="alpha", channels=[0])


def test_valid_length_rejects_length_beyond_time_axis():
    product = _field_product(valid_length=np.full(N_TRAJ, N_TIME + 1))
    with pytest.raises(ValueError, match="exceeds the time axis"):
        read_sampled_signal(product, variable="alpha", channels=[0])


def test_value_resolves_channel_names():
    product = _observable_product(valid_length=np.full(N_TRAJ, N_TIME))
    signal = read_sampled_signal(product, variable="value", channels=["n1"])
    assert signal.labels == ("n1",)
    assert signal.columns == (1,)
    source = np.asarray(product.handle("value").materialize())
    np.testing.assert_array_equal(signal.array, source)
    np.testing.assert_array_equal(signal.array[:, :, 1], source[:, :, 1])
    assert signal.variable_units == "occupation"


def test_value_rejects_missing_channel_names():
    product = _observable_product(
        channel_names=None, valid_length=np.full(N_TRAJ, N_TIME)
    )
    with pytest.raises(TypeError, match="channel_names"):
        read_sampled_signal(product, variable="value", channels=["n0"])


def test_value_rejects_unrecorded_channel_name():
    product = _observable_product(valid_length=np.full(N_TRAJ, N_TIME))
    with pytest.raises(ValueError, match="available"):
        read_sampled_signal(product, variable="value", channels=["n9"])


def test_value_requires_valid_length():
    product = _observable_product()
    with pytest.raises(TypeError, match="valid_length"):
        read_sampled_signal(product, variable="value", channels=["n0"])


def test_value_rejects_wrong_quantity():
    product = _observable_product(
        quantity="wrong_quantity", valid_length=np.full(N_TRAJ, N_TIME)
    )
    with pytest.raises(TypeError, match="quantity"):
        read_sampled_signal(product, variable="value", channels=["n0"])


def test_value_rejects_complex_domain():
    axes = _axes(2)
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=axes,
        variables=[
            VariableSchema(
                name="value",
                dtype="<c16",
                value_domain="complex",
                dims=("trajectory", "time", "channel"),
                quantity=SDEQuantity.SAMPLED_OBSERVABLE.value,
            ),
            VariableSchema(
                name="valid_length",
                dtype="<i8",
                value_domain="real",
                dims=("trajectory",),
                quantity="valid_length",
            ),
        ],
        attributes={"channel_names": ["n0", "n1"]},
    )
    product = TimeSeriesDataset.from_arrays(
        schema,
        {
            "value": np.zeros((N_TRAJ, N_TIME, 2), dtype=np.complex128),
            "valid_length": np.full(N_TRAJ, N_TIME, dtype=np.int64),
        },
        owner="test",
    )
    with pytest.raises(TypeError, match="real-valued"):
        read_sampled_signal(product, variable="value", channels=["n0"])


def test_value_truncates_by_valid_length():
    product = _observable_product(valid_length=np.full(N_TRAJ, 4))
    signal = read_sampled_signal(product, variable="value", channels=["n0", "n1"])
    assert signal.array.shape == (N_TRAJ, 4, 2)


def test_time_coordinate_defaults_without_start_step():
    axes = _axes()
    axes[1] = AxisSchema(name="time", role=AxisRole.COORDINATE, size=N_TIME)
    rng = np.random.default_rng(0)
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=axes,
        variables=[
            VariableSchema(
                name="alpha",
                dtype="<c16",
                value_domain="complex",
                dims=("trajectory", "time", "channel"),
                quantity=SDEQuantity.FIELD_AMPLITUDE.value,
            )
        ],
    )
    product = TimeSeriesDataset.from_arrays(
        schema,
        {
            "alpha": rng.standard_normal((N_TRAJ, N_TIME, 2))
            + 1j * rng.standard_normal((N_TRAJ, N_TIME, 2))
        },
        owner="test",
    )
    signal = read_sampled_signal(product, variable="alpha", channels=[0])
    assert signal.t0 == 0.0
    assert signal.dt == 1.0
