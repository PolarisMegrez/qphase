"""Boundary tests for the shared sampled-signal reader.

The reader is the single materialization boundary of the spectral and Allan
consumers: it resolves physical mode indices (``alpha``) or observable
channel names (``value``), enforces the regular time coordinate and the
``valid_length`` censoring contract of both product families, and rejects
malformed typed inputs instead of silently defaulting anything.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from qphase.core.artifacts import ArtifactStore
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ProductSchema,
    TimeSeriesDataset,
    VariableSchema,
)
from qphase.data.resolver import default_artifact_resolver
from qphase_sde.analyser.sampled_signal import read_sampled_signal
from qphase_sde.contracts.quantities import SDEQuantity

from tests.conftest import cuda_available

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
    include_valid_length: bool = True,
    valid_dtype: str = "<i8",
    valid_quantity: str = "valid_length",
    valid_domain: str = "real",
    quantity: str = SDEQuantity.FIELD_AMPLITUDE.value,
    domain: str = "complex",
    dtype: str = "<c16",
    dims: tuple[str, ...] | None = None,
    units: str = "sqrt(inverse_time)",
    axes: list[AxisSchema] | None = None,
):
    axes = _axes(n_channel, scan) if axes is None else axes
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
    if include_valid_length:
        valid_shape = (N_TRAJ,) if scan is None else (scan, N_TRAJ)
        lengths = (
            np.full(valid_shape, N_TIME) if valid_length is None else valid_length
        )
        variables.append(
            VariableSchema(
                name="valid_length",
                dtype=valid_dtype,
                value_domain=valid_domain,
                dims=(*leading, "trajectory"),
                quantity=valid_quantity,
            )
        )
        arrays["valid_length"] = np.asarray(
            lengths, dtype=np.dtype(valid_dtype)
        ).reshape(valid_shape)
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
    include_valid_length: bool = True,
    valid_dtype: str = "<i8",
    valid_quantity: str = "valid_length",
    quantity: str = SDEQuantity.SAMPLED_OBSERVABLE.value,
    domain: str = "real",
    dtype: str = "<f8",
    units: str = "occupation",
    axes: list[AxisSchema] | None = None,
):
    axes = _axes(n_channel) if axes is None else axes
    rng = np.random.default_rng(20261213)
    value = rng.standard_normal((N_TRAJ, N_TIME, n_channel))
    variables = [
        VariableSchema(
            name="value",
            dtype=dtype,
            value_domain=domain,
            dims=("trajectory", "time", "channel"),
            quantity=quantity,
            units=units,
        )
    ]
    arrays: dict[str, np.ndarray] = {
        "value": np.asarray(value, dtype=np.dtype(dtype))
    }
    if include_valid_length:
        lengths = np.full(N_TRAJ, N_TIME) if valid_length is None else valid_length
        variables.append(
            VariableSchema(
                name="valid_length",
                dtype=valid_dtype,
                value_domain="real",
                dims=("trajectory",),
                quantity=valid_quantity,
            )
        )
        arrays["valid_length"] = np.asarray(lengths, dtype=np.dtype(valid_dtype))
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


def test_alpha_requires_valid_length():
    product = _field_product(include_valid_length=False)
    with pytest.raises(TypeError, match="missing the valid_length variable"):
        read_sampled_signal(product, variable="alpha", channels=[0])


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


@pytest.mark.parametrize("length", [0, -1])
def test_valid_length_rejects_non_positive_lengths(length):
    product = _field_product(valid_length=np.full(N_TRAJ, length))
    with pytest.raises(ValueError, match="positive sample count"):
        read_sampled_signal(product, variable="alpha", channels=[0])


def test_valid_length_rejects_non_int64_dtype():
    product = _field_product(
        valid_length=np.full(N_TRAJ, N_TIME, dtype=np.float64),
        valid_dtype="<f8",
    )
    with pytest.raises(TypeError, match="int64"):
        read_sampled_signal(product, variable="alpha", channels=[0])


def test_valid_length_rejects_wrong_quantity():
    product = _field_product(valid_quantity="wrong_quantity")
    with pytest.raises(TypeError, match="quantity"):
        read_sampled_signal(product, variable="alpha", channels=[0])


def test_valid_length_rejects_complex_domain():
    axes = _axes(2)
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
            ),
            VariableSchema(
                name="valid_length",
                dtype="<c16",
                value_domain="complex",
                dims=("trajectory",),
                quantity="valid_length",
            ),
        ],
    )
    product = TimeSeriesDataset.from_arrays(
        schema,
        {
            "alpha": np.zeros((N_TRAJ, N_TIME, 2), dtype=np.complex128),
            "valid_length": np.full(N_TRAJ, N_TIME, dtype=np.complex128),
        },
        owner="test",
    )
    with pytest.raises(TypeError, match="real-valued"):
        read_sampled_signal(product, variable="alpha", channels=[0])


def test_valid_length_schema_dims_follow_scan_presence():
    # A scan-carrying point declares valid_length over (scan, trajectory);
    # the shared-length rule reads the fused vector after the squeeze.
    product = _field_product(scan=1, valid_length=np.full(N_TRAJ, 7))
    signal = read_sampled_signal(product, variable="alpha", channels=[0])
    assert signal.array.shape == (N_TRAJ, 7, 2)


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
    product = _observable_product(channel_names=None)
    with pytest.raises(TypeError, match="channel_names"):
        read_sampled_signal(product, variable="value", channels=["n0"])


def test_value_rejects_unrecorded_channel_name():
    product = _observable_product()
    with pytest.raises(ValueError, match="available"):
        read_sampled_signal(product, variable="value", channels=["n9"])


@pytest.mark.parametrize(
    "channel_names",
    [("n0", "n0"), ["dup", "dup"]],
)
def test_value_rejects_duplicate_channel_names(channel_names):
    product = _observable_product(channel_names=channel_names)
    with pytest.raises(ValueError, match="unique"):
        read_sampled_signal(product, variable="value", channels=["n0"])


@pytest.mark.parametrize("channel_names", [("n0",), ("n0", "n1", "n2")])
def test_value_rejects_channel_name_count_mismatch(channel_names):
    product = _observable_product(channel_names=channel_names)
    with pytest.raises(ValueError, match="channel axis"):
        read_sampled_signal(product, variable="value", channels=["n0"])


@pytest.mark.parametrize("channel_names", [[], ("n0", 1), ("n0", "")])
def test_value_rejects_malformed_channel_names(channel_names):
    product = _observable_product(channel_names=channel_names)
    with pytest.raises(TypeError, match="channel_names"):
        read_sampled_signal(product, variable="value", channels=["n0"])


def test_value_requires_valid_length():
    product = _observable_product(include_valid_length=False)
    with pytest.raises(TypeError, match="valid_length"):
        read_sampled_signal(product, variable="value", channels=["n0"])


def test_value_rejects_wrong_quantity():
    product = _observable_product(quantity="wrong_quantity")
    with pytest.raises(TypeError, match="quantity"):
        read_sampled_signal(product, variable="value", channels=["n0"])


def test_value_rejects_integer_dtype():
    product = _observable_product(dtype="<i8")
    with pytest.raises(TypeError, match="real floating dtype"):
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


def _time_axis_product(**time_axis_overrides):
    axes = _axes()
    overrides = {
        "name": "time",
        "role": AxisRole.COORDINATE,
        "size": N_TIME,
        "coordinate": "regular",
        "start": T0,
        "step": DT,
        "units": "time",
    }
    overrides.update(time_axis_overrides)
    axes[1] = AxisSchema(**overrides)
    return _field_product(axes=axes)


def test_time_axis_must_be_a_regular_coordinate():
    product = _time_axis_product(coordinate="explicit")
    with pytest.raises(TypeError, match="regular coordinate"):
        read_sampled_signal(product, variable="alpha", channels=[0])


@pytest.mark.parametrize("units", ["", "seconds", "inverse_time"])
def test_time_axis_requires_time_units(units):
    product = _time_axis_product(units=units)
    with pytest.raises(TypeError, match="units"):
        read_sampled_signal(product, variable="alpha", channels=[0])


def test_time_axis_requires_a_declared_start():
    product = _time_axis_product(start=None)
    with pytest.raises(ValueError, match="finite start"):
        read_sampled_signal(product, variable="alpha", channels=[0])


@pytest.mark.parametrize("step", [None, 0.0, -0.5, float("nan"), float("inf")])
def test_time_axis_requires_a_finite_positive_step(step):
    product = _time_axis_product(step=step)
    with pytest.raises(ValueError, match="finite, positive step"):
        read_sampled_signal(product, variable="alpha", channels=[0])


def test_time_axis_role_is_enforced():
    axes = _axes()
    axes[0] = AxisSchema(name="trajectory", role=AxisRole.COORDINATE, size=N_TRAJ)
    product = _field_product(axes=axes)
    with pytest.raises(TypeError, match="trajectory axis role"):
        read_sampled_signal(product, variable="alpha", channels=[0])


def test_channel_axis_role_is_enforced():
    axes = _axes()
    axes[2] = AxisSchema(name="channel", role=AxisRole.INDEX, size=2)
    product = _field_product(axes=axes)
    with pytest.raises(TypeError, match="channel axis role"):
        read_sampled_signal(product, variable="alpha", channels=[0])


def test_scan_axis_role_is_enforced():
    axes = _axes(2, scan=1)
    axes[0] = AxisSchema(name="scan", role=AxisRole.REALIZATION, size=1)
    product = _field_product(scan=1, axes=axes)
    with pytest.raises(TypeError, match="scan axis role"):
        read_sampled_signal(product, variable="alpha", channels=[0])


@pytest.mark.gpu
@pytest.mark.skipif(not cuda_available(), reason="CuPy is not installed")
def test_device_valid_length_is_checked_on_the_host():
    import cupy

    product = _field_product()
    backing_arrays = {
        "alpha": cupy.asarray(product.handle("alpha").materialize()),
        "valid_length": cupy.asarray(np.full(N_TRAJ, -1, dtype=np.int64)),
    }
    device_product = TimeSeriesDataset.from_arrays(
        product.schema, backing_arrays, owner="test", device="cuda"
    )
    with pytest.raises(ValueError, match="positive sample count"):
        read_sampled_signal(device_product, variable="alpha", channels=[0])


def _roundtrip(product, tmp_path):
    store = ArtifactStore(tmp_path, SimpleNamespace(shard_target_mib=1))
    store.save_products({"signal": product}, provenance={"engine": "test"})
    default_artifact_resolver().clear()
    return ArtifactStore(tmp_path, SimpleNamespace(shard_target_mib=1)).load_products()[
        "signal"
    ]


def test_artifact_restored_negative_valid_length_fails(tmp_path):
    product = _observable_product(valid_length=np.full(N_TRAJ, -1))
    restored = _roundtrip(product, tmp_path)
    with pytest.raises(ValueError, match="positive sample count"):
        read_sampled_signal(restored, variable="value", channels=["n0"])


def test_artifact_restored_duplicate_channel_names_fail(tmp_path):
    product = _observable_product(channel_names=("n0", "n0"))
    restored = _roundtrip(product, tmp_path)
    with pytest.raises(ValueError, match="unique"):
        read_sampled_signal(restored, variable="value", channels=["n0"])


def test_artifact_restored_non_regular_time_axis_fails(tmp_path):
    product = _time_axis_product(coordinate="explicit")
    restored = _roundtrip(product, tmp_path)
    with pytest.raises(TypeError, match="regular coordinate"):
        read_sampled_signal(restored, variable="alpha", channels=[0])


def test_artifact_restored_valid_product_reads(tmp_path):
    product = _field_product(mode_indices=[2, 5], valid_length=np.full(N_TRAJ, 8))
    restored = _roundtrip(product, tmp_path)
    signal = read_sampled_signal(restored, variable="alpha", channels=[5])
    assert signal.labels == (5,)
    assert signal.array.shape == (N_TRAJ, 8, 2)
    assert signal.t0 == T0
    assert signal.dt == DT
