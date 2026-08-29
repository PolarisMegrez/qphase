"""Tests for the typed dataset containers."""

import json

import numpy as np
import pytest
from qphase.data import (
    ArtifactRef,
    AxisRole,
    AxisSchema,
    DataKind,
    DataProduct,
    DictProductBacking,
    DirectoryArtifactResolver,
    HostArrayHandle,
    ProductSchema,
    SamplingBasisSchema,
    SpectralAttributes,
    SpectralDataset,
    StatisticsDataset,
    TimeSeriesDataset,
    UncertaintySchema,
    VariableConstraints,
    VariableSchema,
)


class _FakeAdapter:
    """Test adapter: opens zero-filled host arrays for the ref's schema."""

    @property
    def adapter_id(self) -> str:
        return "fake/1"

    def write_product(self, *args, **kwargs):
        raise NotImplementedError

    def open_product(self, *args, **kwargs):
        raise NotImplementedError

    def open_ref(self, ref: ArtifactRef, *, resolver=None) -> DictProductBacking:
        del resolver
        handles = {}
        for variable in ref.product_schema.variables:
            shape = tuple(
                ref.product_schema.axis(dim).size or 0 for dim in variable.dims
            )
            handles[variable.name] = HostArrayHandle(
                np.zeros(shape, dtype=variable.dtype),
                variable,
                owner="test.loader",
            )
        return DictProductBacking(handles)


_FAKE_ADAPTER = _FakeAdapter()


def _time_series_schema() -> ProductSchema:
    return ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=3),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=5,
                coordinate="regular",
                start=0.0,
                step=0.1,
                units="s",
            ),
        ],
        variables=[
            VariableSchema(
                name="x",
                dtype="complex128",
                value_domain="complex",
                dims=("trajectory", "time"),
                quantity="amplitude",
            ),
        ],
    )


def _spectral_schema() -> ProductSchema:
    return ProductSchema(
        kind=DataKind.SPECTRAL,
        axes=[
            AxisSchema(name="frequency", role=AxisRole.COORDINATE, size=4, units="Hz"),
            AxisSchema(name="i", role=AxisRole.COMPONENT, size=2),
            AxisSchema(name="j", role=AxisRole.COMPONENT, size=2),
        ],
        variables=[
            VariableSchema(
                name="psd",
                dtype="float64",
                value_domain="real",
                dims=("frequency",),
                quantity="power_spectral_density",
                units="1/Hz",
                constraints=VariableConstraints(nonnegative=True),
            ),
            VariableSchema(
                name="cross",
                dtype="complex128",
                value_domain="complex",
                dims=("i", "j"),
                quantity="cross_spectral_density",
                constraints=VariableConstraints(symmetry="hermitian"),
            ),
        ],
        attributes={
            "frequency_units": "Hz",
            "orientation": "increasing",
            "sidedness": "one_sided",
            "normalization": "density",
            "window": "hann",
            "estimator": "welch",
        },
    )


def test_time_series_dataset_from_arrays_summary_and_guards():
    schema = _time_series_schema()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    dataset = TimeSeriesDataset.from_arrays(
        schema, {"x": x}, owner="engine.fake", provenance={"job": "test"}
    )

    assert isinstance(dataset, DataProduct)
    assert dataset.kind is DataKind.TIME_SERIES
    assert dataset.is_runtime_backed and not dataset.is_artifact_backed
    assert dataset.devices == ("cpu",)
    assert dataset.shape == {"x": (3, 5)}
    assert dataset.nbytes == x.nbytes
    assert [axis.name for axis in dataset.axes] == ["trajectory", "time"]
    assert dataset.axis("time").step == 0.1
    assert [variable.name for variable in dataset.variables] == ["x"]
    assert dataset.attributes == {}
    assert dataset.provenance == {"job": "test"}

    handle = dataset.handle("x")
    assert handle.read_only
    payload = handle.materialize()
    assert not payload.flags.writeable
    np.testing.assert_array_equal(payload, x)
    with pytest.raises(ValueError):
        payload[0, 0] = 0.0

    summary = dataset.summary()
    json.dumps(summary)  # must be JSON-serializable
    assert summary["backing"] == "runtime"
    assert summary["nbytes"] == x.nbytes
    assert summary["kind"] == "time_series"
    assert "TimeSeriesDataset" in repr(dataset)

    with pytest.raises(TypeError, match="never coerce"):
        np.asarray(dataset)

    # No-op materialization is free and returns the dataset itself.
    assert dataset.materialize() is dataset
    assert dataset.materialize("cpu", copy_policy="never") is dataset

    with pytest.raises(ValueError, match="requires kind"):
        SpectralDataset(schema, dataset.backing)
    with pytest.raises(KeyError, match="unknown variable"):
        dataset.handle("y")


def test_spectral_dataset_mixed_real_complex():
    schema = _spectral_schema()
    psd = np.linspace(1.0, 2.0, 4)
    cross = np.ones((2, 2), dtype=np.complex128)
    dataset = SpectralDataset.from_arrays(
        schema, {"psd": psd, "cross": cross}, owner="engine.fake"
    )

    attributes = dataset.spectral_attributes
    assert isinstance(attributes, SpectralAttributes)
    assert attributes.window == "hann"
    assert attributes.sidedness == "one_sided"
    assert dataset.shape == {"psd": (4,), "cross": (2, 2)}
    assert dataset.nbytes == psd.nbytes + cross.nbytes
    assert dataset.variable("cross").constraints.symmetry == "hermitian"

    with pytest.raises(ValueError, match="requires kind"):
        TimeSeriesDataset(schema, dataset.backing)


def test_statistics_dataset_typed_columns():
    schema = ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[
            AxisSchema(name="candidate", role=AxisRole.INDEX, size=3),
            AxisSchema(name="i", role=AxisRole.COMPONENT, size=2),
            AxisSchema(name="j", role=AxisRole.COMPONENT, size=2),
        ],
        variables=[
            VariableSchema(
                name="frequency",
                dtype="float64",
                value_domain="real",
                dims=("candidate",),
                units="Hz",
            ),
            VariableSchema(
                name="rank",
                dtype="int64",
                value_domain="real",
                dims=("candidate",),
            ),
            VariableSchema(
                name="second_moment",
                dtype="float64",
                value_domain="real",
                dims=("i", "j"),
                constraints=VariableConstraints(symmetry="symmetric"),
            ),
        ],
    )
    arrays = {
        "frequency": np.array([1.0, 2.0, 3.0]),
        "rank": np.array([1, 2, 3]),
        "second_moment": np.eye(2),
    }
    dataset = StatisticsDataset.from_arrays(schema, arrays, owner="engine.fake")

    assert dataset.row_axis is not None
    assert dataset.row_axis.name == "candidate"
    assert dataset.columns == ("frequency", "rank", "second_moment")
    assert dataset.column("rank").dtype == "<i8"

    view = dataset.slice_view(candidate=slice(0, 2))
    assert view.axis("candidate").size == 2
    assert view.shape["frequency"] == (2,)
    assert view.shape["rank"] == (2,)
    assert view.shape["second_moment"] == (2, 2)  # untouched
    assert view.variable("second_moment").constraints.symmetry == "symmetric"

    point = dataset.point_view(candidate=1)
    assert "candidate" not in [axis.name for axis in point.axes]
    assert point.variable("frequency").dims == ()
    assert point.shape["frequency"] == ()
    assert float(point.handle("frequency").materialize()) == 2.0
    assert point.handle("rank").materialize().shape == ()


def test_slice_and_point_views_share_memory():
    schema = _time_series_schema()
    x = np.arange(15, dtype=np.float64).reshape(3, 5).astype(np.complex128)
    dataset = TimeSeriesDataset.from_arrays(schema, {"x": x}, owner="o")

    sliced = dataset.slice_view(time=slice(1, 4))
    assert sliced.axis("time").size == 3
    assert sliced.variable("x").dims == ("trajectory", "time")
    assert sliced.shape["x"] == (3, 3)
    materialized = sliced.handle("x").materialize()
    np.testing.assert_array_equal(materialized, x[:, 1:4])
    assert np.shares_memory(materialized, x)
    assert not materialized.flags.writeable

    point = dataset.point_view(trajectory=0)
    assert [axis.name for axis in point.axes] == ["time"]
    assert point.variable("x").dims == ("time",)
    np.testing.assert_array_equal(point.handle("x").materialize(), x[0])


def test_slice_view_rejects_invalid_selection():
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="time", role=AxisRole.COORDINATE, size=4),
            AxisSchema(name="batch", role=AxisRole.COORDINATE, size=2),
        ],
        variables=[
            VariableSchema(
                name="x", dtype="float64", value_domain="real", dims=("time",)
            ),
        ],
    )
    dataset = TimeSeriesDataset.from_arrays(schema, {"x": np.zeros(4)}, owner="o")

    with pytest.raises(ValueError, match="unknown axis"):
        dataset.slice_view(frequency=0)
    with pytest.raises(ValueError, match="not spanned"):
        dataset.slice_view(batch=0)
    with pytest.raises(TypeError, match="int or slice"):
        dataset.slice_view(time=[0, 1])
    with pytest.raises(TypeError, match="must be an int"):
        dataset.point_view(time=slice(0, 2))
    with pytest.raises(TypeError, match="must be an int"):
        dataset.point_view(time=True)


def test_slice_view_on_open_axis_uses_handle_shape():
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[AxisSchema(name="time", role=AxisRole.COORDINATE, size=None)],
        variables=[
            VariableSchema(
                name="x", dtype="float64", value_domain="real", dims=("time",)
            ),
        ],
    )
    dataset = TimeSeriesDataset.from_arrays(schema, {"x": np.arange(6.0)}, owner="o")
    assert dataset.shape == {"x": (6,)}  # runtime shape is concrete
    view = dataset.slice_view(time=slice(2, 5))
    assert view.axis("time").size == 3
    np.testing.assert_array_equal(view.handle("x").materialize(), [2.0, 3.0, 4.0])


def test_point_view_sampling_basis_rules():
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=3),
            AxisSchema(name="time", role=AxisRole.COORDINATE, size=4),
        ],
        sampling_bases=[
            SamplingBasisSchema(name="trajectories", source_axis="trajectory")
        ],
        variables=[
            VariableSchema(
                name="x",
                dtype="float64",
                value_domain="real",
                dims=("trajectory", "time"),
            ),
        ],
        uncertainties=[
            UncertaintySchema(
                target="x",
                kind="sem",
                sampling_basis="trajectories",
                scope="sampling",
            ),
        ],
    )
    dataset = TimeSeriesDataset.from_arrays(schema, {"x": np.zeros((3, 4))}, owner="o")

    # Dropping the basis' source axis would orphan the uncertainty.
    with pytest.raises(ValueError, match="sampling basis"):
        dataset.point_view(trajectory=0)

    # Slicing keeps the basis and updates the axis size.
    view = dataset.slice_view(trajectory=slice(0, 2))
    assert view.axis("trajectory").size == 2
    assert [b.name for b in view.schema.sampling_bases] == ["trajectories"]

    # Point-selecting an unrelated axis keeps the basis.
    point = dataset.point_view(time=0)
    assert [b.name for b in point.schema.sampling_bases] == ["trajectories"]


def test_artifact_backed_dataset_materializes_through_loader():
    from qphase.data import ArtifactAdapterError, register_adapter

    register_adapter(_FAKE_ADAPTER)  # idempotent for the same instance
    schema = _time_series_schema()
    ref = ArtifactRef(
        artifact_id="art-1",
        product_name="trajectories",
        product_schema=schema,
        storage_adapter="fake/1",
    )
    dataset = TimeSeriesDataset(schema, ref, provenance={"origin": "test"})

    assert isinstance(dataset, DataProduct)
    assert dataset.is_artifact_backed and not dataset.is_runtime_backed
    assert dataset.backing is ref
    assert dataset.devices == ()
    assert dataset.shape == {"x": (3, 5)}
    assert dataset.nbytes == 3 * 5 * 16
    summary = dataset.summary()
    json.dumps(summary)
    assert summary["backing"] == "artifact"
    assert summary["artifact_id"] == "art-1"

    with pytest.raises(RuntimeError, match="materialize"):
        dataset.handle("x")
    with pytest.raises(RuntimeError, match="materialize"):
        dataset.slice_view(time=0)

    loaded = dataset.materialize(resolver=DirectoryArtifactResolver())
    assert loaded.is_runtime_backed
    assert loaded.shape == {"x": (3, 5)}
    assert loaded.handle("x").materialize().dtype == np.complex128

    bad_ref = ArtifactRef(
        artifact_id="art-2",
        product_name="trajectories",
        product_schema=schema,
        storage_adapter="missing/1",
    )
    with pytest.raises(ArtifactAdapterError, match="unknown storage adapter"):
        TimeSeriesDataset(schema, bad_ref).materialize()

    mismatched = schema.model_copy(
        update={"axes": [schema.axes[0], schema.axes[1].model_copy(update={"size": 6})]}
    )
    other_ref = ArtifactRef(
        artifact_id="art-3",
        product_name="trajectories",
        product_schema=mismatched,
        storage_adapter="fake/1",
    )
    with pytest.raises(ValueError, match="does not match"):
        TimeSeriesDataset(schema, other_ref)


def test_from_arrays_validates_coverage_and_metadata():
    schema = _time_series_schema()
    x = np.zeros((3, 5), dtype=np.complex128)

    with pytest.raises(ValueError, match="missing arrays"):
        TimeSeriesDataset.from_arrays(schema, {}, owner="o")
    with pytest.raises(ValueError, match="without schema variables"):
        TimeSeriesDataset.from_arrays(schema, {"x": x, "y": np.zeros(1)}, owner="o")
    with pytest.raises(ValueError, match="dtype"):
        TimeSeriesDataset.from_arrays(schema, {"x": np.zeros((3, 5))}, owner="o")
    with pytest.raises(ValueError, match="JSON"):
        TimeSeriesDataset.from_arrays(
            schema, {"x": x}, owner="o", provenance={"bad": object()}
        )


def test_device_backed_materialize_is_explicit():
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[AxisSchema(name="time", role=AxisRole.COORDINATE, size=4)],
        variables=[
            VariableSchema(
                name="x", dtype="float64", value_domain="real", dims=("time",)
            ),
        ],
    )
    x = np.arange(4.0)
    dataset = TimeSeriesDataset.from_arrays(
        schema, {"x": x}, owner="o", device="cuda:0"
    )
    assert dataset.devices == ("cuda:0",)
    assert dataset.nbytes == x.nbytes

    with pytest.raises(RuntimeError, match="copy policy"):
        dataset.materialize("cpu", copy_policy="never")

    host = dataset.materialize("cpu")
    assert host.devices == ("cpu",)
    np.testing.assert_array_equal(host.handle("x").materialize(), x)

    # Host handles never perform host-to-device transfers.
    with pytest.raises(RuntimeError, match="cannot materialize"):
        host.materialize("cuda:0")


# -- coordinate preservation in dataset views (C5) ---------------------------------

from qphase.data import CoordinateSchema  # noqa: E402


def _coordinate_view_schema() -> ProductSchema:
    return ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="scan", role=AxisRole.PARAMETER, size=4),
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=3),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=8,
                coordinate="regular",
                start=1.0,
                step=0.5,
                units="s",
            ),
        ],
        variables=[
            VariableSchema(
                name="x",
                dtype="float64",
                value_domain="real",
                dims=("scan", "trajectory", "time"),
            ),
            VariableSchema(
                name="omega_b",
                dtype="float64",
                value_domain="real",
                dims=("scan",),
                units="rad/s",
            ),
        ],
        coordinates=[
            CoordinateSchema(
                name="omega_b",
                variable="omega_b",
                dims=("scan",),
                role="parameter",
                units="rad/s",
            ),
        ],
    )


def _coordinate_view_dataset() -> TimeSeriesDataset:
    rng = np.random.default_rng(21)
    return TimeSeriesDataset.from_arrays(
        _coordinate_view_schema(),
        {
            "x": rng.normal(size=(4, 3, 8)),
            "omega_b": np.array([10.0, 20.0, 30.0, 40.0]),
        },
        owner="engine.fake",
    )


def test_slice_view_updates_regular_axis_start_step_and_size():
    dataset = _coordinate_view_dataset()
    view = dataset.slice_view(time=slice(1, 7, 2))
    axis = view.axis("time")
    assert axis.size == 3
    assert axis.start == 1.0 + 1 * 0.5
    assert axis.step == 0.5 * 2
    assert view.shape["x"] == (4, 3, 3)
    np.testing.assert_array_equal(
        view.handle("x").materialize(),
        dataset.handle("x").materialize()[:, :, 1:7:2],
    )


def test_slice_view_reverse_slice_keeps_coordinate_semantics():
    dataset = _coordinate_view_dataset()
    view = dataset.slice_view(time=slice(None, None, -1))
    axis = view.axis("time")
    assert axis.size == 8
    # Reversed regular coordinate: first sample is the old last one and the
    # step picks up the negative stride.
    assert axis.start == 1.0 + 7 * 0.5
    assert axis.step == -0.5
    np.testing.assert_array_equal(
        view.handle("x").materialize(),
        dataset.handle("x").materialize()[:, :, ::-1],
    )


def test_slice_view_empty_slice_keeps_axis_with_zero_size():
    dataset = _coordinate_view_dataset()
    view = dataset.slice_view(time=slice(2, 2))
    axis = view.axis("time")
    assert axis.size == 0
    assert view.shape["x"] == (4, 3, 0)


def test_slice_view_slices_coordinate_variables_with_data():
    dataset = _coordinate_view_dataset()
    view = dataset.slice_view(scan=slice(1, 3))
    assert view.shape["x"] == (2, 3, 8)
    np.testing.assert_array_equal(view.coordinate("omega_b"), [20.0, 30.0])
    coordinate = view.coordinates()[0]
    assert coordinate.role == "parameter"
    assert coordinate.dims == ("scan",)


def test_point_view_keeps_scalar_parameter_coordinate():
    dataset = _coordinate_view_dataset()
    view = dataset.point_view(scan=2)
    assert "scan" not in {axis.name for axis in view.axes}
    # The selected scan coordinate survives as a scalar parameter label.
    coordinate = view.coordinates()[0]
    assert coordinate.dims == ()
    assert coordinate.role == "parameter"
    np.testing.assert_array_equal(view.coordinate("omega_b"), np.asarray(30.0))


def test_slice_view_preserves_coordinates_through_roundtrip(tmp_path):
    from qphase.data import load_products, save_products

    dataset = _coordinate_view_dataset()
    view = dataset.slice_view(scan=slice(1, 3), time=slice(0, 8, 2))
    save_products(tmp_path, {"trajectories": view})
    restored = load_products(tmp_path)["trajectories"]
    np.testing.assert_array_equal(restored.coordinate("omega_b"), [20.0, 30.0])
    axis = restored.axis("time")
    assert (axis.size, axis.start, axis.step) == (4, 1.0, 1.0)
