"""Tests for coordinates, bundle descriptors and the generic data bundle."""

import json

import numpy as np
import pytest
from pydantic import ValidationError
from qphase.core.catalog import CatalogQuery, ProjectObjectCatalog
from qphase.core.project import ProjectContext
from qphase.core.result_loader import load_result
from qphase.data import (
    ArtifactAdapterError,
    ArtifactAmbiguousError,
    ArtifactCorruptError,
    ArtifactManifest,
    ArtifactNotFoundError,
    ArtifactRef,
    ArtifactUnsupportedError,
    AxisRole,
    AxisSchema,
    CoordinateSchema,
    DataKind,
    DirectoryArtifactResolver,
    GenericDataBundle,
    ProductSchema,
    ProjectArtifactResolver,
    TimeSeriesDataset,
    VariableSchema,
    load_bundle,
    register_bundle_adapter,
    save_products,
)
from qphase.data.npz import NpzStorageAdapter
from qphase.data.store import (
    GENERIC_BUNDLE_ADAPTER_ID,
    GENERIC_BUNDLE_TYPE_ID,
    BundleDescriptor,
)


def _scan_schema() -> ProductSchema:
    return ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="scan", role=AxisRole.PARAMETER, size=3),
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=4),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=8,
                coordinate="regular",
                start=0.0,
                step=0.1,
                units="s",
            ),
        ],
        variables=[
            VariableSchema(
                name="x",
                dtype="float64",
                value_domain="real",
                dims=("scan", "trajectory", "time"),
                quantity="amplitude",
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


def _scan_dataset() -> TimeSeriesDataset:
    rng = np.random.default_rng(11)
    return TimeSeriesDataset.from_arrays(
        _scan_schema(),
        {
            "x": rng.normal(size=(3, 4, 8)),
            "omega_b": np.array([1.0, 2.0, 3.0]),
        },
        owner="engine.fake",
    )


# -- coordinate schema validation ----------------------------------------------


def test_coordinate_references_unknown_variable():
    with pytest.raises(ValidationError, match="unknown variable"):
        ProductSchema(
            kind=DataKind.TIME_SERIES,
            axes=[AxisSchema(name="time", role=AxisRole.COORDINATE, size=4)],
            variables=[
                VariableSchema(
                    name="x",
                    dtype="float64",
                    value_domain="real",
                    dims=("time",),
                )
            ],
            coordinates=[
                CoordinateSchema(name="t", variable="missing", dims=("time",))
            ],
        )


def test_coordinate_dims_must_match_backing_variable():
    with pytest.raises(ValidationError, match="do not match"):
        ProductSchema(
            kind=DataKind.TIME_SERIES,
            axes=[
                AxisSchema(name="scan", role=AxisRole.PARAMETER, size=3),
                AxisSchema(name="time", role=AxisRole.COORDINATE, size=4),
            ],
            variables=[
                VariableSchema(
                    name="omega_b",
                    dtype="float64",
                    value_domain="real",
                    dims=("scan",),
                )
            ],
            coordinates=[
                CoordinateSchema(name="omega_b", variable="omega_b", dims=("time",))
            ],
        )


def test_coordinate_names_must_be_unique():
    schema = _scan_schema()
    with pytest.raises(ValidationError, match="coordinate names must be unique"):
        ProductSchema(
            kind=schema.kind,
            axes=list(schema.axes),
            variables=list(schema.variables),
            coordinates=[
                CoordinateSchema(name="dup", variable="omega_b", dims=("scan",)),
                CoordinateSchema(
                    name="dup",
                    variable="omega_b",
                    dims=("scan",),
                    role="parameter",
                ),
            ],
        )


def test_dimension_coordinate_must_be_one_dimensional():
    with pytest.raises(ValidationError, match="one-dimensional"):
        ProductSchema(
            kind=DataKind.STATISTICS,
            axes=[
                AxisSchema(name="row", role=AxisRole.INDEX, size=2),
                AxisSchema(name="col", role=AxisRole.INDEX, size=2),
            ],
            variables=[
                VariableSchema(
                    name="m",
                    dtype="float64",
                    value_domain="real",
                    dims=("row", "col"),
                )
            ],
            coordinates=[
                CoordinateSchema(name="m_coord", variable="m", dims=("row", "col"))
            ],
        )


def test_parameter_coordinate_rejects_non_parameter_axes():
    with pytest.raises(ValidationError, match="non-parameter axes"):
        ProductSchema(
            kind=DataKind.TIME_SERIES,
            axes=[
                AxisSchema(name="scan", role=AxisRole.PARAMETER, size=3),
                AxisSchema(name="time", role=AxisRole.COORDINATE, size=4),
            ],
            variables=[
                VariableSchema(
                    name="grid",
                    dtype="float64",
                    value_domain="real",
                    dims=("scan", "time"),
                )
            ],
            coordinates=[
                CoordinateSchema(
                    name="grid",
                    variable="grid",
                    dims=("scan", "time"),
                    role="parameter",
                )
            ],
        )


# -- dataset coordinate access ---------------------------------------------------


def test_dataset_coordinate_access_and_summary():
    dataset = _scan_dataset()
    np.testing.assert_array_equal(
        dataset.coordinate("omega_b"), np.array([1.0, 2.0, 3.0])
    )
    assert [c.name for c in dataset.coordinates()] == ["omega_b"]
    with pytest.raises(KeyError, match="unknown coordinate"):
        dataset.coordinate("bogus")
    summary = dataset.summary()
    assert summary["coordinates"] == [
        {
            "name": "omega_b",
            "variable": "omega_b",
            "dims": ["scan"],
            "role": "parameter",
            "units": "rad/s",
            "monotonic": True,
        }
    ]


# -- bundle persistence ----------------------------------------------------------


def test_generic_bundle_roundtrip(tmp_path):
    dataset = _scan_dataset()
    manifest = save_products(
        tmp_path, {"trajectories": dataset}, provenance={"plugin": "test"}
    )
    assert manifest.bundle.type_id == GENERIC_BUNDLE_TYPE_ID
    assert manifest.bundle.adapter_id == GENERIC_BUNDLE_ADAPTER_ID
    assert manifest.bundle.product_roles == {}

    raw = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert raw["bundle"]["type_id"] == GENERIC_BUNDLE_TYPE_ID
    coordinate = raw["products"][0]["product_schema"]["coordinates"][0]
    assert coordinate["role"] == "parameter"
    assert coordinate["variable"] == "omega_b"

    bundle = load_bundle(tmp_path)
    assert isinstance(bundle, GenericDataBundle)
    assert bundle.bundle_descriptor.type_id == GENERIC_BUNDLE_TYPE_ID
    assert bundle.provenance["plugin"] == "test"
    assert bundle.metadata["artifact_id"] == manifest.artifact_id
    assert bundle.metadata["bundle_type"] == GENERIC_BUNDLE_TYPE_ID

    product = bundle.require("trajectories")
    np.testing.assert_array_equal(
        product.coordinate("omega_b"), np.array([1.0, 2.0, 3.0])
    )
    with pytest.raises(KeyError, match="no product"):
        bundle.require("missing")


def test_load_result_returns_generic_bundle(tmp_path):
    save_products(tmp_path, {"trajectories": _scan_dataset()})
    result = load_result("whatever", tmp_path)
    assert isinstance(result, GenericDataBundle)
    assert sorted(result.data) == ["trajectories"]


def test_bundle_axes_shape_and_point_view(tmp_path):
    save_products(tmp_path, {"trajectories": _scan_dataset()})
    bundle = load_bundle(tmp_path)

    assert bundle.shape == (3,)
    np.testing.assert_array_equal(bundle.axes["scan"], np.array([1.0, 2.0, 3.0]))

    point = bundle.point_view((1,))
    assert isinstance(point, GenericDataBundle)
    assert point.shape == ()
    assert point.metadata["point_index"] == (1,)
    viewed = point.require("trajectories")
    assert "scan" not in {axis.name for axis in viewed.axes}
    assert viewed.shape["x"] == (4, 8)
    np.testing.assert_array_equal(
        viewed.handle("omega_b").materialize(), np.asarray(2.0)
    )


def test_bundle_save_repersists_with_descriptor(tmp_path):
    save_products(tmp_path, {"trajectories": _scan_dataset()})
    bundle = load_bundle(tmp_path)
    target = tmp_path / "resaved"
    bundle.save(target)

    restored = load_bundle(target)
    assert restored.bundle_descriptor == bundle.bundle_descriptor
    np.testing.assert_array_equal(
        restored.require("trajectories").coordinate("omega_b"),
        np.array([1.0, 2.0, 3.0]),
    )


def test_bundle_product_roles_must_reference_products(tmp_path):
    save_products(tmp_path, {"trajectories": _scan_dataset()})
    path = tmp_path / "artifact_manifest.json"
    raw = json.loads(path.read_text())
    raw["bundle"]["product_roles"]["extra"] = "missing_product"
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactCorruptError, match="unknown products"):
        ArtifactManifest.read(tmp_path)


def test_known_bundle_descriptor_is_validated_at_manifest_read(tmp_path):
    save_products(tmp_path, {"trajectories": _scan_dataset()})
    path = tmp_path / "artifact_manifest.json"
    raw = json.loads(path.read_text())
    raw["bundle"]["descriptor"] = {"undeclared": True}
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactCorruptError, match="must be empty"):
        ArtifactManifest.read(tmp_path)


# -- bundle adapters ---------------------------------------------------------------


def test_register_bundle_adapter_restores_custom_bundle(tmp_path):
    descriptor = BundleDescriptor(
        type_id="test.bundle/1",
        adapter_id="test/1",
        descriptor_schema="test.bundle/1",
        descriptor={"answer": 42},
        product_roles={"trajectories": "trajectories"},
    )
    save_products(tmp_path, {"trajectories": _scan_dataset()}, bundle=descriptor)

    class Adapter:
        adapter_id = "test/1"
        descriptor_schema = "test.bundle/1"

        def validate_descriptor(self, value):
            if value.descriptor_schema != self.descriptor_schema:
                raise ArtifactUnsupportedError("unsupported test descriptor")
            if value.descriptor != {"answer": 42}:
                raise ArtifactCorruptError("invalid test descriptor")

        def validate_manifest(self, manifest):
            self.validate_descriptor(manifest.bundle)

        def build(self, manifest, products):
            return {
                "descriptor": manifest.bundle.descriptor,
                "products": sorted(products),
            }

    adapter = Adapter()

    import qphase.data.store as store_module

    register_bundle_adapter(adapter)
    try:
        restored = load_bundle(tmp_path)
        assert restored == {
            "descriptor": {"answer": 42},
            "products": ["trajectories"],
        }
        # Idempotent re-registration of the same builder is allowed.
        register_bundle_adapter(adapter)
        with pytest.raises(ArtifactAdapterError, match="already registered"):
            register_bundle_adapter(Adapter())
    finally:
        store_module._BUNDLE_ADAPTERS.pop("test/1", None)


def test_unregistered_bundle_adapter_falls_back_to_generic(tmp_path):
    descriptor = BundleDescriptor(
        type_id="vendor.bundle/1",
        adapter_id="vendor/1",
        descriptor_schema="vendor.bundle/1",
        descriptor={},
        product_roles={"trajectories": "trajectories"},
    )
    save_products(tmp_path, {"trajectories": _scan_dataset()}, bundle=descriptor)
    bundle = load_bundle(tmp_path)
    assert isinstance(bundle, GenericDataBundle)
    assert bundle.bundle_descriptor.type_id == "vendor.bundle/1"


# -- resolver ------------------------------------------------------------------------


def test_directory_resolver_rejects_unbound_refs():
    resolver = DirectoryArtifactResolver()
    dataset = _scan_dataset()
    ref = ArtifactRef(
        artifact_id="unbound",
        product_name="trajectories",
        product_schema=dataset.schema,
        storage_adapter=NpzStorageAdapter.ADAPTER_ID,
    )
    with pytest.raises(ArtifactNotFoundError, match="not bound"):
        resolver.resolve(ref)

    resolver.register("unbound", "/somewhere")
    from pathlib import Path

    assert resolver.resolve(ref) == Path("/somewhere")
    resolver.clear()
    with pytest.raises(ArtifactNotFoundError):
        resolver.resolve(ref)


def test_project_resolver_indexes_repeated_identity_as_occurrences(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    dataset = _scan_dataset()
    first = project.session_root / "session-a" / "job1"
    manifest = save_products(first, {"scan": dataset}, artifact_id="shared-artifact")
    ref = manifest.product_ref("scan")
    second = project.session_root / "session-b" / "job1"
    save_products(second, {"scan": dataset}, artifact_id="shared-artifact")
    for name in ("session-a", "session-b"):
        (project.session_root / name / "session_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "qphase.session/2",
                    "session_id": name,
                    "status": "completed",
                    "jobs": {},
                }
            ),
            encoding="utf-8",
        )

    resolver = ProjectArtifactResolver(project)
    # Repeated identity is legal under the occurrence model, but an artifact
    # ref without occurrence context must not pick a location arbitrarily.
    with pytest.raises(ArtifactAmbiguousError, match="2 locations"):
        resolver.resolve(ref)
    assert resolver.locations("shared-artifact") == [
        first.resolve(),
        second.resolve(),
    ]

    catalog = ProjectObjectCatalog(project)
    catalog.reindex()
    rows = catalog.query(
        CatalogQuery(
            object_kind="occurrence",
            facets={"artifact_id": "shared-artifact"},
        )
    )
    assert len(rows) == 2
    # The catalog path answers the same ambiguity without a rescan.
    with pytest.raises(ArtifactAmbiguousError):
        resolver.resolve(ref)


def test_project_resolver_survives_project_move(tmp_path):
    import shutil

    project = ProjectContext.create(tmp_path / "project")
    dataset = _scan_dataset()
    job_dir = project.session_root / "session" / "job1"
    manifest = save_products(job_dir, {"scan": dataset}, artifact_id="movable")
    ref = manifest.product_ref("scan")
    (project.session_root / "session" / "session_manifest.json").write_text(
        json.dumps(
            {
                "schema": "qphase.session/2",
                "session_id": "session",
                "status": "completed",
                "jobs": {},
            }
        ),
        encoding="utf-8",
    )
    resolver = ProjectArtifactResolver(project)
    assert resolver.resolve(ref) == job_dir.resolve()

    moved = tmp_path / "moved"
    shutil.move(str(tmp_path / "project"), str(moved))
    moved_project = ProjectContext.load(moved / "qphase.toml")

    assert moved_project.project_id == project.project_id
    assert (
        ProjectArtifactResolver(moved_project).resolve(ref)
        == (moved_project.session_root / "session" / "job1").resolve()
    )


def test_coordinates_roundtrip_in_clean_subprocess(tmp_path):
    """Explicit coordinates survive a clean-process artifact reopen."""
    import subprocess
    import sys

    save_products(tmp_path, {"trajectories": _scan_dataset()})

    script = (
        "import json, sys;"
        "from qphase.data import load_products;"
        "products = load_products(sys.argv[1]);"
        "dataset = products['trajectories'];"
        "print(json.dumps({"
        "'coordinates': [c.name for c in dataset.coordinates()],"
        "'omega_b': dataset.coordinate('omega_b').tolist(),"
        "}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip())
    assert payload == {"coordinates": ["omega_b"], "omega_b": [1.0, 2.0, 3.0]}
