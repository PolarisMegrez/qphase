"""Tests for the artifact manifest v4 store and the NPZ 3.x adapter."""

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from qphase.core.artifacts import ArtifactStore
from qphase.core.result_loader import load_result
from qphase.data import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactAdapterError,
    ArtifactCorruptError,
    ArtifactError,
    ArtifactManifest,
    ArtifactNotFoundError,
    ArtifactRef,
    ArtifactUnsupportedError,
    AxisRole,
    AxisSchema,
    BundleDescriptor,
    DataKind,
    GenericDataBundle,
    ProductSchema,
    ProductStorage,
    StorageVariableSummary,
    TimeSeriesDataset,
    VariableSchema,
    default_artifact_resolver,
    load_products,
    register_adapter,
    save_products,
)
from qphase.data import npz as npz_module
from qphase.data.npz import NpzStorageAdapter, ShardedNpzArrayHandle


def _schema(rows: int) -> ProductSchema:
    return ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=rows),
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
                dtype="complex128",
                value_domain="complex",
                dims=("trajectory", "time"),
                quantity="amplitude",
            ),
            VariableSchema(
                name="count",
                dtype="int64",
                value_domain="real",
                dims=("trajectory",),
            ),
        ],
    )


def _dataset(rows: int = 5) -> TimeSeriesDataset:
    schema = _schema(rows)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(rows, 8)) + 1j * rng.normal(size=(rows, 8))
    return TimeSeriesDataset.from_arrays(
        schema,
        {"x": x, "count": np.arange(rows)},
        owner="engine.fake",
        provenance={"integrator": "test"},
    )


def _rewrite_manifest(directory: Path) -> None:
    """Rewrite a mutated manifest without any content-hash bookkeeping."""
    path = directory / "artifact_manifest.json"
    raw = json.loads(path.read_text())
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def _mutate_manifest(directory: Path, mutate) -> None:
    path = directory / "artifact_manifest.json"
    raw = json.loads(path.read_text())
    mutate(raw)
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def test_save_load_roundtrip_single_product(tmp_path):
    dataset = _dataset()
    manifest = save_products(
        tmp_path,
        {"trajectories": dataset},
        provenance={"plugin": "engine.sde"},
        parents=["parent-artifact"],
    )

    assert manifest.schema_version == ARTIFACT_SCHEMA_VERSION
    assert manifest.parents == ["parent-artifact"]
    assert manifest.provenance == {"plugin": "engine.sde"}
    assert not hasattr(manifest, "content_hash")
    assert {entry.storage.adapter for entry in manifest.products} == {"npz/3"}

    raw = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert raw["schema_version"] == "qphase.artifact/4"
    assert "content_hash" not in raw
    assert "loader" not in raw  # manifests name adapter ids, never code paths
    assert {entry["storage"]["adapter"] for entry in raw["products"]} == {"npz/3"}
    assert {
        entry["storage"]["descriptor_schema"] for entry in raw["products"]
    } == {"npz.product/3"}
    files = {
        chunk["file"]
        for entry in raw["products"]
        for variable in entry["storage"]["descriptor"]["variables"].values()
        for chunk in variable["chunks"]
    }
    assert files == {"00_trajectories__x.npz", "00_trajectories__count.npz"}
    summary = raw["products"][0]["storage"]["summary"]
    assert summary["x"]["full_shape"] == [5, 8]
    assert summary["x"]["chunk_count"] == 1
    for name in files:  # native dtypes only: no pickle needed to open
        with np.load(tmp_path / name) as npz_file:
            assert set(npz_file.files) == {"data"}

    restored = load_products(tmp_path)["trajectories"]
    assert restored.is_runtime_backed
    assert restored.shape == {"x": (5, 8), "count": (5,)}
    assert restored.provenance["artifact_id"] == manifest.artifact_id
    np.testing.assert_array_equal(
        restored.handle("x").materialize(), dataset.handle("x").materialize()
    )
    np.testing.assert_array_equal(
        restored.handle("count").materialize(), np.arange(5)
    )

    # The product reference resolves through the adapter registry.
    ref = manifest.product_ref("trajectories")
    assert isinstance(ref, ArtifactRef)
    assert ref.artifact_id == manifest.artifact_id
    assert ref.product_name == "trajectories"
    assert ref.storage_adapter == "npz/3"
    assert "content_hash" not in ref.model_dump()
    backed = TimeSeriesDataset(dataset.schema, ref)
    np.testing.assert_array_equal(
        backed.materialize().handle("x").materialize(),
        dataset.handle("x").materialize(),
    )


def test_persistence_requires_closed_product_schema(tmp_path):
    schema = _schema(5).model_copy(deep=True)
    schema.axes[0].size = None
    dataset = TimeSeriesDataset.from_arrays(
        schema,
        {
            "x": np.zeros((5, 8), dtype=np.complex128),
            "count": np.arange(5),
        },
        owner="engine.fake",
    )

    with pytest.raises(ValueError, match="closed before persistence"):
        save_products(tmp_path, {"trajectories": dataset})


def test_manifest_rejects_open_persisted_product_schema(tmp_path):
    save_products(tmp_path, {"trajectories": _dataset()})

    def open_axis(raw):
        raw["products"][0]["product_schema"]["axes"][0]["size"] = None

    _mutate_manifest(tmp_path, open_axis)
    _rewrite_manifest(tmp_path)

    with pytest.raises(ArtifactCorruptError, match="closed before persistence"):
        ArtifactManifest.read(tmp_path)


def test_unknown_artifact_ref_requires_store_open(tmp_path):
    dataset = _dataset()
    manifest = save_products(tmp_path, {"trajectories": dataset})
    ref = manifest.product_ref("trajectories")
    default_artifact_resolver().clear()
    with pytest.raises(ArtifactNotFoundError, match="not bound"):
        TimeSeriesDataset(dataset.schema, ref).materialize()
    # Opening through the store re-registers the location.
    load_products(tmp_path)
    TimeSeriesDataset(dataset.schema, ref).materialize()


def test_sharded_lazy_selection_prunes_untouched_chunks(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 8)) + 1j * rng.normal(size=(40, 8))
    schema = _schema(40)
    dataset = TimeSeriesDataset.from_arrays(
        schema, {"x": x, "count": np.arange(40)}, owner="engine.fake"
    )
    # 40*8*16 = 5120 bytes for x -> 8 chunks of 5 rows at 640 bytes.
    save_products(tmp_path, {"trajectories": dataset}, shard_target_bytes=640)

    loaded = load_products(tmp_path)["trajectories"]
    handle = loaded.handle("x")
    assert isinstance(handle, ShardedNpzArrayHandle)
    assert handle.chunk_count == 8
    assert handle.shape == (40, 8)  # metadata without reading

    reads: list[str] = []
    real_load = np.load

    def counting_load(path, *args, **kwargs):
        reads.append(str(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(npz_module.np, "load", counting_load)

    def x_reads() -> int:
        return sum(1 for path in reads if "__x__" in path or "__x." in path)

    point = loaded.point_view(trajectory=7)
    assert x_reads() == 1  # only the covering shard was read
    np.testing.assert_array_equal(point.handle("x").materialize(), x[7])

    reads.clear()
    view = loaded.slice_view(trajectory=slice(5, 15))
    assert x_reads() == 2  # shards [5,10) and [10,15)
    np.testing.assert_array_equal(view.handle("x").materialize(), x[5:15])

    reads.clear()
    full = loaded.handle("x").materialize()
    assert x_reads() == 8  # explicit full read concatenates all chunks
    np.testing.assert_array_equal(full, x)


def test_payload_bytes_are_not_hashed_during_normal_read(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})
    chunk_file = tmp_path / "00_trajectories__x.npz"
    np.savez(chunk_file, data=np.zeros((5, 8), dtype=np.complex128))

    loaded = load_products(tmp_path)["trajectories"]
    np.testing.assert_array_equal(
        loaded.handle("x").materialize(),
        np.zeros((5, 8), dtype=np.complex128),
    )


def test_chunk_read_rejects_reinterpreted_payload(tmp_path):
    """Same payload bytes under a different dtype/shape never verify."""
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=4),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=8,
                coordinate="regular",
                start=0.0,
                step=0.1,
            ),
        ],
        variables=[
            VariableSchema(
                name="y",
                dtype="float64",
                value_domain="real",
                dims=("trajectory", "time"),
            ),
        ],
    )
    y = np.arange(32, dtype=np.float64).reshape(4, 8)
    dataset = TimeSeriesDataset.from_arrays(schema, {"y": y}, owner="engine.fake")
    save_products(tmp_path, {"trajectories": dataset})
    chunk_file = tmp_path / "00_trajectories__y.npz"
    # Identical 256 payload bytes reinterpreted as complex128.
    np.savez(chunk_file, data=y.reshape(-1).view(np.complex128).reshape(4, 4))

    loaded = load_products(tmp_path)["trajectories"]
    with pytest.raises(ArtifactCorruptError, match="dtype"):
        loaded.handle("y").materialize()


def test_unknown_adapter_is_rejected(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})
    _mutate_manifest(
        tmp_path,
        lambda raw: raw["products"][0]["storage"].update(adapter="zarr/0"),
    )
    _rewrite_manifest(tmp_path)

    # The manifest itself parses and lists fine; materializing fails clearly.
    ArtifactManifest.read(tmp_path)
    with pytest.raises(ArtifactAdapterError, match="unknown storage adapter"):
        load_products(tmp_path)


def test_artifact_relocation_roundtrip(tmp_path):
    dataset = _dataset()
    save_products(tmp_path / "original", {"trajectories": dataset})
    shutil.move(str(tmp_path / "original"), str(tmp_path / "moved"))

    loaded = load_products(tmp_path / "moved")["trajectories"]
    np.testing.assert_array_equal(
        loaded.handle("x").materialize(), dataset.handle("x").materialize()
    )


def test_manifest_rejects_wrong_version(tmp_path):
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps({"schema_version": "2.0"})
    )
    with pytest.raises(ArtifactUnsupportedError, match="unsupported artifact schema"):
        ArtifactManifest.read(tmp_path)
    with pytest.raises(FileNotFoundError):
        ArtifactManifest.read(tmp_path / "missing")


def test_manifest_rejects_unsafe_paths(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})

    def set_chunk_file(raw, value):
        chunks = raw["products"][0]["storage"]["descriptor"]["variables"]["x"][
            "chunks"
        ]
        chunks[0]["file"] = value

    for bad in (
        "../outside.npz",
        "sub/../../outside.npz",
        "/abs/outside.npz",
        "C:/outside.npz",
        "sub\\chunk.npz",
        "./00_trajectories__x.npz",
    ):
        _mutate_manifest(tmp_path, lambda raw, v=bad: set_chunk_file(raw, v))
        _rewrite_manifest(tmp_path)
        # Payload paths live in the adapter descriptor: they are validated
        # when the adapter parses the descriptor at load time.
        with pytest.raises(ArtifactCorruptError):
            load_products(tmp_path)


def test_manifest_rejects_duplicate_products_and_parents(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})

    def duplicate_product(raw):
        raw["products"].append(raw["products"][0])

    _mutate_manifest(tmp_path, duplicate_product)
    with pytest.raises(ArtifactCorruptError, match="unique"):
        ArtifactManifest.read(tmp_path)

    save_products(tmp_path / "second", {"trajectories": dataset})
    _mutate_manifest(
        tmp_path / "second", lambda raw: raw.update(parents=["a", "a"])
    )
    with pytest.raises(ArtifactCorruptError, match="unique"):
        ArtifactManifest.read(tmp_path / "second")


def test_manifest_rejects_naive_created_at(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})
    _mutate_manifest(
        tmp_path, lambda raw: raw.update(created_at="2026-08-25T12:00:00")
    )
    with pytest.raises(ArtifactCorruptError, match="timezone"):
        ArtifactManifest.read(tmp_path)


def test_manifest_rejects_shared_chunk_files(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})

    def share_file(raw):
        variables = raw["products"][0]["storage"]["descriptor"]["variables"]
        variables["count"]["chunks"][0]["file"] = variables["x"]["chunks"][0][
            "file"
        ]

    _mutate_manifest(tmp_path, share_file)
    _rewrite_manifest(tmp_path)
    with pytest.raises(ArtifactCorruptError, match="referenced by both"):
        load_products(tmp_path)


def test_manifest_read_rejects_payload_shared_across_products(tmp_path):
    dataset = _dataset()
    save_products(
        tmp_path,
        {"first": dataset, "second": dataset},
        layout="single",
    )

    def share_product_payload(raw):
        raw["products"][1]["storage"] = raw["products"][0]["storage"]

    _mutate_manifest(tmp_path, share_product_payload)
    _rewrite_manifest(tmp_path)

    with pytest.raises(ArtifactCorruptError, match="across products"):
        ArtifactManifest.read(tmp_path)


def test_manifest_rejects_storage_variable_mismatch(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})

    def drop_variable(raw):
        del raw["products"][0]["storage"]["summary"]["count"]

    _mutate_manifest(tmp_path, drop_variable)
    with pytest.raises(ArtifactCorruptError, match="missing"):
        ArtifactManifest.read(tmp_path)

    save_products(tmp_path / "second", {"trajectories": dataset})

    def add_variable(raw):
        summary = raw["products"][0]["storage"]["summary"]
        summary["ghost"] = summary["count"]

    _mutate_manifest(tmp_path / "second", add_variable)
    with pytest.raises(ArtifactCorruptError, match="extra"):
        ArtifactManifest.read(tmp_path / "second")


def test_manifest_rejects_removed_hash_fields(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})

    _mutate_manifest(tmp_path, lambda raw: raw.update(content_hash="removed"))
    with pytest.raises(ArtifactCorruptError, match="content_hash"):
        ArtifactManifest.read(tmp_path)

    save_products(tmp_path / "second", {"trajectories": dataset})
    _mutate_manifest(
        tmp_path / "second",
        lambda raw: raw["products"][0]["storage"]["descriptor"][
            "variables"
        ]["x"]["chunks"][0].update(sha256="removed"),
    )
    with pytest.raises(ArtifactCorruptError, match="sha256"):
        ArtifactManifest.read(tmp_path / "second")


def test_manifest_rejects_bad_chunk_ranges(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 8)) + 1j * rng.normal(size=(40, 8))
    schema = _schema(40)
    dataset = TimeSeriesDataset.from_arrays(
        schema, {"x": x, "count": np.arange(40)}, owner="engine.fake"
    )
    save_products(tmp_path, {"trajectories": dataset}, shard_target_bytes=640)

    def mutate_ranges(raw, ranges):
        chunks = raw["products"][0]["storage"]["descriptor"]["variables"]["x"][
            "chunks"
        ]
        for chunk, logical_range in zip(chunks, ranges, strict=True):
            chunk["logical_range"] = logical_range

    original = [(i * 5, (i + 1) * 5) for i in range(8)]
    cases = {
        # Range lengths stay shape-consistent so the contiguity check fires.
        "overlap": [(0, 5), (4, 9), *original[2:]],
        "gap": [(0, 5), (6, 11), *original[2:]],
        "out of order": [original[1], original[0], *original[2:]],
        "out-of-bounds": [(0, 0), *original[1:]],
    }
    for index, (label, ranges) in enumerate(cases.items()):
        target = tmp_path / f"case_{index}"
        shutil.copytree(tmp_path, target, ignore=shutil.ignore_patterns("case_*"))
        _mutate_manifest(target, lambda raw, r=ranges: mutate_ranges(raw, r))
        _rewrite_manifest(target)
        with pytest.raises(ArtifactCorruptError, match=label):
            load_products(target)


def test_artifact_store_integration(tmp_path):
    config = SimpleNamespace(shard_target_mib=1)
    store = ArtifactStore(tmp_path, config)
    dataset = _dataset()
    manifest = store.save_products({"trajectories": dataset})
    assert (tmp_path / "artifact_manifest.json").exists()

    loaded = store.load_products()
    assert sorted(loaded) == ["trajectories"]

    result = load_result("whatever", tmp_path)
    assert isinstance(result, GenericDataBundle)
    assert sorted(result.data) == ["trajectories"]
    assert result.metadata["artifact_id"] == manifest.artifact_id
    assert result.metadata["bundle_type"] == "generic.dataset_bundle/1"
    np.testing.assert_array_equal(
        result.data["trajectories"].handle("count").materialize(), np.arange(5)
    )


def test_save_products_validates_inputs(tmp_path):
    with pytest.raises(TypeError, match="must be a mapping"):
        save_products(tmp_path, [("x", object())])
    with pytest.raises(TypeError, match="must be a Dataset"):
        save_products(tmp_path / "bad", {"x": object()})


def test_save_products_allows_empty_products(tmp_path):
    manifest = save_products(tmp_path, {}, provenance={"engine": "test"})
    assert manifest.products == []


def test_artifact_ref_uses_trusted_adapter_ids():
    schema = _schema(2)
    ref = ArtifactRef(
        artifact_id="art-1",
        product_name="trajectories",
        product_schema=schema,
        storage_adapter="npz/3",
    )
    payload = json.loads(json.dumps(ref.model_dump(mode="json")))
    assert ArtifactRef.model_validate(payload) == ref

    with pytest.raises(ValidationError, match="registry id"):
        ArtifactRef(
            artifact_id="art-1",
            product_name="trajectories",
            product_schema=schema,
            storage_adapter="os:system",
        )
    with pytest.raises(ValidationError):  # no dotted loader field remains
        ArtifactRef(
            artifact_id="art-1",
            product_name="trajectories",
            product_schema=schema,
            storage_adapter="npz/3",
            loader="os:system",
        )


def test_dataset_materialize_unknown_adapter_fails_clearly():
    schema = _schema(2)
    ref = ArtifactRef(
        artifact_id="art-1",
        product_name="trajectories",
        product_schema=schema,
        storage_adapter="zarr/9",
    )
    with pytest.raises(ArtifactAdapterError, match="unknown storage adapter"):
        TimeSeriesDataset(schema, ref).materialize()


def test_register_adapter_forbids_silent_overwrite():
    from qphase.data.store import _resolve_adapter

    _resolve_adapter("npz/3")  # ensure the built-in adapter is registered

    class _FakeAdapter:
        @property
        def adapter_id(self):
            return "npz/3"

    with pytest.raises(ArtifactAdapterError, match="already registered"):
        register_adapter(_FakeAdapter())


# -- storage descriptor versioning (C3) ------------------------------------------


def _storage(raw):
    return raw["products"][0]["storage"]


def test_load_rejects_unsupported_descriptor_schema(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})
    _mutate_manifest(
        tmp_path,
        lambda raw: _storage(raw).update(descriptor_schema="npz.product/99"),
    )
    _rewrite_manifest(tmp_path)

    # A known adapter rejects unsupported metadata at the read boundary.
    with pytest.raises(ArtifactUnsupportedError, match="descriptor schema"):
        ArtifactManifest.read(tmp_path)


def test_npz_rejects_undeclared_payload_keys(tmp_path):
    save_products(tmp_path, {"trajectories": _dataset()})
    manifest = ArtifactManifest.read(tmp_path)
    descriptor = manifest.products[0].storage.descriptor
    chunk = next(iter(descriptor["variables"].values()))["chunks"][0]
    path = tmp_path / chunk["file"]
    with np.load(path) as payload:
        data = np.asarray(payload[chunk["key"]])
    np.savez(path, **{chunk["key"]: data, "undeclared": np.array([1])})

    product = load_products(tmp_path)["trajectories"]
    with pytest.raises(ArtifactCorruptError, match="expected exactly"):
        product.handle("x").materialize()


def test_first_publish_refuses_existing_payload_path(tmp_path):
    collision = tmp_path / "00_trajectories__x.npz"
    collision.write_bytes(b"user-owned")
    with pytest.raises(ArtifactError, match="payload paths already exist"):
        save_products(tmp_path, {"trajectories": _dataset()})
    assert collision.read_bytes() == b"user-owned"
    assert not (tmp_path / "artifact_manifest.json").exists()


def test_manifest_metadata_rejects_nonfinite_json():
    with pytest.raises(ValidationError, match="JSON-serializable"):
        BundleDescriptor(
            type_id="test/1",
            adapter_id="test/1",
            descriptor_schema="test/1",
            descriptor={"bad": float("nan")},
        )


def test_manifest_rejects_summary_mismatch(tmp_path):
    dataset = _dataset()
    cases = {
        "dtype": lambda summary: summary["x"].update(dtype="<f4"),
        "rank": lambda summary: summary["x"].update(full_shape=[5, 8, 1]),
        "closed axis": lambda summary: summary["x"].update(full_shape=[6, 8]),
        "bytes": lambda summary: summary["count"].update(nbytes=1),
    }
    for index, (label, mutate) in enumerate(cases.items()):
        target = tmp_path / f"case_{index}"
        save_products(target, {"trajectories": dataset})

        def apply(raw, m=mutate):
            m(raw["products"][0]["storage"]["summary"])

        _mutate_manifest(target, apply)
        _rewrite_manifest(target)
        with pytest.raises(ArtifactCorruptError, match=label):
            ArtifactManifest.read(target)



def test_storage_descriptor_must_be_json():
    with pytest.raises(ValidationError, match="JSON-serializable"):
        ProductStorage(
            adapter="npz/3",
            descriptor_schema="npz.product/3",
            summary={
                "x": StorageVariableSummary(
                    full_shape=(5, 8), dtype="<c16", nbytes=640, chunk_count=1
                )
            },
            descriptor={"bad": object()},
        )


def test_load_rejects_descriptor_schema_inconsistencies(tmp_path):
    """Descriptor details are adapter-validated even when the summary passes."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 8)) + 1j * rng.normal(size=(40, 8))
    dataset = TimeSeriesDataset.from_arrays(
        _schema(40), {"x": x, "count": np.arange(40)}, owner="engine.fake"
    )
    save_products(tmp_path, {"trajectories": dataset}, shard_target_bytes=640)

    def descriptor_variables(raw):
        return raw["products"][0]["storage"]["descriptor"]["variables"]

    cases = {
        "dtype": lambda raw: descriptor_variables(raw)["x"].update(dtype="<f4"),
        "chunk_axis": lambda raw: descriptor_variables(raw)["count"].update(
            chunk_axis="trajectory"
        ),
        "no chunk_axis": lambda raw: descriptor_variables(raw)["x"].update(
            chunk_axis=None
        ),
        "unknown chunk axis": lambda raw: descriptor_variables(raw)["x"].update(
            chunk_axis="ghost"
        ),
        "logical_range": lambda raw: descriptor_variables(raw)["count"][
            "chunks"
        ][0].update(logical_range=[0, 5]),
        "full shape": lambda raw: descriptor_variables(raw)["x"]["chunks"][
            0
        ].update(shape=[5, 1]),
    }
    for index, (label, mutate) in enumerate(cases.items()):
        target = tmp_path / f"case_{index}"
        shutil.copytree(tmp_path, target, ignore=shutil.ignore_patterns("case_*"))
        _mutate_manifest(target, mutate)
        _rewrite_manifest(target)
        with pytest.raises(ArtifactCorruptError, match=label):
            load_products(target)


# -- transactional writes and axis-aware sharding (C4) ----------------------------


def test_single_layout_writes_one_multikey_file_per_product(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset}, layout="single")

    raw = json.loads((tmp_path / "artifact_manifest.json").read_text())
    variables = raw["products"][0]["storage"]["descriptor"]["variables"]
    files = {
        chunk["file"]
        for variable in variables.values()
        for chunk in variable["chunks"]
    }
    assert files == {"00_trajectories.npz"}
    keys = {
        chunk["key"] for variable in variables.values() for chunk in variable["chunks"]
    }
    assert keys == {"x", "count"}
    with np.load(tmp_path / "00_trajectories.npz") as npz_file:
        assert set(npz_file.files) == {"x", "count"}

    restored = load_products(tmp_path)["trajectories"]
    np.testing.assert_array_equal(
        restored.handle("x").materialize(), dataset.handle("x").materialize()
    )
    np.testing.assert_array_equal(
        restored.handle("count").materialize(), np.arange(5)
    )


def test_single_layout_ignores_shard_target(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 8)) + 1j * rng.normal(size=(40, 8))
    dataset = TimeSeriesDataset.from_arrays(
        _schema(40), {"x": x, "count": np.arange(40)}, owner="engine.fake"
    )
    save_products(
        tmp_path, {"trajectories": dataset}, layout="single",
        shard_target_bytes=64,
    )
    summary = json.loads((tmp_path / "artifact_manifest.json").read_text())[
        "products"
    ][0]["storage"]["summary"]
    assert summary["x"]["chunk_count"] == 1


def _scan_trajectory_schema() -> ProductSchema:
    return ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="scan", role=AxisRole.PARAMETER, size=1),
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=40),
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
                dtype="complex128",
                value_domain="complex",
                dims=("scan", "trajectory", "time"),
            ),
        ],
    )


def test_writer_shards_along_trajectory_when_scan_is_singleton(tmp_path):
    """The old axis0-only writer could not shard a scan=1 payload at all."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(1, 40, 8)) + 1j * rng.normal(size=(1, 40, 8))
    dataset = TimeSeriesDataset.from_arrays(
        _scan_trajectory_schema(), {"x": x}, owner="engine.fake"
    )
    # 1*40*8*16 = 5120 bytes -> 8 chunks of 5 trajectories along axis 1.
    save_products(tmp_path, {"trajectories": dataset}, shard_target_bytes=640)

    raw = json.loads((tmp_path / "artifact_manifest.json").read_text())
    variable = raw["products"][0]["storage"]["descriptor"]["variables"]["x"]
    assert variable["chunk_axis"] == "trajectory"
    assert len(variable["chunks"]) == 8

    restored = load_products(tmp_path)["trajectories"]
    handle = restored.handle("x")
    assert isinstance(handle, ShardedNpzArrayHandle)
    np.testing.assert_array_equal(handle.materialize(), x)
    # Point/slice selection prunes chunks along the trajectory axis.
    np.testing.assert_array_equal(
        handle.materialize_selection((0, 7, slice(None))), x[0, 7, :]
    )
    np.testing.assert_array_equal(
        handle.materialize_selection((0, slice(3, 12), slice(None))),
        x[0, 3:12, :],
    )


def test_writer_prefers_scan_axis_for_point_pruning(tmp_path, monkeypatch):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(8, 2, 4)) + 1j * rng.normal(size=(8, 2, 4))
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="scan", role=AxisRole.PARAMETER, size=8),
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=2),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=4,
                coordinate="regular",
                start=0.0,
                step=0.1,
            ),
        ],
        variables=[
            VariableSchema(
                name="x",
                dtype="complex128",
                value_domain="complex",
                dims=("scan", "trajectory", "time"),
            ),
        ],
    )
    dataset = TimeSeriesDataset.from_arrays(schema, {"x": x}, owner="engine.fake")
    # 8*2*4*16 = 1024 bytes -> 4 chunks of 2 scan rows along axis 0.
    save_products(tmp_path, {"trajectories": dataset}, shard_target_bytes=256)

    raw = json.loads((tmp_path / "artifact_manifest.json").read_text())
    variable = raw["products"][0]["storage"]["descriptor"]["variables"]["x"]
    assert variable["chunk_axis"] == "scan"
    assert len(variable["chunks"]) == 4

    loaded = load_products(tmp_path)["trajectories"]
    handle = loaded.handle("x")
    real_load = np.load
    calls = 0

    def counting_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(npz_module.np, "load", counting_load)
    # One scan point touches exactly one of the four chunks.
    np.testing.assert_array_equal(
        handle.materialize_selection((5, slice(None), slice(None))), x[5]
    )
    assert calls == 1


def test_save_products_refuses_overwrite_by_default(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})
    with pytest.raises(ArtifactError, match="replace=True"):
        save_products(tmp_path, {"trajectories": dataset})

    replaced = save_products(tmp_path, {"trajectories": dataset}, replace=True)
    restored = load_products(tmp_path)["trajectories"]
    np.testing.assert_array_equal(
        restored.handle("count").materialize(), np.arange(5)
    )
    # Replacement wrote fresh file names and removed the old payload.
    files = {path.name for path in tmp_path.glob("*.npz")}
    assert files and all("__r" in name for name in files)
    assert replaced.artifact_id != ""


def test_failed_replace_keeps_old_artifact_readable(tmp_path, monkeypatch):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})
    old_files = {path.name for path in tmp_path.glob("*.npz")}

    adapter = NpzStorageAdapter()
    real_write = adapter.write_product

    def failing_write(self, *args, **kwargs):
        real_write(*args, **kwargs)
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(
        NpzStorageAdapter, "write_product", failing_write
    )
    with pytest.raises(RuntimeError, match="injected"):
        save_products(tmp_path, {"trajectories": _dataset()}, replace=True)

    # The old manifest and payload are untouched; staging is cleaned up.
    restored = load_products(tmp_path)["trajectories"]
    np.testing.assert_array_equal(
        restored.handle("count").materialize(), np.arange(5)
    )
    assert {path.name for path in tmp_path.glob("*.npz")} == old_files
    assert not list(tmp_path.glob(".staging-*"))
    assert not list(tmp_path.glob("*.tmp"))


def test_replace_removes_stale_payload_files(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset}, shard_target_bytes=64)
    old_files = {path.name for path in tmp_path.glob("*.npz")}
    assert len(old_files) > 2  # sharded payload

    save_products(tmp_path, {"trajectories": dataset}, replace=True, layout="single")
    files = {path.name for path in tmp_path.glob("*.npz")}
    assert len(files) == 1  # single payload file replaces the shard set
    assert not files & old_files
    restored = load_products(tmp_path)["trajectories"]
    np.testing.assert_array_equal(
        restored.handle("count").materialize(), np.arange(5)
    )


def test_interrupted_chunk_rename_keeps_old_artifact(tmp_path, monkeypatch):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})
    old_files = {path.name for path in tmp_path.glob("*.npz")}

    import qphase.data.store as store_module

    real_replace = os.replace
    calls = 0

    def failing_replace(src, dst):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated crash during chunk publish")
        return real_replace(src, dst)

    monkeypatch.setattr(store_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated crash"):
        save_products(tmp_path, {"trajectories": _dataset()}, replace=True)

    # The old artifact is still fully readable; no half-published chunk or
    # staging/tmp file survives.
    restored = load_products(tmp_path)["trajectories"]
    np.testing.assert_array_equal(
        restored.handle("count").materialize(), np.arange(5)
    )
    assert {path.name for path in tmp_path.glob("*.npz")} == old_files
    assert not list(tmp_path.glob(".staging-*"))
    assert not list(tmp_path.glob("*.tmp"))


def test_interrupted_manifest_publish_keeps_old_artifact(tmp_path, monkeypatch):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})

    import qphase.data.store as store_module

    real_replace = os.replace

    def failing_replace(src, dst):
        if str(dst).endswith(store_module.MANIFEST_FILENAME):
            raise OSError("simulated crash at manifest publish")
        return real_replace(src, dst)

    monkeypatch.setattr(store_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="manifest publish"):
        save_products(tmp_path, {"trajectories": _dataset()}, replace=True)

    restored = load_products(tmp_path)["trajectories"]
    np.testing.assert_array_equal(
        restored.handle("count").materialize(), np.arange(5)
    )
    # New chunks published before the crash are rolled back as well.
    assert not [
        name
        for name in (p.name for p in tmp_path.glob("*.npz"))
        if "__r" in name
    ]
    assert not list(tmp_path.glob("*.tmp"))
