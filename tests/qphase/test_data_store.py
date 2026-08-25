"""Tests for the artifact manifest v3 store and the NPZ 2.x adapter."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from qphase.core.artifacts import ArtifactStore
from qphase.core.result_loader import GenericResult, load_result
from qphase.data import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactAdapterError,
    ArtifactChecksumError,
    ArtifactCorruptError,
    ArtifactManifestV3,
    ArtifactNotFoundError,
    ArtifactRef,
    ArtifactUnsupportedError,
    AxisRole,
    AxisSchema,
    DataKind,
    ProductSchema,
    TimeSeriesDataset,
    VariableSchema,
    load_products,
    register_adapter,
    save_products,
)
from qphase.data import npz as npz_module
from qphase.data.npz import ShardedNpzArrayHandle
from qphase.data.store import artifact_content_hash, product_content_hash


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


def _recompute_hashes(directory: Path) -> None:
    """Recompute product/artifact hashes after a raw manifest mutation.

    Lets tests craft manifests that pass the hash layer so the specific
    cross-field validation under test is what actually fires.
    """
    path = directory / "artifact_manifest.json"
    manifest = ArtifactManifestV3.model_validate(json.loads(path.read_text()))
    for entry in manifest.products:
        entry.sha256 = product_content_hash(
            entry.name, entry.product_schema, entry.storage
        )
    manifest.content_hash = artifact_content_hash(
        None, manifest.products, manifest.provenance, manifest.parents
    )
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


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
    assert len(manifest.content_hash) == 64
    assert {entry.storage.adapter for entry in manifest.products} == {"npz/2"}

    raw = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert raw["schema_version"] == "qphase.artifact/3"
    assert "loader" not in raw  # manifests name adapter ids, never code paths
    files = {
        chunk["file"]
        for entry in raw["products"]
        for chunks in entry["storage"]["variables"].values()
        for chunk in chunks
    }
    assert files == {"00_trajectories__x.npz", "00_trajectories__count.npz"}
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
    assert ref.storage_adapter == "npz/2"
    assert ref.content_hash == manifest.products[0].sha256
    backed = TimeSeriesDataset(dataset.schema, ref)
    np.testing.assert_array_equal(
        backed.materialize().handle("x").materialize(),
        dataset.handle("x").materialize(),
    )


def test_unknown_artifact_ref_requires_store_open(tmp_path):
    dataset = _dataset()
    manifest = save_products(tmp_path, {"trajectories": dataset})
    ref = manifest.product_ref("trajectories")
    npz_module._LOCATIONS.clear()
    with pytest.raises(ArtifactNotFoundError, match="not registered"):
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


def test_corrupted_chunk_is_detected(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})
    chunk_file = tmp_path / "00_trajectories__x.npz"
    np.savez(chunk_file, data=np.zeros((5, 8), dtype=np.complex128))

    loaded = load_products(tmp_path)["trajectories"]
    with pytest.raises(ArtifactChecksumError, match="checksum mismatch"):
        loaded.handle("x").materialize()


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
    with pytest.raises(ArtifactChecksumError, match="dtype"):
        loaded.handle("y").materialize()


def test_unknown_adapter_is_rejected(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})
    _mutate_manifest(
        tmp_path,
        lambda raw: raw["products"][0]["storage"].update(adapter="zarr/0"),
    )
    _recompute_hashes(tmp_path)

    # The manifest itself parses and lists fine; materializing fails clearly.
    ArtifactManifestV3.read(tmp_path)
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
        ArtifactManifestV3.read(tmp_path)
    with pytest.raises(FileNotFoundError):
        ArtifactManifestV3.read(tmp_path / "missing")


def test_manifest_rejects_unsafe_paths(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})

    def set_chunk_file(raw, value):
        raw["products"][0]["storage"]["variables"]["x"][0]["file"] = value

    for bad in (
        "../outside.npz",
        "sub/../../outside.npz",
        "/abs/outside.npz",
        "C:/outside.npz",
        "sub\\chunk.npz",
        "./00_trajectories__x.npz",
    ):
        _mutate_manifest(tmp_path, lambda raw, v=bad: set_chunk_file(raw, v))
        with pytest.raises(ArtifactCorruptError):
            ArtifactManifestV3.read(tmp_path)


def test_manifest_rejects_duplicate_products_and_parents(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})

    def duplicate_product(raw):
        raw["products"].append(raw["products"][0])

    _mutate_manifest(tmp_path, duplicate_product)
    with pytest.raises(ArtifactCorruptError, match="unique"):
        ArtifactManifestV3.read(tmp_path)

    save_products(tmp_path / "second", {"trajectories": dataset})
    _mutate_manifest(
        tmp_path / "second", lambda raw: raw.update(parents=["a", "a"])
    )
    with pytest.raises(ArtifactCorruptError, match="unique"):
        ArtifactManifestV3.read(tmp_path / "second")


def test_manifest_rejects_naive_created_at(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})
    _mutate_manifest(
        tmp_path, lambda raw: raw.update(created_at="2026-08-25T12:00:00")
    )
    with pytest.raises(ArtifactCorruptError, match="timezone"):
        ArtifactManifestV3.read(tmp_path)


def test_manifest_rejects_shared_chunk_files(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})

    def share_file(raw):
        variables = raw["products"][0]["storage"]["variables"]
        variables["count"][0]["file"] = variables["x"][0]["file"]

    _mutate_manifest(tmp_path, share_file)
    with pytest.raises(ArtifactCorruptError, match="referenced by both"):
        ArtifactManifestV3.read(tmp_path)


def test_manifest_rejects_storage_variable_mismatch(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})

    def drop_variable(raw):
        del raw["products"][0]["storage"]["variables"]["count"]

    _mutate_manifest(tmp_path, drop_variable)
    with pytest.raises(ArtifactCorruptError, match="missing"):
        ArtifactManifestV3.read(tmp_path)

    save_products(tmp_path / "second", {"trajectories": dataset})

    def add_variable(raw):
        storage = raw["products"][0]["storage"]["variables"]
        storage["ghost"] = [storage["count"][0]]

    _mutate_manifest(tmp_path / "second", add_variable)
    with pytest.raises(ArtifactCorruptError, match="extra"):
        ArtifactManifestV3.read(tmp_path / "second")


def test_manifest_rejects_stale_hashes(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})

    def flip_chunk_hash(raw):
        raw["products"][0]["storage"]["variables"]["x"][0]["sha256"] = "0" * 64

    _mutate_manifest(tmp_path, flip_chunk_hash)
    with pytest.raises(ArtifactCorruptError, match="content hash mismatch"):
        ArtifactManifestV3.read(tmp_path)

    save_products(tmp_path / "second", {"trajectories": dataset})
    _mutate_manifest(
        tmp_path / "second", lambda raw: raw.update(content_hash="0" * 64)
    )
    with pytest.raises(ArtifactCorruptError, match="content hash mismatch"):
        ArtifactManifestV3.read(tmp_path / "second")


def test_manifest_rejects_bad_chunk_ranges(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 8)) + 1j * rng.normal(size=(40, 8))
    schema = _schema(40)
    dataset = TimeSeriesDataset.from_arrays(
        schema, {"x": x, "count": np.arange(40)}, owner="engine.fake"
    )
    save_products(tmp_path, {"trajectories": dataset}, shard_target_bytes=640)

    def mutate_ranges(raw, ranges):
        chunks = raw["products"][0]["storage"]["variables"]["x"]
        for chunk, axis0_range in zip(chunks, ranges, strict=True):
            chunk["axis0_range"] = axis0_range

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
        _recompute_hashes(target)
        with pytest.raises(ArtifactCorruptError, match=label):
            ArtifactManifestV3.read(target)


def test_artifact_store_integration(tmp_path):
    config = SimpleNamespace(shard_target_mib=1)
    store = ArtifactStore(tmp_path, config)
    dataset = _dataset()
    manifest = store.save_products({"trajectories": dataset})
    assert (tmp_path / "artifact_manifest.json").exists()

    loaded = store.load_products()
    assert sorted(loaded) == ["trajectories"]

    result = load_result("whatever", tmp_path)
    assert isinstance(result, GenericResult)
    assert sorted(result.data) == ["trajectories"]
    assert result.metadata["artifact_id"] == manifest.artifact_id
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
        storage_adapter="npz/2",
        content_hash="ab" * 32,
    )
    payload = json.loads(json.dumps(ref.model_dump(mode="json")))
    assert ArtifactRef.model_validate(payload) == ref

    with pytest.raises(ValidationError, match="registry id"):
        ArtifactRef(
            artifact_id="art-1",
            product_name="trajectories",
            product_schema=schema,
            storage_adapter="os:system",
            content_hash="ab" * 32,
        )
    with pytest.raises(ValidationError):  # no dotted loader field remains
        ArtifactRef(
            artifact_id="art-1",
            product_name="trajectories",
            product_schema=schema,
            storage_adapter="npz/2",
            content_hash="ab" * 32,
            loader="os:system",
        )


def test_dataset_materialize_unknown_adapter_fails_clearly():
    schema = _schema(2)
    ref = ArtifactRef(
        artifact_id="art-1",
        product_name="trajectories",
        product_schema=schema,
        storage_adapter="zarr/9",
        content_hash="ab" * 32,
    )
    with pytest.raises(ArtifactAdapterError, match="unknown storage adapter"):
        TimeSeriesDataset(schema, ref).materialize()


def test_register_adapter_forbids_silent_overwrite():
    from qphase.data.store import _resolve_adapter

    _resolve_adapter("npz/2")  # ensure the built-in adapter is registered

    class _FakeAdapter:
        @property
        def adapter_id(self):
            return "npz/2"

    with pytest.raises(ArtifactAdapterError, match="already registered"):
        register_adapter(_FakeAdapter())
