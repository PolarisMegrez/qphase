"""Tests for the artifact manifest v3 store and the NPZ 2.x adapter."""

import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from qphase.core.artifacts import ArtifactStore
from qphase.core.result_loader import GenericResult, load_result
from qphase.data import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactManifestV3,
    ArtifactRef,
    AxisRole,
    AxisSchema,
    DataKind,
    Dataset,
    ProductSchema,
    TimeSeriesDataset,
    VariableSchema,
    load_products,
    save_products,
)
from qphase.data import npz as npz_module
from qphase.data.npz import ShardedNpzArrayHandle


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


def test_save_load_roundtrip_single_product(tmp_path):
    dataset = _dataset()
    manifest = save_products(
        tmp_path,
        {"trajectories": dataset},
        provenance={"plugin": "engine.sde"},
        parents=["parent-artifact"],
    )

    assert manifest.schema_version == ARTIFACT_SCHEMA_VERSION
    assert manifest.loader == "qphase.data.npz:load_product_backing"
    assert manifest.parents == ["parent-artifact"]
    assert manifest.provenance == {"plugin": "engine.sde"}
    assert len(manifest.content_hash) == 64

    raw = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert raw["schema_version"] == "qphase.artifact/3"
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

    loaded = load_products(tmp_path)
    assert sorted(loaded) == ["trajectories"]
    restored = loaded["trajectories"]
    assert isinstance(restored, TimeSeriesDataset)
    assert restored.is_runtime_backed
    assert restored.shape == {"x": (5, 8), "count": (5,)}
    assert restored.provenance["artifact_id"] == manifest.artifact_id
    np.testing.assert_array_equal(
        restored.handle("x").materialize(), dataset.handle("x").materialize()
    )
    np.testing.assert_array_equal(
        restored.handle("count").materialize(), np.arange(5)
    )

    # The product reference resolves through the registered loader.
    ref = manifest.product_ref("trajectories")
    assert isinstance(ref, ArtifactRef)
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
    with pytest.raises(RuntimeError, match="not registered"):
        TimeSeriesDataset(dataset.schema, ref).materialize()
    # Opening through the store re-registers the location.
    load_products(tmp_path)
    TimeSeriesDataset(dataset.schema, ref).materialize()


def test_sharded_lazy_selection_prunes_untouched_chunks(
    tmp_path, monkeypatch
):
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
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        loaded.handle("x").materialize()


def test_unknown_adapter_is_rejected(tmp_path):
    dataset = _dataset()
    save_products(tmp_path, {"trajectories": dataset})
    manifest_path = tmp_path / "artifact_manifest.json"
    raw = json.loads(manifest_path.read_text())
    raw["products"][0]["storage"]["adapter"] = "zarr/0"
    manifest_path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="unknown storage adapter"):
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
    with pytest.raises(ValueError, match="unsupported artifact schema"):
        ArtifactManifestV3.read(tmp_path)
    with pytest.raises(FileNotFoundError):
        ArtifactManifestV3.read(tmp_path / "missing")


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
    with pytest.raises(ValueError, match="without products"):
        save_products(tmp_path, {})
    with pytest.raises(TypeError, match="must be a Dataset"):
        save_products(tmp_path, {"x": object()})
