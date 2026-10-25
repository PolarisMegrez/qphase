"""The NPZ sharded writer streams chunks instead of materializing a product.

The writer used to collect every variable of a product into one mapping before
writing anything, so re-saving a product larger than memory failed. These tests
pin the two things that changed and the two that must not:

* the chunk plan, file names, logical ranges and descriptor of the streaming path
  are exactly the ones the array path produces;
* a lazy storage-backed handle is asked for a chunk at a time and never for a full
  materialization of the large variable;
* values round-trip unchanged in both layouts, across several variables, across
  chunk boundaries, with an unsharded variable, a string column and a 0-d scalar;
* a failure inside the write still publishes nothing (the ``save_products``
  transaction is unchanged).
"""

from __future__ import annotations

import numpy as np
import pytest
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ProductSchema,
    StatisticsDataset,
    VariableSchema,
    load_products,
    save_products,
)
from qphase.data import npz as npz_module
from qphase.data.npz import NpzStorageAdapter, ShardedNpzArrayHandle


def _schema(rows: int, *, columns: int = 8) -> ProductSchema:
    return ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[
            AxisSchema(name="scan", role=AxisRole.PARAMETER, size=rows),
            AxisSchema(name="tau", role=AxisRole.COORDINATE, size=columns),
        ],
        variables=[
            VariableSchema(
                name="value",
                dtype="float64",
                value_domain="real",
                dims=("scan", "tau"),
            ),
            VariableSchema(
                name="flag",
                dtype="int64",
                value_domain="real",
                dims=("scan",),
            ),
            VariableSchema(
                name="label", dtype="U6", value_domain="real", dims=("scan",)
            ),
            VariableSchema(name="scale", dtype="float64", value_domain="real"),
        ],
    )


def _payload(rows: int, *, columns: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    return {
        "value": rng.normal(size=(rows, columns)),
        "flag": np.arange(rows, dtype=np.int64),
        "label": np.asarray([f"s{index:05d}" for index in range(rows)]),
        "scale": np.asarray(1.25),
    }


def _dataset(rows: int = 40, *, columns: int = 8) -> StatisticsDataset:
    schema = _schema(rows, columns=columns)
    return StatisticsDataset.from_arrays(
        schema, _payload(rows, columns=columns), owner="engine.fake"
    )


# --- the plan and the descriptor are unchanged -------------------------------


@pytest.mark.parametrize(
    "shape, target",
    [
        ((40, 8), 640),
        ((40, 8), 100_000),
        ((7,), 16),
        ((3, 5, 2), 32),
        ((), 640),
        ((40, 8), 0),
    ],
)
def test_planning_from_a_shape_matches_planning_from_an_array(shape, target):
    array = np.zeros(shape, dtype="float64")
    assert NpzStorageAdapter._plan_shape(
        shape, array.dtype.itemsize, target
    ) == NpzStorageAdapter._plan_chunk_axis(array, target)


def test_the_streamed_descriptor_equals_the_materialized_one(tmp_path):
    rows, columns = 40, 8
    array = np.random.default_rng(5).normal(size=(rows, columns))
    variable = _schema(rows, columns=columns).variable("value")
    array_dir = tmp_path / "array"
    stream_dir = tmp_path / "stream"
    array_dir.mkdir()
    stream_dir.mkdir()
    # 40*8*8 = 2560 bytes for the variable -> 4 chunks of 10 rows at 640 bytes.
    from_array = NpzStorageAdapter._write_variable(
        "00_p", variable, array, array_dir, 640
    )
    dataset = StatisticsDataset.from_arrays(
        _schema(rows, columns=columns),
        {
            "value": array,
            "flag": np.arange(rows, dtype=np.int64),
            "label": np.asarray([f"s{index:05d}" for index in range(rows)]),
            "scale": np.asarray(1.25),
        },
        owner="engine.fake",
    )
    from_stream = NpzStorageAdapter._write_sharded_variable(
        "00_p", variable, dataset, stream_dir, 640
    )
    assert from_array == from_stream
    assert from_stream.chunk_axis == "scan"
    assert len(from_stream.chunks) == 4
    assert sorted(path.name for path in array_dir.iterdir()) == sorted(
        path.name for path in stream_dir.iterdir()
    )
    for record in from_stream.chunks:
        left = np.load(array_dir / record.file)["data"]
        right = np.load(stream_dir / record.file)["data"]
        np.testing.assert_array_equal(left, right)


# --- the write path never reads the whole variable ---------------------------


def test_resaving_a_lazy_product_reads_one_chunk_at_a_time(tmp_path, monkeypatch):
    rows, columns = 40, 8
    payload = _payload(rows, columns=columns)
    dataset = StatisticsDataset.from_arrays(
        _schema(rows, columns=columns), payload, owner="engine.fake"
    )
    source = tmp_path / "source"
    # value: 2560 bytes -> chunks of 5 rows at 320 bytes.
    save_products(source, {"p": dataset}, shard_target_bytes=320)
    lazy = load_products(source)["p"]
    handle = lazy.handle("value")
    assert isinstance(handle, ShardedNpzArrayHandle)
    assert handle.chunk_count == 8

    resaved = tmp_path / "resaved"
    materialized = lazy.materialize()
    assert isinstance(materialized.handle("value"), ShardedNpzArrayHandle)

    full_reads: list[int] = []
    selections: list[tuple] = []
    real_materialize = ShardedNpzArrayHandle.materialize
    real_selection = ShardedNpzArrayHandle.materialize_selection

    def counting_materialize(self, *args, **kwargs):
        full_reads.append(self.shape[0] if self.shape else 0)
        return real_materialize(self, *args, **kwargs)

    def counting_selection(self, indexers):
        selections.append((tuple(self.shape), tuple(indexers)))
        return real_selection(self, indexers)

    monkeypatch.setattr(ShardedNpzArrayHandle, "materialize", counting_materialize)
    monkeypatch.setattr(
        ShardedNpzArrayHandle, "materialize_selection", counting_selection
    )
    try:
        save_products(resaved, {"p": materialized}, shard_target_bytes=320)
    finally:
        monkeypatch.setattr(ShardedNpzArrayHandle, "materialize", real_materialize)
        monkeypatch.setattr(
            ShardedNpzArrayHandle, "materialize_selection", real_selection
        )

    # The large variable was never fully read, and every read is one chunk window
    # of that variable's own plan (5 rows for value, 14 for the 24-byte label).
    assert full_reads == []
    windows = [
        indexers[0]
        for shape, indexers in selections
        if shape == (rows, columns)
    ]
    assert windows
    assert all(isinstance(window, slice) for window in windows)
    assert all(window.stop - window.start <= 5 for window in windows)
    restored = load_products(resaved)["p"]
    np.testing.assert_array_equal(
        restored.handle("value").materialize(), payload["value"]
    )
    np.testing.assert_array_equal(
        restored.handle("flag").materialize(), payload["flag"]
    )


# --- coverage: layouts, several variables, unsharded, string, 0-d ------------


@pytest.mark.parametrize("layout", ["single", "sharded"])
def test_both_layouts_round_trip_every_variable(tmp_path, layout):
    rows, columns = 37, 6
    payload = _payload(rows, columns=columns)
    target = tmp_path / layout
    save_products(
        target,
        {"p": StatisticsDataset.from_arrays(
            _schema(rows, columns=columns), payload, owner="engine.fake"
        )},
        layout=layout,
        shard_target_bytes=200,
    )
    restored = load_products(target)["p"]
    for name, expected in payload.items():
        np.testing.assert_array_equal(restored.handle(name).materialize(), expected)
    if layout == "sharded":
        handle = restored.handle("value")
        assert isinstance(handle, ShardedNpzArrayHandle)
        # 37*6*8 = 1776 bytes -> several chunks, and the scan window is never
        # larger than the plan's rows per chunk.
        assert handle.chunk_count > 1
    else:
        assert sorted(path.name for path in target.iterdir() if path.suffix == ".npz")


def test_a_resaved_product_keeps_its_read_only_contract(tmp_path):
    rows, columns = 12, 4
    payload = _payload(rows, columns=columns)
    dataset = StatisticsDataset.from_arrays(
        _schema(rows, columns=columns), payload, owner="engine.fake"
    )
    source = tmp_path / "source"
    save_products(source, {"p": dataset}, shard_target_bytes=64)
    lazy = load_products(source)["p"].materialize()
    resaved = tmp_path / "resaved"
    save_products(resaved, {"p": lazy}, shard_target_bytes=64)
    restored = load_products(resaved)["p"]
    for name, expected in payload.items():
        np.testing.assert_array_equal(restored.handle(name).materialize(), expected)
    # A restored handle stays read-only: the writer never hands back a writable
    # view of the source payload.
    assert restored.handle("value").read_only is True


# --- a failing write publishes nothing --------------------------------------


def test_a_failure_in_the_streamed_write_publishes_nothing(tmp_path, monkeypatch):
    rows, columns = 40, 8
    dataset = StatisticsDataset.from_arrays(
        _schema(rows, columns=columns), _payload(rows, columns=columns),
        owner="engine.fake",
    )
    target = tmp_path / "artifact"
    calls = {"count": 0}
    real_savez = npz_module.np.savez

    def failing_savez(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 3:
            raise OSError("synthetic write failure")
        return real_savez(*args, **kwargs)

    monkeypatch.setattr(npz_module.np, "savez", failing_savez)
    with pytest.raises(OSError, match="synthetic write failure"):
        save_products(target, {"p": dataset}, shard_target_bytes=320)
    # No manifest was published and no payload file was left behind.
    assert not (target / "artifact_manifest.json").exists()
    assert sorted(path.name for path in target.iterdir()) == []


def test_a_failed_rewrite_keeps_the_previous_artifact_readable(tmp_path, monkeypatch):
    rows, columns = 8, 4
    payload = _payload(rows, columns=columns)
    target = tmp_path / "artifact"
    save_products(
        target,
        {"p": StatisticsDataset.from_arrays(
            _schema(rows, columns=columns), payload, owner="engine.fake"
        )},
        shard_target_bytes=64,
    )
    before = load_products(target)["p"]
    real_savez = npz_module.np.savez

    def failing_savez(*args, **kwargs):
        raise OSError("synthetic rewrite failure")

    monkeypatch.setattr(npz_module.np, "savez", failing_savez)
    with pytest.raises(OSError, match="synthetic rewrite failure"):
        save_products(
            target,
            {"p": StatisticsDataset.from_arrays(
                _schema(rows, columns=columns),
                {**payload, "flag": payload["flag"] + 1},
                owner="engine.fake",
            )},
            shard_target_bytes=64,
            replace=True,
        )
    monkeypatch.setattr(npz_module.np, "savez", real_savez)
    after = load_products(target)["p"]
    np.testing.assert_array_equal(
        after.handle("value").materialize(), before.handle("value").materialize()
    )
