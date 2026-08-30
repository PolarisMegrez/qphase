"""Tests for manifest-declared attachments and subdirectory-sharded payloads.

The Kerr-full retained scans are assembled as one root artifact whose sharded
payload files live in ``chunks/chunk_*`` subdirectories; these tests freeze
that pattern: hand-assembled manifests load through the standard path, point
views map every scan point, and declared attachments are the only auxiliary
files readable through the public interface.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from qphase.data import (
    ArtifactCorruptError,
    ArtifactManifest,
    ArtifactNotFoundError,
    AxisRole,
    AxisSchema,
    BundleDescriptor,
    DataKind,
    ProductEntry,
    ProductSchema,
    VariableSchema,
    list_artifact_attachments,
    load_products,
    read_artifact_attachment,
    save_products,
)
from qphase.data.npz import (
    NpzChunkRecord,
    NpzVariableDescriptor,
    build_product_storage,
)
from qphase.data.store import GENERIC_BUNDLE_ADAPTER_ID, GENERIC_BUNDLE_TYPE_ID


def _scan_schema(points: int) -> ProductSchema:
    return ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="scan", role=AxisRole.PARAMETER, size=points),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=4,
                coordinate="regular",
                start=0.0,
                step=0.5,
                units="s",
            ),
        ],
        variables=[
            VariableSchema(
                name="x",
                dtype="complex128",
                value_domain="complex",
                dims=("scan", "time"),
                quantity="amplitude",
            ),
            VariableSchema(
                name="parameter.gain",
                dtype="float64",
                value_domain="real",
                dims=("scan",),
                quantity="scan_parameter",
            ),
        ],
    )


def _assembled_subdirectory_artifact(root: Path, points_per_chunk=(2, 1)) -> np.ndarray:
    """Write a root artifact whose payload lives in chunks/chunk_* subdirs."""
    rng = np.random.default_rng(7)
    points = sum(points_per_chunk)
    x = rng.normal(size=(points, 4)) + 1j * rng.normal(size=(points, 4))
    gains = np.linspace(0.5, 1.5, points)

    x_chunks: list[NpzChunkRecord] = []
    gain_chunks: list[NpzChunkRecord] = []
    offset = 0
    for index, count in enumerate(points_per_chunk):
        subdir = root / "chunks" / f"chunk_{index:04d}"
        subdir.mkdir(parents=True)
        x_file = f"chunks/chunk_{index:04d}/00_scan__x__{index:04d}.npz"
        np.savez(root / x_file, data=x[offset : offset + count])
        x_chunks.append(
            NpzChunkRecord(
                file=x_file,
                key="data",
                logical_range=(offset, offset + count),
                shape=(count, 4),
                dtype="<c16",
            )
        )
        gain_file = f"chunks/chunk_{index:04d}/00_scan__parameter.gain.npz"
        np.savez(root / gain_file, data=gains[offset : offset + count])
        gain_chunks.append(
            NpzChunkRecord(
                file=gain_file,
                key="data",
                logical_range=(offset, offset + count),
                shape=(count,),
                dtype="<f8",
            )
        )
        offset += count

    schema = _scan_schema(points)
    storage = build_product_storage(
        schema,
        {
            "x": NpzVariableDescriptor(
                full_shape=(points, 4),
                dtype="<c16",
                chunk_axis="scan",
                chunks=x_chunks,
            ),
            "parameter.gain": NpzVariableDescriptor(
                full_shape=(points,),
                dtype="<f8",
                chunk_axis="scan",
                chunks=gain_chunks,
            ),
        },
    )
    manifest = ArtifactManifest(
        artifact_id="assembled-subdirectory-scan",
        created_at="2026-08-30T00:00:00+00:00",
        bundle=BundleDescriptor(
            type_id=GENERIC_BUNDLE_TYPE_ID,
            adapter_id=GENERIC_BUNDLE_ADAPTER_ID,
            descriptor_schema=GENERIC_BUNDLE_TYPE_ID,
        ),
        products=[ProductEntry(name="scan", product_schema=schema, storage=storage)],
        provenance={"engine": "test"},
    )
    manifest.write(root)
    return x


def test_subdirectory_sharded_payload_point_mapping(tmp_path):
    x = _assembled_subdirectory_artifact(tmp_path)
    dataset = load_products(tmp_path)["scan"]

    # Every point maps to its source slice, with the parameter coordinate.
    for point in range(3):
        view = dataset.point_view(scan=point)
        np.testing.assert_array_equal(view.handle("x").materialize(), x[point])
        assert float(view.handle("parameter.gain").materialize()) == pytest.approx(
            np.linspace(0.5, 1.5, 3)[point]
        )
    np.testing.assert_array_equal(dataset.handle("x").materialize(), x)


def test_subdirectory_sharded_payload_clean_process_restore(tmp_path):
    x = _assembled_subdirectory_artifact(tmp_path)
    script = (
        "import sys\n"
        "import numpy as np\n"
        "from qphase.data import load_products\n"
        "dataset = load_products(sys.argv[1])['scan']\n"
        "view = dataset.point_view(scan=2)\n"
        "np.testing.assert_array_equal(view.handle('x').materialize(), "
        "np.asarray(json_x)[2])\n"
    )
    probe = tmp_path / "probe.py"
    probe.write_text(script.replace("json_x", repr(x.tolist())), encoding="utf-8")
    subprocess.run([sys.executable, str(probe), str(tmp_path)], check=True)


def test_read_declared_attachment(tmp_path):
    save_products(
        tmp_path,
        {},
        provenance={
            "attachments": [
                {
                    "name": "analysis_sidecar",
                    "path": "aux/analysis_sidecar.json",
                    "media_type": "application/json",
                }
            ]
        },
    )
    aux = tmp_path / "aux"
    aux.mkdir()
    payload = {"analysers": {"psd": [{"peaks": {"0": {"indices": [1, 2]}}}]}}
    (aux / "analysis_sidecar.json").write_text(json.dumps(payload), encoding="utf-8")

    assert read_artifact_attachment(tmp_path, "analysis_sidecar") == payload


def test_undeclared_attachment_is_not_readable(tmp_path):
    save_products(tmp_path, {})
    (tmp_path / "stray.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactNotFoundError, match="no attachment"):
        read_artifact_attachment(tmp_path, "stray.json")
    with pytest.raises(ArtifactNotFoundError, match="no attachment"):
        read_artifact_attachment(tmp_path, "missing")


def test_attachment_path_escape_is_rejected(tmp_path):
    save_products(
        tmp_path,
        {},
        provenance={
            "attachments": [
                {
                    "name": "evil",
                    "path": "../outside.json",
                    "media_type": "application/json",
                }
            ]
        },
    )

    with pytest.raises(ArtifactCorruptError, match="invalid path"):
        read_artifact_attachment(tmp_path, "evil")


def test_list_artifact_attachments_is_metadata_only(tmp_path):
    save_products(
        tmp_path,
        {},
        provenance={
            "attachments": [
                {
                    "name": "config_snapshot",
                    "path": "config_snapshot.json",
                    "media_type": "application/json",
                },
                {
                    "name": "fit_results",
                    "path": "exports/fit_results.csv",
                    "media_type": "text/csv",
                },
            ]
        },
    )
    (tmp_path / "config_snapshot.json").write_text('{"a": 1}', encoding="utf-8")
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "fit_results.csv").write_bytes(b"x,y\n1,2\n")

    infos = list_artifact_attachments(tmp_path)

    assert [(i.name, i.media_type) for i in infos] == [
        ("config_snapshot", "application/json"),
        ("fit_results", "text/csv"),
    ]
    assert [i.size for i in infos] == [8, 8]


def test_list_artifact_attachments_rejects_invalid_path(tmp_path):
    save_products(
        tmp_path,
        {},
        provenance={
            "attachments": [
                {"name": "evil", "path": "../outside.json", "media_type": "text/csv"}
            ]
        },
    )

    with pytest.raises(ArtifactCorruptError, match="invalid path"):
        list_artifact_attachments(tmp_path)
