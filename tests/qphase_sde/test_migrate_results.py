"""Tests for the one-way 1.x result migration tool (runtime/migrate.py)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from qphase.data import load_products
from qphase_sde.runtime.migrate import (
    LegacyFormatError,
    migrate_legacy_result,
    migrate_scan_artifact,
)

pytestmark = pytest.mark.integration

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "legacy_1x"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_hashes() -> dict[Path, str]:
    return {
        path: _sha256(path)
        for path in FIXTURE_DIR.rglob("*")
        if path.is_file() and path.name != "_generate.py"
    }


def test_migrate_single_sde_result(tmp_path):
    before = _fixture_hashes()
    out = tmp_path / "migrated"
    report = migrate_legacy_result(FIXTURE_DIR / "legacy_sde_result.npz", out)

    assert report.products == ["psd", "trajectories"]
    assert report.warnings == []
    assert report.sources == {
        "legacy_sde_result.npz": _sha256(FIXTURE_DIR / "legacy_sde_result.npz")
    }
    manifest = json.loads((out / "artifact_manifest.json").read_text())
    migration = manifest["provenance"]["migration"]
    assert migration["legacy_format"] == "sde_result/1"
    assert migration["sources"] == report.sources
    assert manifest["parents"] == list(report.sources.values())

    products = load_products(out)
    alpha = products["trajectories"].handle("alpha").materialize()
    assert alpha.shape == (1, 2, 32, 1)
    assert alpha.dtype == np.complex128
    with np.load(FIXTURE_DIR / "legacy_sde_result.npz", allow_pickle=True) as npz:
        np.testing.assert_array_equal(alpha[0], npz["data"])
    psd = products["psd"]
    assert [variable.name for variable in psd.variables] == ["frequency", "power"]
    np.testing.assert_array_equal(
        psd.handle("frequency").materialize(), [0.0, 1.0]
    )
    # Fixture content is untouched by the conversion.
    assert _fixture_hashes() == before


def test_migrate_masked_trajectory_preserves_valid_lengths(tmp_path):
    out = tmp_path / "masked"
    report = migrate_legacy_result(FIXTURE_DIR / "masked_trajectory.npz", out)

    assert report.products == ["trajectories"]
    products = load_products(out)
    traj = products["trajectories"]
    alpha = traj.handle("alpha").materialize()
    assert alpha.shape == (1, 4, 128, 2)
    valid = traj.handle("valid_length").materialize()
    np.testing.assert_array_equal(valid, [[128, 96, 128, 64]])
    with np.load(FIXTURE_DIR / "masked_trajectory.npz") as npz:
        np.testing.assert_array_equal(alpha[0], npz["trajectories"])
    # NaN padding survives beyond each valid length.
    assert np.isnan(alpha[0, 1, 96:]).all()
    assert np.isfinite(alpha[0, 1, :96]).all()


def test_migrate_scan_artifact_streams_per_point(tmp_path):
    before = _fixture_hashes()
    out = tmp_path / "scan"
    report = migrate_scan_artifact(FIXTURE_DIR / "artifact_manifest_v2.json", out)

    assert report.products == ["trajectories", "psd"]
    assert report.warnings == []
    assert "artifact_manifest_v2.json" in report.sources
    assert len(report.sources) == 4  # manifest + 3 shards

    manifest = json.loads((out / "artifact_manifest.json").read_text())
    migration = manifest["provenance"]["migration"]
    assert migration["legacy_format"] == "sde_scan/2"
    assert migration["scan"]["shape"] == [3]
    assert migration["scan"]["n_traj_per_point"] == 2
    assert migration["sde"]["scan_grid"] == {"delta": [0.5, 1.0, 1.5]}

    products = load_products(out)
    alpha = products["trajectories"].handle("alpha").materialize()
    assert alpha.shape == (3, 2, 16, 1)
    assert alpha.dtype == np.complex128
    # Every point chunk matches its source shard exactly.
    for point in range(3):
        shard = FIXTURE_DIR / "scan_per_point" / "scan" / f"point_{point:06d}.npz"
        with np.load(shard, allow_pickle=True) as npz:
            np.testing.assert_array_equal(alpha[point], npz["data"])
    # Lazy point view: only the selected point is materialized.
    view = products["trajectories"].point_view(scan=1)
    np.testing.assert_array_equal(view.handle("alpha").materialize(), alpha[1])
    psd = products["psd"]
    assert {variable.name for variable in psd.variables} == {
        "frequency",
        "power",
    }
    assert psd.handle("power").materialize().shape == (3, 1)
    assert _fixture_hashes() == before


def test_migrate_rejects_unknown_object_payload(tmp_path):
    source = tmp_path / "unknown_object.npz"
    np.savez(
        source,
        data=np.zeros((1, 4, 1), dtype=np.complex128),
        t0=0.0,
        dt=0.1,
        mystery=np.array({"custom": object()}, dtype=object),
    )
    with pytest.raises(LegacyFormatError, match="unknown object payloads"):
        migrate_legacy_result(source, tmp_path / "out")


def test_migrate_rejects_unknown_keys(tmp_path):
    source = tmp_path / "unknown_keys.npz"
    np.savez(source, data=np.zeros((1, 4, 1)), extra=np.arange(3))
    with pytest.raises(LegacyFormatError, match="unknown keys"):
        migrate_legacy_result(source, tmp_path / "out")


def test_migrate_never_overwrites_output(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale").write_text("x")
    with pytest.raises(LegacyFormatError, match="not empty"):
        migrate_legacy_result(FIXTURE_DIR / "legacy_sde_result.npz", out)
    with pytest.raises(LegacyFormatError, match="overlaps"):
        migrate_legacy_result(
            FIXTURE_DIR / "legacy_sde_result.npz", FIXTURE_DIR
        )


def test_migrate_scan_rejects_non_per_point_layout(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "layout": "single",
                "shape": [1],
                "files": ["x.npz"],
            }
        )
    )
    with pytest.raises(LegacyFormatError, match="per_point"):
        migrate_scan_artifact(manifest, tmp_path / "out")


def test_migrate_adapter_handles_non_mapping_payload(tmp_path):
    source = tmp_path / "custom_payload.npz"
    np.savez(
        source,
        t0=0.0,
        dt=0.1,
        meta=np.array({"seed": 3}, dtype=object),
        analysis=np.array({"custom": "opaque_string_payload"}, dtype=object),
        trajectory_meta=np.array({}, dtype=object),
    )

    def adapter(key: str, value):
        assert key == "custom"
        assert value == "opaque_string_payload"
        return {"flattened": np.arange(2.0)}

    out = tmp_path / "out"
    report = migrate_legacy_result(source, out, adapter=adapter)
    assert report.products == ["custom"]
    products = load_products(out)
    np.testing.assert_array_equal(
        products["custom"].handle("flattened").materialize(), [0.0, 1.0]
    )


def test_migrate_without_adapter_rejects_non_mapping_payload(tmp_path):
    # A non-mapping analysis payload that is JSON-safe (dict of strings)
    # lands in payload_meta without an adapter; a truly opaque payload has
    # no numeric leaves and is reported, never pickled.
    source = tmp_path / "opaque_payload.npz"
    np.savez(
        source,
        t0=0.0,
        dt=0.1,
        meta=np.array({}, dtype=object),
        analysis=np.array({"notes": {"text": "keep me"}}, dtype=object),
        trajectory_meta=np.array({}, dtype=object),
    )
    report = migrate_legacy_result(source, tmp_path / "out")
    # No numeric leaves: product cannot form, warning is recorded.
    assert report.products == []
    assert [warning.code for warning in report.warnings] == [
        "analysis-product-dropped"
    ]
