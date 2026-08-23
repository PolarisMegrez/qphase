"""Golden tests for the frozen qphase_sde 1.x migration fixtures.

The fixtures under ``tests/fixtures/legacy_1x/`` pin the 1.x serialization
formats the 2.x migration tooling must read. Regenerate them only when the
legacy format intentionally changes, via ``_generate.py`` in that directory.
"""

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from qphase_sde.contracts.migration import convert_analyser_config
from qphase_sde.result import SDEResult
from qphase_sde.scan import SDEScanResult

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "legacy_1x"
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_fixtures_stay_small():
    """Migration fixtures must stay small enough to live in git."""
    oversized = [
        path.name
        for path in FIXTURE_DIR.rglob("*")
        if path.is_file() and path.stat().st_size > 256 * 1024
    ]
    assert oversized == []


def test_peak_fixtures_are_plain_arrays():
    """Single/double-peak fixtures carry plain arrays, no pickled objects."""
    for name in ("single_peak_psd", "double_peak_psd"):
        with np.load(FIXTURE_DIR / f"{name}.npz", allow_pickle=False) as npz:
            assert set(npz) >= {"frequency", "power", "sample_std"}
            assert npz["power"].shape == npz["frequency"].shape
            assert np.all(np.isfinite(npz["power"]))
            assert np.all(npz["power"] > 0)
    # The single-peak fixture peaks once; the double-peak fixture twice.
    with np.load(FIXTURE_DIR / "single_peak_psd.npz") as single:
        power = single["power"]
        assert int(np.argmax(power)) not in (0, power.size - 1)


def test_masked_trajectory_fixture_semantics():
    """The masked trajectory fixture carries valid lengths and NaN padding."""
    with np.load(FIXTURE_DIR / "masked_trajectory.npz", allow_pickle=False) as npz:
        data = npz["trajectories"]
        valid_length = npz["valid_length"]
        assert data.dtype == np.complex128
        assert data.shape[0] == valid_length.shape[0]
        for traj, length in enumerate(valid_length):
            assert np.all(np.isfinite(data[traj, :length]))
            assert np.all(np.isnan(data[traj, length:]))


def test_legacy_sde_result_roundtrip():
    """The frozen 1.x SDEResult (object-array npz) still loads via 1.x API."""
    result = SDEResult.load(FIXTURE_DIR / "legacy_sde_result.npz")
    assert result.meta["engine"] == "sde"
    assert result.meta["seed"] == 7
    assert "psd" in result.analysis
    trajectory = result.trajectory
    assert trajectory.data.shape == (2, 32, 1)
    assert trajectory.dt == pytest.approx(0.05)


def test_sharded_scan_fixture_loads_all_points():
    """The frozen per-point sharded scan fixture loads and reassembles."""
    scan = SDEScanResult.load_dataset(FIXTURE_DIR / "scan_per_point" / "scan")
    assert scan.grid.size == 3
    assert scan.shape == (3,)
    assert list(scan.axes) == ["delta"]
    np.testing.assert_allclose(scan.axes["delta"], [0.5, 1.0, 1.5])
    assert scan.combined.trajectory.data.shape[0] == 3 * scan.n_traj_per_point


def test_artifact_manifest_v2_fixture_shape():
    """The artifact-manifest v2 sample carries the frozen v2 keys."""
    manifest = json.loads(
        (FIXTURE_DIR / "artifact_manifest_v2.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "2.0"
    assert set(manifest) >= {
        "result_type",
        "result_schema",
        "axes",
        "shape",
        "layout",
        "files",
        "loader",
    }
    for shard in manifest["files"]:
        assert (FIXTURE_DIR / shard).is_file()


def test_plugin_catalog_snapshot_matches_pyprojects():
    """The frozen plugin catalog matches the installed entry-point metadata."""
    catalog = json.loads(
        (FIXTURE_DIR / "plugin_catalog.json").read_text(encoding="utf-8")
    )
    for package, entries in catalog.items():
        pyproject = REPO_ROOT / "packages" / package / "pyproject.toml"
        current = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
            "entry-points"
        ]["qphase"]
        assert dict(sorted(current.items())) == entries, (
            f"{package} entry points drifted; regenerate the fixture with "
            "tests/fixtures/legacy_1x/_generate.py"
        )


def test_converter_golden():
    """The one-shot converter reproduces the frozen golden output exactly."""
    legacy = json.loads(
        (FIXTURE_DIR / "converter_legacy_input.json").read_text(encoding="utf-8")
    )
    expected = json.loads(
        (FIXTURE_DIR / "converter_expected.json").read_text(encoding="utf-8")
    )
    report = convert_analyser_config(legacy)
    assert report.converted == expected["converted"]
    assert report.diff == expected["diff"]
    assert report.unmapped == expected["unmapped"]
    assert report.needs_review == expected["needs_review"]
