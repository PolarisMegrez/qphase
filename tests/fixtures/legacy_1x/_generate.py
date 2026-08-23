"""Regenerate the frozen qphase_sde 1.x migration fixtures.

Run manually from the repository root when the frozen 1.x formats change:

    .venv/Scripts/python tests/fixtures/legacy_1x/_generate.py

The fixtures are intentionally small and free of model-specific semantics; the
22/26/30-point CSV summaries used for migration reports are *not* committed
(see reports/sde_phase0_contracts_plan.md).
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "packages" / "qphase"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "qphase_sde"))

from qphase.core.scan import ParameterGrid  # noqa: E402
from qphase_sde.result import SDEResult  # noqa: E402
from qphase_sde.scan import SDEScanResult  # noqa: E402
from qphase_sde.state import TrajectorySet  # noqa: E402

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260801)


def _lorentzian(freq: np.ndarray, center: float, width: float, height: float):
    return height * width**2 / ((freq - center) ** 2 + width**2)


def write_peak_fixtures() -> None:
    freq = np.linspace(-4.0, 4.0, 64)
    single = _lorentzian(freq, 0.75, 0.1, 12.0) + 0.05
    double = (
        _lorentzian(freq, -0.5, 0.08, 8.0)
        + _lorentzian(freq, 1.1, 0.15, 5.0)
        + 0.05
    )
    for name, power in (("single_peak_psd", single), ("double_peak_psd", double)):
        np.savez(
            OUT / f"{name}.npz",
            frequency=freq,
            power=power,
            sample_std=np.full_like(freq, 0.02),
            independent_count=np.array(32, dtype=np.int64),
        )


def write_masked_trajectory_fixture() -> None:
    n_traj, n_time, n_modes = 4, 128, 2
    t = np.arange(n_time) * 0.05
    data = np.exp(-0.5j * t[None, :, None]) * (
        1.0 + 0.01 * RNG.standard_normal((n_traj, n_time, n_modes))
    )
    valid_length = np.array([128, 96, 128, 64], dtype=np.int64)
    for traj, length in enumerate(valid_length):
        data[traj, length:, :] = np.nan
    np.savez(
        OUT / "masked_trajectory.npz",
        trajectories=data.astype(np.complex128),
        valid_length=valid_length,
        t0=np.array(0.0),
        dt=np.array(0.05),
    )


def write_legacy_result_fixture() -> None:
    data = RNG.standard_normal((2, 32, 1)) + 1j * RNG.standard_normal((2, 32, 1))
    trajectory = TrajectorySet(data.astype(np.complex128), t0=0.0, dt=0.05)
    result = SDEResult(
        trajectory=trajectory,
        analysis={"psd": {"frequency": [0.0, 1.0], "power": [1.0, 0.5]}},
        meta={"engine": "sde", "label": "legacy-fixture", "seed": 7},
    )
    result.save(OUT / "legacy_sde_result.npz")


def write_sharded_scan_fixture() -> None:
    n_points, n_traj, n_steps, n_modes = 3, 2, 16, 1
    grid = ParameterGrid(
        "cartesian",
        {"delta": np.array([0.5, 1.0, 1.5])},
        {"delta": "model.delta"},
        (n_points,),
    )
    data = RNG.standard_normal((n_points * n_traj, n_steps, n_modes)) + 1j * 0.0
    trajectory = TrajectorySet(data.astype(np.complex128), t0=0.0, dt=0.1)
    combined = SDEResult(
        trajectory=trajectory,
        analysis={"psd": {"frequency": [0.0], "power": [1.0]}},
        meta={"engine": "sde", "seed": 11},
    )
    scan = SDEScanResult(
        combined=combined,
        grid=grid,
        base_params={"model.delta": 0.0},
        n_traj_per_point=n_traj,
    )
    target = OUT / "scan_per_point"
    target.mkdir(exist_ok=True)
    for stale in target.glob("*.npz"):
        stale.unlink()
    report = scan.save_dataset(
        target / "scan", layout="per_point", shard_target_bytes=1 << 20
    )
    assert len(report.files) == n_points


def write_artifact_manifest_fixture() -> None:
    payload = {
        "schema_version": "2.0",
        "result_type": "qphase_sde.scan:SDEScanResult",
        "result_schema": "1.0",
        "axes": {"delta": [0.5, 1.0, 1.5]},
        "shape": [3],
        "layout": "per_point",
        "files": [
            "scan_per_point/scan/point_000000.npz",
            "scan_per_point/scan/point_000001.npz",
            "scan_per_point/scan/point_000002.npz",
        ],
        "loader": "qphase_sde.scan:SDEScanResult.load_dataset",
    }
    (OUT / "artifact_manifest_v2.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def write_plugin_catalog_fixture() -> None:
    import tomllib

    catalog: dict[str, dict[str, str]] = {}
    for package in ("qphase", "qphase_sde"):
        pyproject = REPO_ROOT / "packages" / package / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        catalog[package] = dict(
            sorted(data["project"]["entry-points"]["qphase"].items())
        )
    (OUT / "plugin_catalog.json").write_text(
        json.dumps(catalog, indent=2), encoding="utf-8"
    )


def write_converter_golden() -> None:
    from qphase_sde.contracts.migration import convert_analyser_config

    legacy = {
        "psd": {"method": "welch", "segment_length": 512, "overlap": 0.5},
        "spectral_ridge": {
            "scan_param": "delta",
            "psd_key": "psd",
            "smoothing_scale_bins": [2.0, 4.0, 8.0],
            "tracking_enabled": True,
        },
        "band_limited_carrier": {"min_points": 8},
    }
    (OUT / "converter_legacy_input.json").write_text(
        json.dumps(legacy, indent=2), encoding="utf-8"
    )
    report = convert_analyser_config(legacy)
    expected = {
        "converted": report.converted,
        "diff": report.diff,
        "unmapped": report.unmapped,
        "needs_review": report.needs_review,
    }
    (OUT / "converter_expected.json").write_text(
        json.dumps(expected, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    write_peak_fixtures()
    write_masked_trajectory_fixture()
    write_legacy_result_fixture()
    write_sharded_scan_fixture()
    write_artifact_manifest_fixture()
    write_plugin_catalog_fixture()
    write_converter_golden()
    print(f"fixtures regenerated under {OUT}")
