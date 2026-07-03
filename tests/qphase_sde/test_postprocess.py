"""Tests for SDE result postprocessing."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
from qphase.core.config import JobConfig, JobList
from qphase.core.registry import registry
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig
from qphase.main import app
from qphase_sde.postprocess import (
    POSTPROCESS_SCHEMA_VERSION,
    PostprocessBundle,
    export_postprocess_bundle,
    fit_lorentzian,
    lorentzian_with_baseline,
    postprocess_run,
)
from qphase_sde.result import SDEResult
from qphase_sde.workflows.postprocess.engine import (
    SDEPostprocessEngine,
    SDEPostprocessEngineConfig,
)
from typer.testing import CliRunner


def test_fit_lorentzian_recovers_synthetic_peak():
    axis = np.linspace(-3.0, 3.0, 301)
    psd = lorentzian_with_baseline(
        axis, center=0.6, gamma=0.15, amplitude=2.0, base=0.2
    )

    result = fit_lorentzian(axis, psd)

    assert result.status == "ok"
    assert result.center == np.float64(0.6)
    assert np.isclose(result.linewidth, 0.3)
    assert result.R2 > 0.999


def test_fit_lorentzian_frequency_range(tmp_path):
    """--freq-min/--freq-max should restrict the data used for fitting."""
    axis = np.linspace(-3.0, 3.0, 301)
    psd = lorentzian_with_baseline(
        axis, center=0.6, gamma=0.15, amplitude=2.0, base=0.2
    )

    result = fit_lorentzian(axis, psd, freq_min=0.0, freq_max=2.0)
    assert result.status == "ok"
    assert np.isclose(result.center, 0.6, atol=0.05)


def test_fit_lorentzian_quality_thresholds():
    """Quality thresholds mark poor fits as low_quality without raising."""
    rng = np.random.default_rng(0)
    axis = np.linspace(-3.0, 3.0, 301)
    psd = lorentzian_with_baseline(
        axis, center=0.6, gamma=0.15, amplitude=2.0, base=0.2
    )
    psd_noisy = psd + rng.normal(0.0, 0.05, size=psd.shape)

    result = fit_lorentzian(axis, psd_noisy, min_r2=0.9999)
    assert result.status == "low_quality"
    assert "R2" in result.error

    result = fit_lorentzian(axis, psd, min_peak_height=10.0)
    assert result.status == "low_quality"
    assert "peak_intensity" in result.error

    result = fit_lorentzian(axis, psd, max_linewidth=0.01)
    assert result.status == "low_quality"
    assert "linewidth" in result.error


def test_postprocess_run_exports_fit_and_merged_psd(tmp_path):
    run_dir = _make_run_dir(tmp_path)

    bundle = postprocess_run(run_dir, scan_param="epsilon", mode=0)
    written = export_postprocess_bundle(
        bundle,
        tmp_path / "exports",
        scan_param="epsilon",
    )

    assert len(bundle.fit_rows) == 2
    assert written["fit_results"].exists()
    assert written["psd_merged"].exists()

    with written["fit_results"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["epsilon"] for row in rows] == ["0.1", "0.2"]
    assert all(row["status"] == "ok" for row in rows)

    with written["psd_merged"].open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == ["frequency", "0.1", "0.2"]


def test_postprocess_run_frequency_range(tmp_path):
    """postprocess_run should pass frequency range to the fit."""
    run_dir = _make_run_dir(tmp_path)
    bundle = postprocess_run(
        run_dir, scan_param="epsilon", mode=0, freq_min=0.0, freq_max=2.0
    )
    assert len(bundle.fit_rows) == 2
    assert all(row["status"] == "ok" for row in bundle.fit_rows)


def test_sde_postprocess_engine_runs_saved_results(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    output_dir = tmp_path / "engine_exports"
    engine = SDEPostprocessEngine(
        SDEPostprocessEngineConfig(
            run_dir=str(run_dir), scan_param="epsilon", output_dir=str(output_dir)
        )
    )

    result = engine.run()

    artifacts = result.analysis["postprocess"]["artifacts"]
    assert result.meta["engine"] == "sde_postprocess"
    assert Path(artifacts["fit_results"]).exists()
    assert Path(artifacts["psd_merged"]).exists()


def test_sde_postprocess_engine_runs_as_scheduler_job(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    registry.register("engine", "sde_postprocess", SDEPostprocessEngine, overwrite=True)
    system_config = SystemConfig(
        paths={
            "output_dir": str(tmp_path / "runs"),
            "global_file": str(tmp_path / "global.yaml"),
            "config_dirs": [str(tmp_path / "configs")],
            "plugin_dirs": [str(tmp_path / "plugins")],
        }
    )
    job_list = JobList(
        jobs=[
            JobConfig(
                name="postprocess",
                engine={
                    "sde_postprocess": {
                        "run_dir": str(run_dir),
                        "scan_param": "epsilon",
                    }
                },
            )
        ]
    )

    results = Scheduler(system_config=system_config).run(job_list)

    assert len(results) == 1
    assert results[0].success is True
    assert (results[0].run_dir / "fit_results.csv").exists()
    assert (results[0].run_dir / "psd_merged.csv").exists()


def test_postprocess_cli_dry_run(tmp_path):
    """--dry-run lists files without writing anything."""
    run_dir = _make_run_dir(tmp_path)
    output_dir = tmp_path / "dry_run_exports"

    result = CliRunner().invoke(
        app,
        [
            "postprocess",
            str(run_dir),
            "--scan-param",
            "epsilon",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Would process 2 result file(s)" in result.stdout
    assert not output_dir.exists() or not any(output_dir.iterdir())


def test_postprocess_cli_smoke(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    output_dir = tmp_path / "cli_exports"

    result = CliRunner().invoke(
        app,
        [
            "postprocess",
            str(run_dir),
            "--scan-param",
            "epsilon",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Processed 2 result file(s)." in result.stdout
    assert (output_dir / "fit_results.csv").exists()
    assert (output_dir / "psd_merged.csv").exists()


def test_export_dist_schema_version(tmp_path):
    """dist_merged.npz and pdist_merged.pkl must carry a schema version."""
    axis = np.linspace(-1.0, 1.0, 11)
    bundle = PostprocessBundle(
        fit_rows=[{"job_name": "j1", "epsilon": 0.1, "status": "ok"}],
        psd_columns={"0.1": np.ones_like(axis)},
        axis=axis,
        dist_rows=[{"epsilon": 0.1, "job_name": "j1", "histogram": np.ones(5)}],
        pdist_rows=[{"epsilon": 0.1, "job_name": "j1", "histogram": np.ones(5)}],
    )
    written = export_postprocess_bundle(
        bundle,
        tmp_path / "exports",
        scan_param="epsilon",
        export_dist=True,
    )

    dist_path = written["dist_merged"]
    dist_data = np.load(dist_path, allow_pickle=True)
    assert dist_data["__schema_version__"] == POSTPROCESS_SCHEMA_VERSION
    assert dist_data["__created_by__"] == "qphase postprocess"

    pdist_path = written["pdist_merged"]
    with pdist_path.open("rb") as handle:
        pdist_bundle = pickle.load(handle)
    assert pdist_bundle["__schema_version__"] == POSTPROCESS_SCHEMA_VERSION
    assert pdist_bundle["__created_by__"] == "qphase postprocess"
    assert len(pdist_bundle["rows"]) == len(bundle.pdist_rows)
    assert pdist_bundle["rows"][0]["job_name"] == bundle.pdist_rows[0]["job_name"]


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    axis = np.linspace(-2.0, 2.0, 201)
    for index, epsilon in enumerate([0.1, 0.2], start=1):
        job_dir = run_dir / f"job_{index:03d}"
        job_dir.mkdir(parents=True)
        psd = lorentzian_with_baseline(
            axis,
            center=epsilon,
            gamma=0.1,
            amplitude=1.0 + epsilon,
            base=0.05,
        )
        result = SDEResult(
            meta={"params": {"epsilon": epsilon}},
            analysis={
                "psd": {
                    "axis": axis,
                    "psd": psd[:, None],
                    "modes": [0],
                    "kind": "complex",
                    "convention": "symmetric",
                }
            },
        )
        result.save(job_dir / f"job_{index:03d}.npz")
    return run_dir
