"""Tests for SDE result postprocessing via the lorentz_fitter analyzer."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.aggregation import QPHASE_BUNDLE_SCHEMA_VERSION
from qphase.core.config import JobConfig, JobList
from qphase.core.registry import discovery
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig
from qphase_sde.analyser.lorentz_fitter import (
    LorentzFitter,
    LorentzFitterConfig,
    fit_lorentzian,
)
from qphase_sde.analyser.lorentz_fitter import (
    _lorentzian_with_baseline as lorentzian_with_baseline,
)
from qphase_sde.analyser.result import AnalysisResult
from qphase_sde.result import SDEResult


def test_fit_lorentzian_recovers_synthetic_peak():
    axis = np.linspace(-3.0, 3.0, 301)
    psd = lorentzian_with_baseline(
        axis, center=0.6, gamma=0.15, amplitude=2.0, base=0.2
    )

    result = fit_lorentzian(axis, psd)

    assert result.status == "ok"
    assert np.isclose(result.center, 0.6)
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


def test_fit_lorentzian_clip_by_std_ignores_tails():
    """clip_by_std focuses the fit on the central peak and ignores tail bumps."""
    axis = np.linspace(-10.0, 10.0, 1001)
    psd = lorentzian_with_baseline(axis, center=0.0, gamma=0.3, amplitude=2.0, base=0.1)
    # Add two side bumps that are stronger than the central peak tails.
    side_bump = 2.5 * np.exp(-((axis - 5.0) ** 2) / 0.2)
    side_bump += 2.5 * np.exp(-((axis + 5.0) ** 2) / 0.2)
    psd_with_tails = psd + side_bump

    no_clip = fit_lorentzian(axis, psd_with_tails)
    clipped = fit_lorentzian(axis, psd_with_tails, clip_by_std=True, clip_sigma=1.0)

    assert clipped.status == "ok"
    assert np.isclose(clipped.center, 0.0, atol=0.1)
    # Without clipping the side bumps pull the fit away from the true center.
    assert abs(no_clip.center) > abs(clipped.center)


def test_fit_lorentzian_clip_by_std_sigma():
    """A smaller clip_sigma should tighten the fitting window further."""
    axis = np.linspace(-10.0, 10.0, 1001)
    psd = lorentzian_with_baseline(
        axis, center=0.0, gamma=0.5, amplitude=10.0, base=0.1
    )

    wide = fit_lorentzian(axis, psd, clip_by_std=True, clip_sigma=5.0)
    narrow = fit_lorentzian(axis, psd, clip_by_std=True, clip_sigma=1.0)

    assert wide.status == "ok"
    assert narrow.status == "ok"
    assert np.isclose(wide.center, 0.0, atol=0.05)
    assert np.isclose(narrow.center, 0.0, atol=0.05)


def test_lorentz_fitter_config_clip_by_std(tmp_path):
    """LorentzFitter passes clip_by_std/clip_sigma down to fit_lorentzian."""
    run_dir = _make_run_dir(tmp_path)
    analyzer = LorentzFitter(
        LorentzFitterConfig(
            scan_param="epsilon",
            mode=0,
            clip_by_std=True,
            clip_sigma=2.0,
        )
    )
    result = analyzer.analyze(run_dir, backend=NumpyBackend())

    assert isinstance(result, AnalysisResult)
    assert len(result.data_dict["fit_rows"]) == 2
    for row in result.data_dict["fit_rows"]:
        assert row["status"] in {"ok", "low_quality"}


def test_lorentz_fitter_analyze_directory(tmp_path):
    """LorentzFitter can process a run directory directly."""
    run_dir = _make_run_dir(tmp_path)
    output_dir = tmp_path / "exports"

    analyzer = LorentzFitter(
        LorentzFitterConfig(
            scan_param="epsilon",
            mode=0,
            output_dir=str(output_dir),
        )
    )
    result = analyzer.analyze(run_dir, backend=NumpyBackend())

    assert isinstance(result, AnalysisResult)
    assert len(result.data_dict["fit_rows"]) == 2
    assert (output_dir / "fit_results.csv").exists()
    assert (output_dir / "psd_merged.csv").exists()


def test_lorentz_fitter_engine_analyze_mode(tmp_path):
    """The SDE engine can run in analyze mode via the scheduler."""
    run_dir = _make_run_dir(tmp_path)
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
                name="fit",
                input=str(run_dir),
                engine={"sde": {"mode": "analyze"}},
                analyser={"lorentz_fitter": {"scan_param": "epsilon", "mode": 0}},
            )
        ]
    )

    discovery.discover_plugins()
    discovery.discover_local_plugins()

    results = Scheduler(system_config=system_config).run(job_list)

    assert len(results) == 1
    assert results[0].success is True
    assert (results[0].run_dir / "fit_results.csv").exists()
    assert (results[0].run_dir / "psd_merged.csv").exists()


def test_export_dist_schema_version(tmp_path):
    """dist_merged.npz and pdist_merged.pkl must carry a core schema version."""
    from qphase.core.aggregation import write_npz_bundle, write_pkl_bundle

    output_dir = tmp_path / "exports"
    output_dir.mkdir()

    dist_rows = [{"epsilon": 0.1, "job_name": "j1", "histogram": np.ones(5)}]
    pdist_rows = [{"epsilon": 0.1, "job_name": "j1", "histogram": np.ones(5)}]

    dist_path = write_npz_bundle(
        output_dir / "dist_merged.npz",
        dist_list=np.array(dist_rows, dtype=object),
        scan_params=np.array([row["epsilon"] for row in dist_rows], dtype=object),
    )
    pdist_path = write_pkl_bundle(output_dir / "pdist_merged.pkl", pdist_rows)

    dist_data = np.load(dist_path, allow_pickle=True)
    assert dist_data["__schema_version__"] == QPHASE_BUNDLE_SCHEMA_VERSION

    with pdist_path.open("rb") as handle:
        pdist_bundle = pickle.load(handle)
    assert pdist_bundle["__schema_version__"] == QPHASE_BUNDLE_SCHEMA_VERSION
    assert len(pdist_bundle["rows"]) == len(pdist_rows)


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
