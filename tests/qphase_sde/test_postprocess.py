"""Tests for SDE result postprocessing via the lorentz_fitter analyzer."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.aggregation import QPHASE_BUNDLE_SCHEMA_VERSION
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.errors import QPhaseError
from qphase.core.registry import discovery
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig
from qphase_sde.analyser.lorentz_fitter import (
    LorentzFitter,
    LorentzFitterConfig,
    _lorentzian_jacobian,
    _squared_weighted_moments,
    fit_lorentzian,
)
from qphase_sde.analyser.lorentz_fitter import (
    _lorentzian_with_baseline as lorentzian_with_baseline,
)
from qphase_sde.analyser.psd import PsdAnalyzer, _build_psd_products
from qphase_sde.contracts.bundle import SDEProvenance
from qphase_sde.products import add_scan_parameter_coordinates
from qphase_sde.result import SDEDataBundle

pytestmark = pytest.mark.integration


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


def test_fit_lorentzian_recovers_fine_grid_peak_from_periodogram_noise():
    axis = np.linspace(-0.6, 0.2, 40_001)
    expected = lorentzian_with_baseline(
        axis, center=-0.313, gamma=0.021, amplitude=60_000.0, base=1_000.0
    )
    rng = np.random.default_rng(22)
    psd = np.maximum(expected * (1.0 + rng.normal(0.0, 0.2, axis.size)), 0.0)

    result = fit_lorentzian(
        axis,
        psd,
        fit_window=0.2,
        freq_min=-0.6,
        freq_max=0.2,
    )

    assert result.status == "ok"
    assert result.center == pytest.approx(-0.313, abs=5e-4)
    assert result.linewidth == pytest.approx(0.042, rel=0.03)
    assert result.R2 > 0.9


def test_lorentzian_jacobian_matches_finite_difference():
    """The Jacobian used for uncertainty propagation matches the fit model."""
    axis = np.linspace(-0.5, 1.5, 101)
    params = np.array([0.4, 0.12, 1.7, 0.08])
    analytic = _lorentzian_jacobian(axis, params)
    numeric = np.empty_like(analytic)
    step = 1e-6
    for index in range(params.size):
        delta = np.zeros(params.shape)
        delta[index] = step
        numeric[:, index] = (
            lorentzian_with_baseline(axis, *(params + delta))
            - lorentzian_with_baseline(axis, *(params - delta))
        ) / (2.0 * step)

    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=1e-8)


def test_squared_lorentzian_moments_recover_center_and_hwhm():
    """Squared Lorentzian moments recover center and half-width."""
    center = 0.4
    gamma = 0.12
    axis = np.linspace(center - 1000 * gamma, center + 1000 * gamma, 200001)
    psd = lorentzian_with_baseline(
        axis, center=center, gamma=gamma, amplitude=2.0, base=0.0
    )

    mean, std = _squared_weighted_moments(axis, psd)

    assert np.isclose(mean, center, atol=1e-10)
    assert np.isclose(std, gamma, rtol=0.01)


def test_fit_lorentzian_propagates_psd_uncertainty_without_reweighting():
    """PSD SEM changes parameter uncertainty without changing the fit."""
    rng = np.random.default_rng(17)
    axis = np.linspace(-1.0, 2.0, 401)
    expected = lorentzian_with_baseline(
        axis, center=0.4, gamma=0.12, amplitude=1.7, base=0.08
    )
    sigma = np.full(axis.shape, 0.02)
    psd = expected + rng.normal(0.0, sigma)

    unweighted = fit_lorentzian(axis, psd, clip_by_std=True, clip_sigma=5.0)
    result = fit_lorentzian(
        axis, psd, psd_sigma=sigma, clip_by_std=True, clip_sigma=5.0
    )
    doubled = fit_lorentzian(
        axis, psd, psd_sigma=2.0 * sigma, clip_by_std=True, clip_sigma=5.0
    )

    assert result.status == "ok"
    assert result.uncertainty_source == "psd_sem_sandwich"
    assert np.isfinite(result.reduced_chi2)
    for field in ("center", "linewidth", "base", "peak_intensity", "R2"):
        assert getattr(result, field) == getattr(unweighted, field)
        assert getattr(doubled, field) == getattr(unweighted, field)
    for field in (
        "center_std",
        "linewidth_std",
        "base_std",
        "amplitude_std",
        "peak_intensity_std",
    ):
        value = getattr(result, field)
        assert np.isfinite(value) and value > 0.0
        assert np.isclose(getattr(doubled, field), 2.0 * value, rtol=1e-5)


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
    psd = _make_scan_product()
    analyzer = LorentzFitter(
        LorentzFitterConfig(
            scan_param="epsilon",
            mode=0,
            clip_by_std=True,
            clip_sigma=2.0,
        )
    )
    product = analyzer.execute(
        {"psd": psd}, backend=NumpyBackend(), label="fit"
    )["fit"]
    payload = product.attributes["payload_meta"]
    assert len(payload["fit_rows"]) == 2
    assert payload["orientation"] == "phase_increasing"
    assert all(
        row["orientation"] == "phase_increasing" for row in payload["fit_rows"]
    )
    for row in payload["fit_rows"]:
        assert row["status"] in {"ok", "low_quality"}


def test_lorentz_fitter_analyze_directory(tmp_path):
    """LorentzFitter can process a run directory directly."""
    psd = _make_scan_product()
    output_dir = tmp_path / "exports"

    analyzer = LorentzFitter(
        LorentzFitterConfig(
            scan_param="epsilon",
            mode=0,
            output_dir=str(output_dir),
        )
    )
    product = analyzer.execute(
        {"psd": psd}, backend=NumpyBackend(), label="fit"
    )["fit"]
    rows = product.attributes["payload_meta"]["fit_rows"]
    assert len(rows) == 2
    assert (output_dir / "fit_results.csv").exists()
    assert (output_dir / "psd_merged.csv").exists()
    assert all(
        row["uncertainty_source"] == "residual_covariance"
        for row in rows
    )


def test_lorentz_fitter_transfers_psd_sem_to_csv(tmp_path):
    """Aggregated PSD SEM propagates to fit uncertainty and remains auditable."""
    psd = _make_scan_product(with_uncertainty=True)
    output_dir = tmp_path / "weighted_exports"
    analyzer = LorentzFitter(
        LorentzFitterConfig(
            scan_param="epsilon",
            mode=0,
            uncertainty="required",
            output_dir=str(output_dir),
        )
    )

    product = analyzer.execute(
        {"psd": psd}, backend=NumpyBackend(), label="fit"
    )["fit"]

    for row in product.attributes["payload_meta"]["fit_rows"]:
        assert row["uncertainty_source"] == "psd_sem_sandwich"
        assert float(row["center_std"]) > 0.0
        assert float(row["linewidth_std"]) > 0.0
        assert float(row["amplitude_std"]) > 0.0
        assert float(row["base_std"]) > 0.0
        assert float(row["peak_intensity_std"]) > 0.0

    with (output_dir / "fit_results.csv").open(newline="", encoding="utf-8") as handle:
        headers = csv.DictReader(handle).fieldnames
    assert headers is not None
    assert {
        "center_std",
        "linewidth_std",
        "amplitude_std",
        "base_std",
        "peak_intensity_std",
        "uncertainty_source",
    }.issubset(headers)

    with (output_dir / "psd_merged.csv").open(newline="", encoding="utf-8") as handle:
        headers = next(csv.reader(handle))
    assert "0.1_sem" in headers
    assert "0.2_sem" in headers


def test_lorentz_fitter_can_require_psd_uncertainty(tmp_path):
    """Required propagation rejects typed PSD products without a SEM field."""
    analyzer = LorentzFitter(
        LorentzFitterConfig(scan_param="epsilon", mode=0, uncertainty="required")
    )

    with pytest.raises(QPhaseError, match="no usable PSD standard error"):
        analyzer.execute(
            {"psd": _make_scan_product()},
            backend=NumpyBackend(),
            label="fit",
        )


def test_external_artifact_routes_to_analyze_product_graph(tmp_path, temp_project):
    """The scheduler restores an external v4 artifact before SDE execution."""
    run_dir = _make_run_dir(tmp_path)
    system_config = SystemConfig()
    job_list = WorkflowSpec(
        schema="qphase.workflow/2",
        id="test-workflow",
        title="Test Workflow",
        jobs=[
            JobConfig(
                name="fit",
                input={"from": str(run_dir / "job_001"), "mode": "dataset"},
                engine={"sde": {"mode": "analyze"}},
                analyser={"lorentz_fitter": {"scan_param": "epsilon", "mode": 0}},
            )
        ],
    )

    discovery.discover_plugins()
    discovery.discover_local_plugins()

    results = Scheduler(system_config=system_config, project=temp_project).run(job_list)

    assert len(results) == 1
    assert results[0].success is True
    assert (results[0].job_dir / "fit_results.csv").exists()
    assert (results[0].job_dir / "psd_merged.csv").exists()


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


def _psd_payload(epsilon: float, *, with_uncertainty: bool = False) -> dict:
    axis = np.linspace(-2.0, 2.0, 201)
    psd = lorentzian_with_baseline(
        axis,
        center=epsilon,
        gamma=0.1,
        amplitude=1.0 + epsilon,
        base=0.05,
    )
    payload = {
        "axis": axis,
        "psd": psd[:, None],
        "modes": [0],
        "sample_dt": 1.0,
        "nyquist": np.pi,
        "convention": "symmetric",
        "orientation": "phase_increasing",
    }
    if with_uncertainty:
        payload["psd_sem"] = np.full((axis.size, 1), 0.01)
        payload["uncertainty"] = {
            "kind": "standard_error",
            "field": "psd_sem",
            "independent_unit": "trajectory",
            "n_independent": 100,
        }
    return payload


def _make_scan_product(*, with_uncertainty: bool = False):
    analyzer = PsdAnalyzer(
        measurement={"complex_field": {"channels": [0], "representation": "complex"}}
    )
    products = _build_psd_products(
        [
            _psd_payload(value, with_uncertainty=with_uncertainty)
            for value in (0.1, 0.2)
        ],
        scan_size=2,
        label="psd",
        analyser=analyzer,
    )
    assert products is not None
    return add_scan_parameter_coordinates(
        products["psd"], {"epsilon": np.asarray([0.1, 0.2])}
    )


def _make_run_dir(tmp_path: Path, *, with_uncertainty: bool = False) -> Path:
    run_dir = tmp_path / "run"
    analyzer = PsdAnalyzer(
        measurement={"complex_field": {"channels": [0], "representation": "complex"}}
    )
    for index, epsilon in enumerate([0.1, 0.2], start=1):
        job_dir = run_dir / f"job_{index:03d}"
        job_dir.mkdir(parents=True)
        products = _build_psd_products(
            _psd_payload(epsilon, with_uncertainty=with_uncertainty),
            scan_size=None,
            label="psd",
            analyser=analyzer,
        )
        assert products is not None

        bundle = SDEDataBundle(
            products,
            SDEProvenance(dt=1.0),
            meta={"params": {"epsilon": epsilon}, "job_name": job_dir.name},
        )
        bundle.save(job_dir)
    return run_dir
