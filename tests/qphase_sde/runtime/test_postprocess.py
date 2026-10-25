"""Tests for SDE result postprocessing via the lorentz_fitter analyzer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.errors import QPhaseError
from qphase.core.registry import discovery
from qphase.core.result_loader import load_result
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig
from qphase.data import ArtifactManifest, load_products, save_products
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
from qphase_sde.analyser.psd import PsdAnalyser, _build_psd_products
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


def test_lorentz_fitter_analyze_directory():
    """LorentzFitter can process a run directory directly."""
    psd = _make_scan_product()

    analyzer = LorentzFitter(
        LorentzFitterConfig(
            scan_param="epsilon",
            mode=0,
        )
    )
    product = analyzer.execute(
        {"psd": psd}, backend=NumpyBackend(), label="fit"
    )["fit"]
    out = product.attributes["payload_meta"]
    rows = out["fit_rows"]
    assert len(rows) == 2
    assert all(
        row["uncertainty_source"] == "residual_covariance"
        for row in rows
    )
    # The merged PSD view that used to reach disk as CSV is the payload itself.
    assert sorted(out["psd_columns"]) == ["0.1", "0.2"]
    assert np.asarray(out["axis"]).shape == np.asarray(out["psd_columns"]["0.1"]).shape


def test_lorentz_fitter_transfers_psd_sem_to_uncertainty():
    """Aggregated PSD SEM propagates to fit uncertainty and remains auditable."""
    psd = _make_scan_product(with_uncertainty=True)
    analyzer = LorentzFitter(
        LorentzFitterConfig(
            scan_param="epsilon",
            mode=0,
            uncertainty="required",
        )
    )

    product = analyzer.execute(
        {"psd": psd}, backend=NumpyBackend(), label="fit"
    )["fit"]

    out = product.attributes["payload_meta"]
    for row in out["fit_rows"]:
        assert row["uncertainty_source"] == "psd_sem_sandwich"
        assert float(row["center_std"]) > 0.0
        assert float(row["linewidth_std"]) > 0.0
        assert float(row["amplitude_std"]) > 0.0
        assert float(row["base_std"]) > 0.0
        assert float(row["peak_intensity_std"]) > 0.0

    # Every per-scan SEM trace the CSV used to carry stays in the payload,
    # aligned with the reference axis.
    assert sorted(out["psd_sem_columns"]) == ["0.1", "0.2"]
    axis = np.asarray(out["axis"])
    for scan_key in ("0.1", "0.2"):
        sem = np.asarray(out["psd_sem_columns"][scan_key])
        assert sem.shape == axis.shape
        assert np.all(np.isfinite(sem))
        assert np.all(sem > 0.0)


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


def _save_source_artifact(
    tmp_path: Path, *, generic: bool, ancestor: str, published: str
) -> Path:
    """Save one single-point PSD artifact with a parent chain and a stated id."""
    job_dir = tmp_path / ("generic_source" if generic else "sde_source")
    job_dir.mkdir()
    analyser = PsdAnalyser(
        measurement={"complex_field": {"channels": [0], "representation": "complex"}}
    )
    products = _build_psd_products(
        _psd_payload(0.1), scan_size=None, label="psd", analyser=analyser
    )
    assert products is not None
    meta = {"params": {"epsilon": 0.1, "value": 0.1}, "job_name": job_dir.name}
    if generic:
        # A generic artifact states its id only in the manifest, never in a
        # declared-artifact-id method; the derived run must still find it.
        save_products(
            job_dir,
            products,
            provenance={"job_name": job_dir.name, "meta": meta},
            artifact_id=published,
            parents=[ancestor],
            layout="single",
        )
    else:
        SDEDataBundle(
            products,
            SDEProvenance(dt=1.0),
            meta=meta,
            parents=(ancestor,),
            artifact_id=published,
        ).save(job_dir)
    return job_dir


def _run_analyze_job(source_dir: Path, temp_project):
    job_list = WorkflowSpec(
        schema="qphase.workflow/2",
        id="test-workflow",
        title="Test Workflow",
        jobs=[
            JobConfig(
                name="fit",
                input={"from": str(source_dir), "mode": "dataset"},
                engine={"sde": {"mode": "analyze"}},
                analyser={"lorentz_fitter": {"scan_param": "epsilon", "mode": 0}},
            )
        ],
    )
    discovery.discover_plugins()
    discovery.discover_local_plugins()
    results = Scheduler(
        system_config=SystemConfig(), project=temp_project
    ).run(job_list)
    return results


def test_the_simulate_analyze_merge_keeps_both_parent_chains():
    """Adding the retained trajectory never drops the analysed or source chain."""
    from types import SimpleNamespace

    from qphase_sde.runtime.typed_execution import _merge_simulation_trajectory

    analyser = PsdAnalyser(
        measurement={"complex_field": {"channels": [0], "representation": "complex"}}
    )
    products = _build_psd_products(
        _psd_payload(0.1), scan_size=None, label="psd", analyser=analyser
    )
    assert products is not None
    source = SDEDataBundle(
        {},
        SDEProvenance(dt=1.0),
        parents=("ancestor-parent",),
        artifact_id="source-artifact",
    )
    analysed = SDEDataBundle(
        products,
        SDEProvenance(dt=1.0),
        parents=("ancestor-parent", "source-artifact"),
    )
    merged = _merge_simulation_trajectory(
        SimpleNamespace(config=None), source, analysed
    )
    assert merged.parents == ("ancestor-parent", "source-artifact")
    assert merged.declared_artifact_id() is None


def test_three_levels_of_real_analysis_keep_the_whole_ancestor_chain(
    tmp_path, temp_project
):
    """Peaks -> peak_paths: every real analysis level adds an id and keeps ancestors."""
    ancestor = "0123456789abcdef0123456789abcdef"
    source_id = "fedcba9876543210fedcba9876543210"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    SDEDataBundle(
        {"psd": _make_scan_product()},
        SDEProvenance(dt=1.0),
        scan_axes={"epsilon": [0.1, 0.2]},
        scan_shape=(2,),
        meta={"params": {"epsilon": 0.1}, "job_name": "source"},
        parents=(ancestor,),
        artifact_id=source_id,
    ).save(source_dir)

    def analyze(source: Path, name: str, analyser: dict):
        job = JobConfig(
            name=name,
            input={"from": str(source), "mode": "dataset"},
            engine={"sde": {"mode": "analyze"}},
            analyser=analyser,
        )
        spec = WorkflowSpec(
            schema="qphase.workflow/2", id=f"wf-{name}", title=name, jobs=[job]
        )
        discovery.discover_plugins()
        discovery.discover_local_plugins()
        result = Scheduler(
            system_config=SystemConfig(), project=temp_project
        ).run(spec)[0]
        assert result.success is True, result
        return result

    peaks = analyze(
        source_dir, "peaks_job", {"peaks": {"scan_param": "epsilon", "psd_key": "psd"}}
    )
    paths = analyze(
        peaks.job_dir,
        "paths_job",
        {"peak_paths": {"scan_param": "epsilon", "candidates_key": "peaks"}},
    )
    peaks_manifest = ArtifactManifest.read(peaks.job_dir)
    paths_manifest = ArtifactManifest.read(paths.job_dir)
    # Level two names the source and its ancestor; level three names both.
    assert peaks_manifest.parents == [ancestor, source_id]
    assert paths_manifest.parents == [ancestor, source_id, peaks_manifest.artifact_id]
    assert len(
        {source_id, peaks_manifest.artifact_id, paths_manifest.artifact_id}
    ) == 3
    restored = load_result("paths", paths.job_dir)
    assert isinstance(restored, SDEDataBundle)
    assert restored.parents == tuple(paths_manifest.parents)
    assert restored.declared_artifact_id() == paths_manifest.artifact_id
    assert paths_manifest.artifact_id not in {ancestor, source_id}


@pytest.mark.parametrize("generic_source", [False, True])
def test_analyze_inherits_the_upstream_parent_chain(
    tmp_path, temp_project, generic_source
):
    """A derived analyze artifact names its source and the source's ancestors."""
    ancestor = "0123456789abcdef0123456789abcdef"
    published = "fedcba9876543210fedcba9876543210"
    source_dir = _save_source_artifact(
        tmp_path, generic=generic_source, ancestor=ancestor, published=published
    )
    results = _run_analyze_job(source_dir, temp_project)
    assert len(results) == 1 and results[0].success is True

    manifest = ArtifactManifest.read(results[0].job_dir)
    # The source's own ancestors first, then the source artifact itself.
    assert manifest.parents == [ancestor, published]
    # The derived result never borrows the id of the artifact it came from.
    assert manifest.artifact_id != published

    restored = load_result("fit", results[0].job_dir)
    assert restored.parents == (ancestor, published)
    # Restored, the derived bundle declares its own published id, never the source's.
    assert restored.declared_artifact_id() == manifest.artifact_id
    assert restored.metadata["input_artifact_id"] == published
    assert "artifact_id" not in restored.metadata
    assert restored.metadata["mode"] == "analyze"
    # A chain computed from this derived artifact would start from its id, so the
    # source's ancestors are not dropped when the derived run is analysed again.
    assert (
        restored.parents[-1] == published
        and manifest.artifact_id not in restored.parents
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
    products = load_products(results[0].job_dir)
    assert "lorentz_fitter" in products, sorted(products)
    fit = products["lorentz_fitter"]
    centers = np.asarray(fit.handle("center").materialize())
    r2 = np.asarray(fit.handle("R2").materialize())
    assert centers.shape == r2.shape
    assert centers.size == 1, "expected one fit row for the restored point"
    assert np.all(np.isfinite(centers))
    assert np.all(r2 > 0.0)
    assert np.all(fit.handle("row_valid").materialize())


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
    analyzer = PsdAnalyser(
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
    analyzer = PsdAnalyser(
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
