"""Golden-value baselines for the ``lorentz_fitter`` analyser (Wave 5B).

The existing suite only asserts scan-point labels for ``LorentzFitter``
(``test_bundle_restore.py``); the fitted numeric values are unpinned. These
tests capture the present outputs of ``fit_lorentzian`` and of the full
analyser on synthetic fixtures so the Wave 5B rewrite (``analyser.peaks`` +
``peak_finder.lorentz``) can be checked for numerical equivalence.

Tolerance policy: ``scipy.optimize.curve_fit`` runs an iterative TRF solver,
so fitted parameters and covariance-derived standard deviations are pinned
with ``rtol=1e-8`` (deterministic per environment; last-ulp drift across
BLAS/SciPy builds is possible). The noise-free case additionally asserts the
analytically exact parameters at ``abs=1e-9``. Golden values captured with
SciPy from uv.lock, seeds 20260909/20260910.
"""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.lorentz_fitter import (
    LorentzFitter,
    LorentzFitterConfig,
    fit_lorentzian,
)
from qphase_sde.result import SDEResult

RTOL = 1e-8


def _lorentzian(
    axis: np.ndarray, center: float, gamma: float, amplitude: float, base: float
) -> np.ndarray:
    return amplitude * (gamma**2 / ((axis - center) ** 2 + gamma**2)) + base


def test_fit_lorentzian_clean_baseline():
    axis = np.linspace(-2.0, 2.0, 401)
    psd = _lorentzian(axis, -0.31, 0.07, 4.2, 0.15)

    result = fit_lorentzian(axis, psd)

    assert result.status == "ok"
    assert result.uncertainty_source == "residual_covariance"
    assert result.center == pytest.approx(-0.31, abs=1e-9)
    assert result.linewidth == pytest.approx(0.14, abs=1e-9)
    assert result.base == pytest.approx(0.15, abs=1e-9)
    assert result.peak_intensity - result.base == pytest.approx(4.2, abs=1e-9)
    assert result.R2 == 1.0
    assert result.center_std < 1e-10
    assert np.isnan(result.reduced_chi2)
    # Golden values captured from the current implementation.
    assert result.linewidth == pytest.approx(0.14000000000000404, rel=RTOL)
    assert result.peak_intensity == pytest.approx(4.349999999999894, rel=RTOL)


def test_fit_lorentzian_noisy_with_sigma_baseline():
    axis = np.linspace(-2.0, 2.0, 401)
    rng = np.random.default_rng(20260909)
    sigma = np.full_like(axis, 0.02)
    psd = (
        _lorentzian(axis, 0.42, 0.11, 6.5, 0.05) + rng.standard_normal(axis.size) * 0.02
    )

    result = fit_lorentzian(axis, psd, psd_sigma=sigma)

    assert result.status == "ok"
    assert result.uncertainty_source == "psd_sem_sandwich"
    assert result.center == pytest.approx(0.4200504643985979, rel=RTOL)
    assert result.center_std == pytest.approx(0.00011483450518002873, rel=RTOL)
    assert result.linewidth == pytest.approx(0.21953159648377765, rel=RTOL)
    assert result.linewidth_std == pytest.approx(0.00035509994676968495, rel=RTOL)
    assert result.base == pytest.approx(0.04907250213258484, rel=RTOL)
    assert result.base_std == pytest.approx(0.0011907000418411794, rel=RTOL)
    assert result.peak_intensity - result.base == pytest.approx(
        6.510984137892673, rel=RTOL
    )
    assert result.amplitude_std == pytest.approx(0.006814399781020187, rel=RTOL)
    assert result.peak_intensity == pytest.approx(6.560056640025258, rel=RTOL)
    assert result.peak_intensity_std == pytest.approx(0.0068880743497300185, rel=RTOL)
    assert result.R2 == pytest.approx(0.999727218746823, rel=RTOL)
    assert result.reduced_chi2 == pytest.approx(1.0552457444486487, rel=RTOL)


def test_fit_lorentzian_window_and_clip_baseline():
    axis = np.linspace(-2.0, 2.0, 401)
    rng = np.random.default_rng(20260909)
    sigma = np.full_like(axis, 0.02)
    psd = (
        _lorentzian(axis, 0.42, 0.11, 6.5, 0.05) + rng.standard_normal(axis.size) * 0.02
    )

    result = fit_lorentzian(
        axis,
        psd,
        psd_sigma=sigma,
        fit_window=0.5,
        clip_by_std=True,
        clip_sigma=4.0,
    )

    assert result.status == "ok"
    assert result.uncertainty_source == "psd_sem_sandwich"
    assert result.center == pytest.approx(0.420051425906219, rel=RTOL)
    assert result.center_std == pytest.approx(0.00011518532204142451, rel=RTOL)
    assert result.linewidth == pytest.approx(0.2188308397942939, rel=RTOL)
    assert result.linewidth_std == pytest.approx(0.000652186653609097, rel=RTOL)
    assert result.base == pytest.approx(0.0588079630302443, rel=RTOL)
    assert result.base_std == pytest.approx(0.008494956103618188, rel=RTOL)
    assert result.peak_intensity - result.base == pytest.approx(
        6.505011153168784, rel=RTOL
    )
    assert result.amplitude_std == pytest.approx(0.008973549315762695, rel=RTOL)
    assert result.peak_intensity == pytest.approx(6.563819116199029, rel=RTOL)
    assert result.peak_intensity_std == pytest.approx(0.0074507156401270565, rel=RTOL)
    assert result.R2 == pytest.approx(0.9998919074025878, rel=RTOL)
    assert result.reduced_chi2 == pytest.approx(1.027084798428129, rel=RTOL)


def _scan_results() -> dict[str, SDEResult]:
    axis = np.linspace(-2.0, 2.0, 401)
    rng = np.random.default_rng(20260910)
    results = {}
    for epsilon, (center, gamma, amplitude) in [
        (0.1, (-0.28, 0.06, 3.0)),
        (0.2, (-0.18, 0.09, 5.0)),
    ]:
        trace = (
            _lorentzian(axis, center, gamma, amplitude, 0.1)
            + rng.standard_normal(axis.size) * 0.01
        )
        sem = np.full_like(axis, 0.01)
        results[f"point_{epsilon}"] = SDEResult(
            analysis={
                "psd": {
                    "axis": axis,
                    "psd": trace[:, None],
                    "psd_sem": sem[:, None],
                    "modes": [0],
                    "orientation": "phase_decreasing",
                },
                "dist": {"distributions": {0: {"hist": np.arange(4.0), "label": "d0"}}},
                "pdist": {
                    "distributions": {0: {"hist": np.arange(3.0), "label": "p0"}}
                },
            },
            meta={"params": {"epsilon": epsilon}, "job_name": f"point_{epsilon}"},
        )
    return results


def test_lorentz_fitter_analyzer_scan_rows_baseline(tmp_path):
    fitter = LorentzFitter(
        config=LorentzFitterConfig(
            scan_param="epsilon",
            mode=0,
            uncertainty="auto",
            export_dist=True,
            export=["fit_results.csv", "psd_merged.csv"],
            output_dir=str(tmp_path),
        )
    )

    out = fitter.analyze(_scan_results(), NumpyBackend()).data_dict

    rows = out["fit_rows"]
    assert len(rows) == 2

    assert rows[0]["epsilon"] == 0.1
    assert rows[0]["job_name"] == "point_0.1"
    assert rows[0]["orientation"] == "phase_decreasing"
    assert rows[0]["status"] == "ok"
    assert rows[0]["uncertainty_source"] == "psd_sem_sandwich"
    assert rows[0]["warning"] == ""
    assert rows[0]["center"] == pytest.approx(-0.2799383937542897, rel=RTOL)
    assert rows[0]["center_std"] == pytest.approx(9.185895844024733e-05, rel=RTOL)
    assert rows[0]["linewidth"] == pytest.approx(0.11957048751342425, rel=RTOL)
    assert rows[0]["linewidth_std"] == pytest.approx(0.00027258154867046587, rel=RTOL)
    assert rows[0]["base"] == pytest.approx(0.10065494728791423, rel=RTOL)
    assert rows[0]["amplitude"] == pytest.approx(3.0035202179808587, rel=RTOL)
    assert rows[0]["amplitude_std"] == pytest.approx(0.0046151076876809576, rel=RTOL)
    assert rows[0]["peak_intensity"] == pytest.approx(3.104175165268773, rel=RTOL)
    assert rows[0]["R2"] == pytest.approx(0.999419700860541, rel=RTOL)
    assert rows[0]["reduced_chi2"] == pytest.approx(1.1274132851675074, rel=RTOL)

    assert rows[1]["epsilon"] == 0.2
    assert rows[1]["job_name"] == "point_0.2"
    assert rows[1]["status"] == "ok"
    assert rows[1]["uncertainty_source"] == "psd_sem_sandwich"
    assert rows[1]["center"] == pytest.approx(-0.18010397192848482, rel=RTOL)
    assert rows[1]["center_std"] == pytest.approx(6.772974180240798e-05, rel=RTOL)
    assert rows[1]["linewidth"] == pytest.approx(0.18011144296069376, rel=RTOL)
    assert rows[1]["linewidth_std"] == pytest.approx(0.00020612760748990003, rel=RTOL)
    assert rows[1]["base"] == pytest.approx(0.09985035337073353, rel=RTOL)
    assert rows[1]["amplitude"] == pytest.approx(4.999555444941051, rel=RTOL)
    assert rows[1]["amplitude_std"] == pytest.approx(0.0037608307390084935, rel=RTOL)
    assert rows[1]["peak_intensity"] == pytest.approx(5.099405798311785, rel=RTOL)
    assert rows[1]["R2"] == pytest.approx(0.9998827238260919, rel=RTOL)
    assert rows[1]["reduced_chi2"] == pytest.approx(0.905557539314271, rel=RTOL)

    assert sorted(out["psd_columns"]) == ["0.1", "0.2"]
    assert sorted(out["psd_sem_columns"]) == ["0.1", "0.2"]
    assert sorted(out["written"]) == [
        "dist_merged",
        "fit_results",
        "pdist_merged",
        "psd_merged",
    ]
    assert (tmp_path / "fit_results.csv").exists()
    assert (tmp_path / "psd_merged.csv").exists()
    assert (tmp_path / "dist_merged.npz").exists()
    assert (tmp_path / "pdist_merged.pkl").exists()
