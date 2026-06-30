"""Tests for SDE result postprocessing."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from qphase.main import app
from qphase_sde.postprocess import (
    export_postprocess_bundle,
    fit_lorentzian,
    lorentzian_with_baseline,
    postprocess_run,
)
from qphase_sde.result import SDEResult
from typer.testing import CliRunner


def test_fit_lorentzian_recovers_synthetic_peak():
    axis = np.linspace(-3.0, 3.0, 301)
    psd = lorentzian_with_baseline(axis, center=0.6, gamma=0.15, amplitude=2.0, base=0.2)

    result = fit_lorentzian(axis, psd)

    assert result.status == "ok"
    assert result.center == np.float64(0.6)
    assert np.isclose(result.linewidth, 0.3)
    assert result.R2 > 0.999


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