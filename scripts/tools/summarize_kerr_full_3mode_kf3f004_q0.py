"""Summarize the five-point kf3f_004 closure-calibration Q0."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qphase.backend.numpy_backend import NumpyBackend  # noqa: E402
from qphase.data import Dataset, load_bundle, stack_datasets  # noqa: E402
from qphase_cam.bifurcation_result import CAMBifurcationScanResult  # noqa: E402
from qphase_cam.core.fpgen import FPGenDynamicsAdapter  # noqa: E402
from qphase_cam.postprocessor.local_response import (  # noqa: E402
    LocalResponseValidation,
    LocalResponseValidationConfig,
)
from qphase_sde.analyser.allan_scaling import AllanScalingAnalyzer  # noqa: E402
from qphase_sde.analyser.band_limited_carrier import (  # noqa: E402
    estimate_band_limited_carrier,
)
from qphase_sde.products import add_scan_parameter_coordinates  # noqa: E402
from validate_kerr_full_3mode_frequency_search import (  # noqa: E402
    _frequency,
    _snapshot_model_params,
)

from models.kerr_full_3mode import KerrFull3ModeModel  # noqa: E402

DEFAULT_SESSION = ROOT / "runs/2026/09/2026-09-04T23-42-22_020d34"
DEFAULT_OUTPUT = (
    ROOT
    / "reports/models/kerr_full_3mode/frequency_search_20260904"
    / "kf3f_004_q0"
)
V0_JOB = (
    ROOT
    / "reports/models/kerr_full_3mode/frequency_search_20260904/v0"
    / "kerr_full_3mode_d0_omega_b_gamma_b_gamma_c"
)
POINTS = (
    (1.0e-4, "e1em4", 29_700.431815934735),
    (1.0e-3, "e1em3", 24_956.70194227244),
    (1.0e-2, "e1em2", 17_627.806219326176),
    (3.0e-2, "e3em2", 13_536.338814127814),
    (1.0e-1, "e1em1", 9_137.24051020011),
)


def _array(product: Dataset, name: str) -> np.ndarray:
    return np.asarray(product.handle(name).materialize())


def _payload(product: Dataset) -> dict[Any, Any]:
    payload: dict[Any, Any] = {}
    for variable in product.variables:
        target = payload
        parts = variable.name.split(".")
        for part in parts[:-1]:
            key: Any = int(part) if part.isdigit() else part
            target = target.setdefault(key, {})
        leaf: Any = int(parts[-1]) if parts[-1].isdigit() else parts[-1]
        target[leaf] = product.handle(variable.name).materialize()
    for key, value in product.attributes.get("payload_meta", {}).items():
        target = payload
        parts = key.split(".")
        for part in parts[:-1]:
            nested: Any = int(part) if part.isdigit() else part
            target = target.setdefault(nested, {})
        leaf = int(parts[-1]) if parts[-1].isdigit() else parts[-1]
        target[leaf] = value
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted(set().union(*(row for row in rows))) if rows else ()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _cam_points() -> dict[float, dict[str, Any]]:
    result = CAMBifurcationScanResult.load(next(V0_JOB.glob("*_v0.npz")))
    source_dir = Path(str(result.meta["v0_source_npz"])).parent
    model = KerrFull3ModeModel(**_snapshot_model_params(source_dir))
    adapter = FPGenDynamicsAdapter.from_model(model)
    local = LocalResponseValidation(
        LocalResponseValidationConfig(
            epsilon_min=1.0e-9,
            epsilon_max=1.0e-1,
            epsilon_points=11,
            precision_digits=100,
            max_iterations=60,
            residual_tolerance=1.0e-35,
            fit_points=5,
        )
    )
    candidate = 1
    branch = 3
    branches = result.candidates.branches
    assert branches is not None
    params = local._critical_params(result, candidate, model)
    decimals = local._critical_state_decimals(result.candidates, candidate)
    critical = local._critical_state(result.candidates, candidate, decimals)
    exponent = float(
        branches.exponent_numerator[branch] / branches.exponent_denominator[branch]
    )
    tangent = np.asarray(
        result.candidates.diagnostics["state_tangent_vector"][candidate], dtype=float
    )
    coefficient = float(branches.amplitude[branch]) * tangent
    output = {}
    for epsilon, _, _ in POINTS:
        local_params = dict(params)
        local_params["omega_b"] += epsilon
        state, residual, converged = local._newton(
            adapter,
            critical + coefficient * epsilon**exponent,
            local_params,
            critical_decimals=decimals,
            leading_coefficient=coefficient,
            epsilon_magnitude=epsilon,
            exponent=exponent,
        )
        if not converged:
            raise RuntimeError(f"CAM solve failed at epsilon={epsilon:g}")
        matrix = np.asarray(adapter.state_matrix(state, local_params), dtype=complex)
        output[epsilon] = {
            "matrix": matrix,
            "trace": float(np.real(np.trace(matrix))),
            "frequencies": [
                _frequency(adapter, model, state, local_params, mode)
                for mode in range(3)
            ],
            "residual": residual,
        }
    return output


def _allan_rows(bundles: dict[float, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for mode in range(3):
        ordered = sorted(bundles.items())
        source = stack_datasets(
            [
                product.point_view(scan=0)
                if "scan" in {axis.name for axis in product.axes}
                else product
                for _, bundle in ordered
                for product in [bundle.products["allan_variance"]]
            ],
            axis="scan",
        )
        source = add_scan_parameter_coordinates(
            source,
            {"omega_b": np.asarray([0.5 + epsilon for epsilon, _ in ordered])},
        )
        analyser = AllanScalingAnalyzer(
            scan_param="omega_b",
            critical_value=0.5,
            side="positive",
            mode=mode,
            window_selection="latest",
            white_slope_min=-1.2,
            white_slope_max=-0.8,
            min_local_r2=0.9,
            min_tau_decades=0.25,
            min_averaging_samples=16,
            max_relative_sem=0.5,
            min_independent_windows=8,
            min_scaling_points=3,
            target_scaling_decades=0.5,
            max_scaling_reduced_chi2=30.0,
            bootstrap_samples=500,
            bootstrap_seed=202609110 + mode,
            fit_frequency=False,
            export=[],
        )
        products = analyser.execute(
            {"allan_variance": source},
            backend=NumpyBackend(),
            label="allan_scaling",
        )
        result = _payload(products["allan_scaling"])
        output.extend({"mode": mode, **row} for row in result["rows"])
    return output


def _summaries(
    bundles: dict[float, Any], cam: dict[float, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    closure_rows = []
    carrier_rows = []
    for epsilon, bundle in sorted(bundles.items()):
        coherence = bundle.products["coherence_matrix"]
        moments = bundle.products["moment_statistics"]
        passage = bundle.products["first_passage"]
        psd = bundle.products["psd"]
        matrix = _array(coherence, "matrix")
        cam_matrix = cam[epsilon]["matrix"]
        normalized = matrix / np.trace(matrix)
        cam_normalized = cam_matrix / np.trace(cam_matrix)
        occupation = _array(moments, "occupation")
        covariance = _array(moments, "occupation_covariance")
        near_nyquist = [
            float(
                np.max(
                    _array(
                        bundle.products["allan_variance"],
                        f"mode_results.{mode}.phase_increment."
                        "near_nyquist_fraction_per_trajectory",
                    )
                )
            )
            for mode in range(3)
        ]
        closure_rows.append(
            {
                "epsilon": epsilon,
                "omega_b": 0.5 + epsilon,
                "cam_trace": cam[epsilon]["trace"],
                "sde_trace": float(np.real(np.trace(matrix))),
                "sde_over_cam_trace": float(
                    np.real(np.trace(matrix)) / cam[epsilon]["trace"]
                ),
                "cam_r_aa": float(np.real(cam_matrix[0, 0])),
                "sde_r_aa": float(occupation[0]),
                "sde_over_cam_r_aa": float(
                    occupation[0] / np.real(cam_matrix[0, 0])
                ),
                "normalized_shape_distance": float(
                    np.linalg.norm(normalized - cam_normalized)
                ),
                "purity": float(_array(coherence, "purity")),
                "principal_fraction": float(
                    _array(coherence, "principal_fraction")
                ),
                "first_last_matrix_distance": float(
                    _array(coherence, "time_blocks.first_last_matrix_distance")
                ),
                "first_passage_hits": int(_array(passage, "n_hit")),
                "kerr_frequency_std": float(
                    0.002 * math.sqrt(max(float(covariance[0, 0]), 0.0))
                ),
                "g2_mode_0": float(_array(moments, "g2")[0, 0]),
                "near_nyquist_max_mode_0": near_nyquist[0],
                "near_nyquist_max_mode_1": near_nyquist[1],
                "near_nyquist_max_mode_2": near_nyquist[2],
            }
        )
        axis = _array(psd, "axis")
        spectra = _array(psd, "psd")
        selected = (axis >= -0.5) & (axis <= 1.5)
        for mode in range(3):
            local_axis = axis[selected]
            local_psd = spectra[selected, mode]
            peak = int(np.argmax(local_psd))
            estimate, _ = estimate_band_limited_carrier(
                axis,
                spectra[:, mode],
                freq_min=-0.5,
                freq_max=1.5,
                maximum_lag=2048.0,
            )
            carrier_rows.append(
                {
                    "epsilon": epsilon,
                    "omega_b": 0.5 + epsilon,
                    "mode": mode,
                    "cam_frequency": cam[epsilon]["frequencies"][mode],
                    "raw_psd_peak": float(local_axis[peak]),
                    "raw_peak_minus_cam": float(
                        local_axis[peak] - cam[epsilon]["frequencies"][mode]
                    ),
                    "carrier_status": estimate.status,
                    "carrier_frequency": estimate.frequency,
                    "carrier_minus_cam": (
                        estimate.frequency - cam[epsilon]["frequencies"][mode]
                        if math.isfinite(estimate.frequency)
                        else math.nan
                    ),
                    "peak_hwhm": estimate.peak_hwhm,
                    "spectral_width": estimate.spectral_width,
                    "platform_count": len(estimate.platforms),
                    "diagnostic_uncertainty": estimate.diagnostic_uncertainty,
                    "phase_fit_rms": estimate.phase_fit_rms,
                }
            )
    return closure_rows, carrier_rows


def _plot_psd(
    bundles: dict[float, Any], cam: dict[float, dict[str, Any]], output: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True, sharey=True)
    colors = plt.get_cmap("viridis")
    for index, (epsilon, bundle) in enumerate(sorted(bundles.items())):
        psd = bundle.products["psd"]
        axis = _array(psd, "axis")
        spectra = _array(psd, "psd")
        selected = (axis >= -0.5) & (axis <= 1.5)
        for mode, panel in enumerate(axes):
            values = spectra[selected, mode]
            normalized = values / max(float(np.max(values)), 1.0e-300)
            color = colors(index / 4.0)
            panel.semilogy(
                axis[selected], normalized, color=color, linewidth=1.0,
                label=f"{epsilon:g}",
            )
            panel.axvline(
                cam[epsilon]["frequencies"][mode],
                color=color,
                linestyle="--",
                linewidth=0.9,
                alpha=0.9,
            )
    for mode, panel in enumerate(axes):
        panel.set_title(f"mode {mode}")
        panel.set_xlabel(r"$\omega$")
        panel.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("normalized PSD")
    axes[-1].legend(title=r"$|\epsilon|$", fontsize=8)
    figure.suptitle("kf3f_004 Q0: all-mode PSD; dashed lines are CAM frequencies")
    figure.tight_layout()
    figure.savefig(output / "q0_all_mode_psd_cam.png", dpi=180)
    plt.close(figure)


def _plot_closure(rows: list[dict[str, Any]], output: Path) -> None:
    epsilon = np.asarray([row["epsilon"] for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    axes[0].semilogx(
        epsilon, [row["sde_over_cam_trace"] for row in rows], "o-", label="trace"
    )
    axes[0].semilogx(
        epsilon, [row["sde_over_cam_r_aa"] for row in rows], "s-", label="Raa"
    )
    axes[0].axhline(1.0, color="#555555", linestyle="--")
    axes[0].set_ylabel("SDE / CAM")
    axes[0].legend()
    axes[1].semilogx(
        epsilon, [row["normalized_shape_distance"] for row in rows], "o-"
    )
    axes[1].set_ylabel("normalized-matrix distance")
    axes[2].semilogx(epsilon, [row["purity"] for row in rows], "o-")
    axes[2].set_ylabel("purity")
    for axis in axes:
        axis.set_xlabel(r"$|\epsilon|$")
        axis.grid(True, which="both", alpha=0.2)
    figure.suptitle("kf3f_004 Q0 closure diagnostics")
    figure.tight_layout()
    figure.savefig(output / "q0_closure.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    session = args.session.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundles = {
        epsilon: load_bundle(
            session / f"kerr_full_3mode_kf3f_004_q0_{label}_sim"
        )
        for epsilon, label, _ in POINTS
    }
    cam = _cam_points()
    closure_rows, carrier_rows = _summaries(bundles, cam)
    allan_rows = _allan_rows(bundles)
    _write_csv(output / "q0_closure_points.csv", closure_rows)
    _write_csv(output / "q0_carrier_points.csv", carrier_rows)
    _write_csv(output / "q0_allan_points.csv", allan_rows)
    _plot_psd(bundles, cam, output)
    _plot_closure(closure_rows, output)
    primary = [row for row in carrier_rows if row["mode"] == 2]
    epsilon = np.asarray([float(row["epsilon"]) for row in primary])
    raw_peak = np.asarray([float(row["raw_psd_peak"]) for row in primary])
    free_fit = least_squares(
        lambda values: values[0] + values[1] * epsilon ** values[2] - raw_peak,
        np.asarray([0.8, -1.3, 1.0 / 3.0]),
        bounds=([-10.0, -100.0, 0.05], [10.0, 100.0, 1.5]),
    ).x
    fit_diagnostics = {}
    for name, exponent in (("cubic", 1.0 / 3.0), ("linear", 1.0)):
        design = np.column_stack((np.ones_like(epsilon), epsilon**exponent))
        coefficient = np.linalg.lstsq(design, raw_peak, rcond=None)[0]
        residual = design @ coefficient - raw_peak
        fit_diagnostics[f"{name}_rmse"] = float(
            np.sqrt(np.mean(residual**2))
        )
    summary = {
        "schema": "qphase.kerr-full-kf3f004-q0-summary/1",
        "session": str(session),
        "epsilon": [epsilon for epsilon, _, _ in POINTS],
        "n_traj": 32,
        "primary_readout": "mode_2",
        "first_passage_hits": [row["first_passage_hits"] for row in closure_rows],
        "trace_ratio_range": [
            min(row["sde_over_cam_trace"] for row in closure_rows),
            max(row["sde_over_cam_trace"] for row in closure_rows),
        ],
        "r_aa_ratio_range": [
            min(row["sde_over_cam_r_aa"] for row in closure_rows),
            max(row["sde_over_cam_r_aa"] for row in closure_rows),
        ],
        "shape_distance_range": [
            min(row["normalized_shape_distance"] for row in closure_rows),
            max(row["normalized_shape_distance"] for row in closure_rows),
        ],
        "purity_range": [
            min(row["purity"] for row in closure_rows),
            max(row["purity"] for row in closure_rows),
        ],
        "mode_2_carrier_status": [row["carrier_status"] for row in primary],
        "mode_2_carrier_minus_cam": [row["carrier_minus_cam"] for row in primary],
        "mode_2_raw_peak_minus_cam": [row["raw_peak_minus_cam"] for row in primary],
        "mode_2_raw_peak_cam_rmse": float(
            np.sqrt(
                np.mean(
                    [float(row["raw_peak_minus_cam"]) ** 2 for row in primary]
                )
            )
        ),
        "mode_2_raw_peak_free_fit": {
            "intercept": float(free_fit[0]),
            "coefficient": float(free_fit[1]),
            "exponent": float(free_fit[2]),
            "rmse": float(
                np.sqrt(
                    np.mean(
                        (free_fit[0] + free_fit[1] * epsilon ** free_fit[2] - raw_peak)
                        ** 2
                    )
                )
            ),
            **fit_diagnostics,
        },
        "accepted_white_fm_by_mode": {
            str(mode): sum(
                bool(row["accepted"]) for row in allan_rows if row["mode"] == mode
            )
            for mode in range(3)
        },
    }
    (output / "q0_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
