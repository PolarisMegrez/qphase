"""Audit branch-initialized, independent-point Kerr-full Q1 sessions."""

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

from qphase.backend.numpy_backend import NumpyBackend  # noqa: E402
from qphase.data import Dataset, load_bundle, stack_datasets  # noqa: E402
from qphase_sde.analyser.allan_scaling import AllanScalingAnalyzer  # noqa: E402
from qphase_sde.analyser.band_limited_carrier import (  # noqa: E402
    BandLimitedCarrierAnalyzer,
)
from qphase_sde.analyser.spectral_ridge import SpectralRidgeAnalyzer  # noqa: E402
from qphase_sde.products import add_scan_parameter_coordinates  # noqa: E402

SEARCH = ROOT / "reports/models/kerr_full_3mode/frequency_search_20260904"


def _array(product: Dataset, name: str) -> np.ndarray:
    return np.asarray(product.handle(name).materialize())


def _nested_set(payload: dict[Any, Any], dotted: str, value: Any) -> None:
    target = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        key: Any = int(part) if part.isdigit() else part
        target = target.setdefault(key, {})
    leaf: Any = int(parts[-1]) if parts[-1].isdigit() else parts[-1]
    target[leaf] = value


def _payload(product: Dataset) -> dict[Any, Any]:
    payload: dict[Any, Any] = {}
    for variable in product.variables:
        _nested_set(payload, variable.name, product.handle(variable.name).materialize())
    for key, value in product.attributes.get("payload_meta", {}).items():
        _nested_set(payload, key, value)
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted(set().union(*(row for row in rows))) if rows else ()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _matrix(values: list[list[str]]) -> np.ndarray:
    return np.asarray([[complex(value) for value in row] for row in values])


def _scan_product(
    bundles: dict[int, Any], preregistration: dict[str, Any], product_name: str
) -> Dataset:
    products = []
    scan_values = []
    for item in preregistration["points"]:
        index = int(item["index"])
        product = bundles[index].products[product_name]
        products.append(
            product.point_view(scan=0)
            if "scan" in {axis.name for axis in product.axes}
            else product
        )
        scan_values.append(float(item["scan_value"]))
    stacked = stack_datasets(products, axis="scan")
    return add_scan_parameter_coordinates(
        stacked, {"omega_b": np.asarray(scan_values)}
    )


def _frequency_products(
    bundles: dict[int, Any], preregistration: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = _scan_product(bundles, preregistration, "psd")
    all_cam = [
        frequency
        for item in preregistration["points"]
        for frequency in item["cam_frequency"]
    ]
    lower = math.floor((min(all_cam) - 0.75) * 2.0) / 2.0
    upper = math.ceil((max(all_cam) + 0.75) * 2.0) / 2.0
    ridge_products = SpectralRidgeAnalyzer(
        scan_param="omega_b",
        readouts=[0, 1, 2],
        freq_min=lower,
        freq_max=upper,
        minimum_prominence_fraction=0.01,
        tracking_enabled=True,
        tracking_path_count=3,
        tracking_max_cost_delta=2.0,
        export=[],
    ).execute(
        {"psd": source},
        backend=NumpyBackend(),
        label="spectral_ridge",
    )
    ridge = _payload(ridge_products["spectral_ridge"])
    ridge_rows = list(ridge["ridge_rows"])
    carrier_rows = []
    for mode in range(3):
        products = BandLimitedCarrierAnalyzer(
            scan_param="omega_b",
            readout=mode,
            freq_min=lower,
            freq_max=upper,
            maximum_lag=2048.0,
            tracking_enabled=True,
            center={
                "maximum_neighbor_fraction": 0.45,
                "spectral_ridge": {
                    "scan_param": "omega_b",
                    "readouts": [mode],
                    "freq_min": lower,
                    "freq_max": upper,
                    "minimum_prominence_fraction": 0.01,
                    "tracking_path_count": 3,
                    "tracking_max_cost_delta": 2.0,
                },
            },
            export=[],
        ).execute(
            {"psd": source},
            backend=NumpyBackend(),
            label="band_limited_carrier",
        )
        result = _payload(products["band_limited_carrier"])
        carrier_rows.extend(result["carrier_rows"])
    return ridge_rows, carrier_rows


def _allan_rows(
    bundles: dict[int, Any], preregistration: dict[str, Any]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    source = _scan_product(bundles, preregistration, "allan_variance")
    for mode in range(3):
        analyser = AllanScalingAnalyzer(
            scan_param="omega_b",
            critical_value=float(
                preregistration["points"][0]["scan_value"]
                - preregistration["epsilon_side"]
                * preregistration["points"][0]["epsilon"]
            ),
            side="positive" if preregistration["epsilon_side"] > 0 else "negative",
            mode=mode,
            window_selection="latest",
            white_slope_min=-1.2,
            white_slope_max=-0.8,
            min_local_r2=0.9,
            min_tau_decades=0.25,
            min_averaging_samples=16,
            max_relative_sem=0.5,
            min_independent_windows=8,
            min_scaling_points=7,
            target_scaling_decades=2.0,
            max_scaling_reduced_chi2=30.0,
            bootstrap_samples=1000,
            bootstrap_seed=202609400 + mode,
            fit_frequency=False,
            normal_form={"n": 3, "k": 1, "m": 0, "observable_order": 1},
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


def _state_rows(
    bundles: dict[int, Any], preregistration: dict[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for item in preregistration["points"]:
        index = int(item["index"])
        bundle = bundles[index]
        coherence = bundle.products["coherence_matrix"]
        moments = bundle.products["moment_statistics"]
        passage = bundle.products["first_passage"]
        allan = bundle.products["allan_variance"]
        matrix = _array(coherence, "matrix")
        cam = _matrix(item["cam_matrix"])
        occupation = _array(moments, "occupation")
        covariance = _array(moments, "occupation_covariance")
        output.append(
            {
                "index": index,
                "epsilon": float(item["epsilon"]),
                "omega_b": float(item["scan_value"]),
                "holdout": bool(item["holdout"]),
                "cam_trace": float(item["cam_trace"]),
                "sde_trace": float(np.real(np.trace(matrix))),
                "trace_ratio": float(
                    np.real(np.trace(matrix)) / np.real(np.trace(cam))
                ),
                "cam_r_aa": float(np.real(cam[0, 0])),
                "sde_r_aa": float(occupation[0]),
                "r_aa_ratio": float(occupation[0] / np.real(cam[0, 0])),
                "normalized_shape_distance": float(
                    np.linalg.norm(matrix / np.trace(matrix) - cam / np.trace(cam))
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
                    2.0
                    * abs(float(item["params"]["chi"]))
                    * math.sqrt(max(float(covariance[0, 0]), 0.0))
                ),
                "g2_mode_0": float(_array(moments, "g2")[0, 0]),
                **{
                    f"near_nyquist_max_mode_{mode}": float(
                        np.max(
                            _array(
                                allan,
                                f"mode_results.{mode}.phase_increment."
                                "near_nyquist_fraction_per_trajectory",
                            )
                        )
                    )
                    for mode in range(3)
                },
            }
        )
    return output


def _fit_models(
    epsilon: np.ndarray,
    observed: np.ndarray,
    cam: np.ndarray,
    holdout: np.ndarray,
) -> dict[str, Any]:
    train = ~holdout
    span = float(np.ptp(cam))
    offset = float(np.mean(observed[train] - cam[train]))
    cam_residual = observed[holdout] - cam[holdout] - offset

    predictions = {}
    for name, exponent in (("fixed_cubic", 1.0 / 3.0), ("linear", 1.0)):
        design = np.column_stack(
            (np.ones(np.count_nonzero(train)), epsilon[train] ** exponent)
        )
        coefficient = np.linalg.lstsq(design, observed[train], rcond=None)[0]
        predictions[name] = (
            coefficient[0] + coefficient[1] * epsilon[holdout] ** exponent
        )
    free = least_squares(
        lambda values: (
            values[0] + values[1] * epsilon[train] ** values[2] - observed[train]
        ),
        np.asarray([observed[0], observed[-1] - observed[0], 1.0 / 3.0]),
        bounds=([-20.0, -200.0, 0.05], [20.0, 200.0, 1.5]),
    ).x
    predictions["free_power"] = free[0] + free[1] * epsilon[holdout] ** free[2]
    residuals = {
        name: observed[holdout] - prediction
        for name, prediction in predictions.items()
    }
    return {
        "cam_offset_train": offset,
        "cam_span": span,
        "cam_holdout_rss": float(cam_residual @ cam_residual),
        "cam_holdout_rms_fraction": float(np.sqrt(np.mean(cam_residual**2)))
        / max(span, 1.0e-300),
        "cam_holdout_max_fraction": float(np.max(np.abs(cam_residual)))
        / max(span, 1.0e-300),
        "fixed_cubic_holdout_rss": float(
            residuals["fixed_cubic"] @ residuals["fixed_cubic"]
        ),
        "linear_holdout_rss": float(residuals["linear"] @ residuals["linear"]),
        "cubic_over_linear_rss": float(
            residuals["fixed_cubic"] @ residuals["fixed_cubic"]
        )
        / max(float(residuals["linear"] @ residuals["linear"]), 1.0e-300),
        "free_power_exponent": float(free[2]),
        "free_power_holdout_rss": float(
            residuals["free_power"] @ residuals["free_power"]
        ),
    }


def _frequency_rows(
    preregistration: dict[str, Any],
    ridge_rows: list[dict[str, Any]],
    carrier_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    point_by_value = {
        float(item["scan_value"]): item for item in preregistration["points"]
    }
    carrier_by_key = {
        (float(row["omega_b"]), int(row["readout"])): row for row in carrier_rows
    }
    output = []
    for ridge in ridge_rows:
        value = float(ridge["omega_b"])
        mode = int(ridge["readout"])
        point = point_by_value[value]
        carrier = carrier_by_key.get((value, mode), {})
        output.append(
            {
                "index": int(point["index"]),
                "epsilon": float(point["epsilon"]),
                "omega_b": value,
                "holdout": bool(point["holdout"]),
                "mode": mode,
                "cam_frequency": float(point["cam_frequency"][mode]),
                "ridge_frequency": float(ridge["frequency"]),
                "ridge_frequency_std": float(ridge["frequency_std"]),
                "ridge_status": ridge["status"],
                "ridge_candidate_count": int(ridge["candidate_count"]),
                "ridge_ambiguity_width": float(ridge["ambiguity_width"]),
                "ridge_plateau_width": float(ridge["plateau_width"]),
                "carrier_status": carrier.get("status", "missing"),
                "carrier_frequency": carrier.get("frequency", math.nan),
                "carrier_diagnostic_uncertainty": carrier.get(
                    "diagnostic_uncertainty", math.nan
                ),
                "carrier_peak_hwhm": carrier.get("peak_hwhm", math.nan),
                "carrier_platform_count": carrier.get("platform_count", 0),
            }
        )
    return output


def _plot_frequency(
    rows: list[dict[str, Any]], family: str, primary_mode: int, output: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True)
    for mode, axis in enumerate(axes):
        local = sorted(
            (row for row in rows if row["mode"] == mode),
            key=lambda row: row["epsilon"],
        )
        epsilon = np.asarray([row["epsilon"] for row in local])
        axis.semilogx(
            epsilon,
            [row["cam_frequency"] for row in local],
            "--",
            color="#555555",
            label="CAM",
        )
        axis.errorbar(
            epsilon,
            [row["ridge_frequency"] for row in local],
            yerr=[row["ridge_frequency_std"] for row in local],
            fmt="o-",
            color="#007f5f",
            markersize=4,
            label="tracked PSD ridge",
        )
        accepted = [row for row in local if row["carrier_status"] == "ok"]
        if accepted:
            axis.scatter(
                [row["epsilon"] for row in accepted],
                [row["carrier_frequency"] for row in accepted],
                marker="s",
                color="#c44e52",
                label="band-limited carrier",
            )
        holdout = [row for row in local if row["holdout"]]
        axis.scatter(
            [row["epsilon"] for row in holdout],
            [row["ridge_frequency"] for row in holdout],
            facecolors="none",
            edgecolors="#111111",
            s=64,
            linewidths=1.0,
            label="holdout",
        )
        axis.set_title(f"mode {mode}" + (" (primary)" if mode == primary_mode else ""))
        axis.set_xlabel(r"$|\epsilon|$")
        axis.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel(r"carrier $\Omega$")
    axes[-1].legend(fontsize=8)
    figure.suptitle(f"{family} Q1 frequency response")
    figure.tight_layout()
    figure.savefig(output / "q1_frequency_response.png", dpi=180)
    plt.close(figure)


def _plot_closure(rows: list[dict[str, Any]], family: str, output: Path) -> None:
    epsilon = [row["epsilon"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    axes[0].semilogx(epsilon, [row["trace_ratio"] for row in rows], "o-", label="trace")
    axes[0].semilogx(epsilon, [row["r_aa_ratio"] for row in rows], "s-", label="Raa")
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
    figure.suptitle(f"{family} Q1 closure diagnostics")
    figure.tight_layout()
    figure.savefig(output / "q1_closure.png", dpi=180)
    plt.close(figure)


def _plot_psd(
    bundles: dict[int, Any], preregistration: dict[str, Any], output: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    colors = plt.get_cmap("viridis")
    for item in preregistration["points"]:
        index = int(item["index"])
        psd = bundles[index].products["psd"]
        frequency = _array(psd, "axis")
        spectrum = _array(psd, "psd")
        cam_frequency = np.asarray(item["cam_frequency"], dtype=float)
        lower = float(np.min(cam_frequency) - 0.75)
        upper = float(np.max(cam_frequency) + 0.75)
        selected = (frequency >= lower) & (frequency <= upper)
        color = colors(index / 12.0)
        for mode, axis in enumerate(axes):
            values = spectrum[selected, mode]
            axis.semilogy(
                frequency[selected],
                values / max(float(np.max(values)), 1.0e-300),
                color=color,
                linewidth=0.8,
            )
            axis.axvline(
                cam_frequency[mode],
                color=color,
                linestyle="--",
                linewidth=0.65,
                alpha=0.8,
            )
    for mode, axis in enumerate(axes):
        axis.set_title(f"mode {mode}")
        axis.set_xlabel(r"$\omega$")
        axis.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("normalized PSD")
    figure.suptitle(
        f"{preregistration['root_family']} Q1 PSD; dashed lines are CAM frequencies"
    )
    figure.tight_layout()
    figure.savefig(output / "q1_all_mode_psd_cam.png", dpi=180)
    plt.close(figure)


def _plot_allan(rows: list[dict[str, Any]], family: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 4.5))
    markers = ("o", "s", "^")
    for mode in range(3):
        accepted = sorted(
            (row for row in rows if row["mode"] == mode and row["accepted"]),
            key=lambda row: row["epsilon"],
        )
        rejected = [
            row for row in rows if row["mode"] == mode and not row["accepted"]
        ]
        axis.errorbar(
            [row["epsilon"] for row in accepted],
            [row["allan_intensity"] for row in accepted],
            yerr=[row["allan_intensity_sem"] for row in accepted],
            fmt=f"{markers[mode]}-",
            markersize=4,
            capsize=2,
            label=f"mode {mode}",
        )
        if rejected:
            axis.scatter(
                [row["epsilon"] for row in rejected],
                [
                    row["allan_intensity"]
                    if math.isfinite(row["allan_intensity"])
                    else 1.0e-12
                    for row in rejected
                ],
                marker="x",
                color="#c44e52",
            )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$|\epsilon|$")
    axis.set_ylabel(r"pointwise $N_A=\tau\sigma_A^2$")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend()
    axis.set_title(f"{family} Q1 pointwise white-FM Allan intensity")
    figure.tight_layout()
    figure.savefig(output / "q1_allan_scaling.png", dpi=180)
    plt.close(figure)


def _allan_exponent(rows: list[dict[str, Any]], mode: int, minimum: float) -> float:
    selected = [
        row
        for row in rows
        if row["mode"] == mode
        and row["accepted"]
        and float(row["epsilon"]) >= minimum
    ]
    if len(selected) < 3:
        return math.nan
    epsilon = np.asarray([float(row["epsilon"]) for row in selected])
    intensity = np.asarray([float(row["allan_intensity"]) for row in selected])
    return float(np.polyfit(np.log(epsilon), np.log(intensity), 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=("kf3f_004", "kf3f_007"))
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    preregistration = json.loads(
        (SEARCH / f"{args.family}_q1_preregistration.json").read_text("utf-8")
    )
    session = args.session.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else SEARCH / f"{args.family}_q1_independent"
    )
    output.mkdir(parents=True, exist_ok=True)
    bundles = {
        int(item["index"]): load_bundle(
            session
            / f"kerr_full_3mode_{args.family}_q1_p{int(item['index']):02d}_sim"
        )
        for item in preregistration["points"]
    }
    ridge_rows, carrier_rows = _frequency_products(bundles, preregistration)
    frequency_rows = _frequency_rows(preregistration, ridge_rows, carrier_rows)
    state_rows = _state_rows(bundles, preregistration)
    allan_rows = _allan_rows(bundles, preregistration)
    _write_csv(output / "q1_frequency_points.csv", frequency_rows)
    _write_csv(output / "q1_carrier_audit.csv", carrier_rows)
    _write_csv(output / "q1_state_points.csv", state_rows)
    _write_csv(output / "q1_allan_points.csv", allan_rows)
    primary_mode = int(preregistration["primary_mode"])
    primary = sorted(
        (row for row in frequency_rows if row["mode"] == primary_mode),
        key=lambda row: row["epsilon"],
    )
    epsilon = np.asarray([row["epsilon"] for row in primary])
    observed = np.asarray([row["ridge_frequency"] for row in primary])
    cam = np.asarray([row["cam_frequency"] for row in primary])
    holdout = np.asarray([row["holdout"] for row in primary], dtype=bool)
    fit = _fit_models(epsilon, observed, cam, holdout)
    maximum_phase = max(
        row[f"near_nyquist_max_mode_{mode}"]
        for row in state_rows
        for mode in range(3)
    )
    state_passed = bool(
        sum(row["first_passage_hits"] for row in state_rows) == 0
        and maximum_phase < 1.0e-3
        and max(row["first_last_matrix_distance"] for row in state_rows) < 0.1
        and min(row["trace_ratio"] for row in state_rows) > 0.8
        and max(row["trace_ratio"] for row in state_rows) < 1.2
        and max(row["normalized_shape_distance"] for row in state_rows) < 0.05
    )
    ridge_resolved = [
        bool(
            row["ridge_status"] in {"ok", "flat_top"}
            and math.isfinite(row["ridge_frequency"])
            and row["ridge_candidate_count"] == 1
        )
        for row in primary
    ]
    frequency_passed = bool(
        all(ridge_resolved)
        and fit["cubic_over_linear_rss"] < 1.0
        and fit["cam_holdout_rms_fraction"] < 0.1
        and fit["cam_holdout_max_fraction"] < 0.2
    )
    summary = {
        "schema": "qphase.kerr-full-independent-q1-summary/1",
        "family": args.family,
        "session": str(session),
        "claim_scope": "physical_carrier_frequency",
        "primary_mode": primary_mode,
        "state_passed": state_passed,
        "frequency_passed": frequency_passed,
        "q1_passed": bool(state_passed and frequency_passed),
        "state_gate": {
            "first_passage_hits": sum(
                row["first_passage_hits"] for row in state_rows
            ),
            "maximum_near_nyquist_fraction": maximum_phase,
            "trace_ratio_range": [
                min(row["trace_ratio"] for row in state_rows),
                max(row["trace_ratio"] for row in state_rows),
            ],
            "r_aa_ratio_range": [
                min(row["r_aa_ratio"] for row in state_rows),
                max(row["r_aa_ratio"] for row in state_rows),
            ],
            "maximum_shape_distance": max(
                row["normalized_shape_distance"] for row in state_rows
            ),
            "maximum_first_last_matrix_distance": max(
                row["first_last_matrix_distance"] for row in state_rows
            ),
            "purity_range": [
                min(row["purity"] for row in state_rows),
                max(row["purity"] for row in state_rows),
            ],
        },
        "frequency_gate": {
            **fit,
            "ridge_ok_points": sum(row["ridge_status"] == "ok" for row in primary),
            "ridge_flat_top_points": sum(
                row["ridge_status"] == "flat_top" for row in primary
            ),
            "ridge_resolved_points": sum(ridge_resolved),
            "band_limited_accepted_points": sum(
                row["carrier_status"] == "ok" for row in primary
            ),
            "maximum_ridge_candidate_count": max(
                row["ridge_candidate_count"] for row in primary
            ),
        },
        "allan_accepted_points_by_mode": {
            str(mode): sum(
                bool(row["accepted"]) for row in allan_rows if row["mode"] == mode
            )
            for mode in range(3)
        },
        "primary_allan_diagnostic": {
            "all_accepted_free_exponent": _allan_exponent(
                allan_rows, primary_mode, 0.0
            ),
            "epsilon_ge_1e3_free_exponent": _allan_exponent(
                allan_rows, primary_mode, 1.0e-3
            ),
            "epsilon_ge_1e2_free_exponent": _allan_exponent(
                allan_rows, primary_mode, 1.0e-2
            ),
            "complete_stochastic_scaling_passed": False,
            "reason": "frozen nonlinear CAM order comparison not performed",
        },
    }
    (output / "q1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot_frequency(frequency_rows, args.family, primary_mode, output)
    _plot_closure(state_rows, args.family, output)
    _plot_psd(bundles, preregistration, output)
    _plot_allan(allan_rows, args.family, output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
