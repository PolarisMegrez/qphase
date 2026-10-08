"""Summarize the Kerr-full frequency-search Q0 SDE calibration run."""

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qphase.backend.numpy_backend import NumpyBackend  # noqa: E402
from qphase.data import Dataset, load_bundle  # noqa: E402
from qphase_sde.analyser.allan_scaling import AllanScalingAnalyzer  # noqa: E402

DEFAULT_SESSION = ROOT / "runs/2026/09/2026-09-04T12-58-16_ce3007"
DEFAULT_BRANCH_SESSION = ROOT / "runs/2026/09/2026-09-04T13-13-27_795097"
DEFAULT_FALLBACK_SESSION = ROOT / "runs/2026/09/2026-09-04T13-19-51_5a074c"
DEFAULT_CAM = (
    ROOT
    / "reports/models/kerr_full_3mode/frequency_search_20260904"
    / "nonlinear_cam_orders.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/models/kerr_full_3mode/frequency_search_20260904/q0"
)
JOBS = (
    {
        "family": "kf3f_019",
        "job": "kerr_full_3mode_q0_kf3f_019_omega_c_positive",
        "critical": -2.0,
        "side": "positive",
    },
    {
        "family": "kf3f_036",
        "job": "kerr_full_3mode_q0_kf3f_036_omega_c_negative",
        "critical": 4.0,
        "side": "negative",
    },
)
BRANCH_JOBS = (
    {
        "family": "kf3f_019",
        "point": "inner",
        "job": "kerr_full_3mode_q0b_kf3f_019_inner",
        "critical": -2.0,
        "side": "positive",
        "omega_c": -1.9996018928294465,
        "cam_trace": 39024.21053449559,
    },
    {
        "family": "kf3f_019",
        "point": "outer",
        "job": "kerr_full_3mode_q0b_kf3f_019_outer",
        "critical": -2.0,
        "side": "positive",
        "omega_c": -1.9841510680753888,
        "cam_trace": 117247.18714422283,
    },
    {
        "family": "kf3f_036",
        "point": "crossover",
        "job": "kerr_full_3mode_q0b_kf3f_036_crossover",
        "critical": 4.0,
        "side": "negative",
        "omega_c": 3.9974881135684904,
        "cam_trace": 56612.00713879783,
    },
)
FALLBACK_JOBS = (
    {
        "family": "kf3f_032",
        "point": "outer-gap-fallback",
        "job": "kerr_full_3mode_q0f_kf3f_032_outer",
        "critical": 2.0,
        "side": "positive",
        "scan_param": "omega_b",
        "scan_value": 2.1,
        "cam_trace": 1531.8291951405063,
    },
)


def _assign_nested(payload: dict[Any, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target = payload
    for part in parts[:-1]:
        key: Any = int(part) if part.isdigit() else part
        child = target.get(key)
        if not isinstance(child, dict):
            child = {}
            target[key] = child
        target = child
    leaf = parts[-1]
    target[int(leaf) if leaf.isdigit() else leaf] = value


def _product_payload(
    product: Dataset, scan_index: int | None = None
) -> dict[Any, Any]:
    payload: dict[Any, Any] = {}
    for variable in product.variables:
        _assign_nested(
            payload, variable.name, product.handle(variable.name).materialize()
        )
    per_point = set(product.attributes.get("per_point_meta", ()))
    for key, value in product.attributes.get("payload_meta", {}).items():
        if key in per_point:
            if scan_index is None:
                raise ValueError(f"product metadata {key!r} requires a scan index")
            value = list(value)[scan_index]
        _assign_nested(payload, key, value)
    return payload


def _materialize(product: Dataset, name: str) -> np.ndarray:
    return np.asarray(product.handle(name).materialize())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted(set().union(*(row for row in rows))) if rows else ()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _allan_analyser(entry: dict[str, Any], mode: int) -> AllanScalingAnalyzer:
    return AllanScalingAnalyzer(
        scan_param=entry.get("scan_param", "omega_c"),
        critical_value=entry["critical"],
        side=entry["side"],
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
        bootstrap_seed=20260904 + mode,
        fit_frequency=False,
        export=[],
    )


def _allan_rows(bundle: Any, entry: dict[str, Any], mode: int) -> list[dict[str, Any]]:
    analyser = _allan_analyser(entry, mode)
    products = analyser.execute(
        bundle.products,
        backend=NumpyBackend(),
        label="allan_scaling",
        metadata=bundle.metadata,
    )
    result = _product_payload(products["allan_scaling"], 0)
    rows = []
    for row in result["rows"]:
        rows.append(
            {
                "root_family": entry["family"],
                "job": entry["job"],
                "mode": mode,
                **row,
            }
        )
    return rows


def _diagnostic_rows(bundle: Any, entry: dict[str, Any]) -> list[dict[str, Any]]:
    coherence = bundle.products["coherence_matrix"]
    moments = bundle.products["moment_statistics"]
    passage = bundle.products["first_passage"]
    scan = _materialize(coherence, "parameter.omega_c")
    matrix_drift = _materialize(coherence, "time_blocks.first_last_matrix_distance")
    relative_matrix = _materialize(coherence, "time_blocks.relative_matrix_distance")
    occupation_drift = _materialize(
        moments, "time_blocks.first_last_occupation_distance"
    )
    relative_occupation = _materialize(
        moments, "time_blocks.relative_occupation_distance"
    )
    purity = _materialize(coherence, "purity")
    principal = _materialize(coherence, "principal_fraction")
    trace = _materialize(coherence, "trace")
    per_trajectory_matrix = _materialize(coherence, "per_trajectory_matrix")
    g2 = _materialize(moments, "g2")
    hits = _materialize(passage, "n_hit")
    rows = []
    for index, scan_value in enumerate(scan):
        trajectory_trace = np.real(
            np.trace(per_trajectory_matrix[index], axis1=-2, axis2=-1)
        )
        row: dict[str, Any] = {
            "root_family": entry["family"],
            "job": entry["job"],
            "omega_c": float(scan_value),
            "epsilon": abs(float(scan_value) - float(entry["critical"])),
            "n_hit": int(hits[index]),
            "trace": float(trace[index]),
            "trajectory_trace_minimum": float(np.min(trajectory_trace)),
            "trajectory_trace_maximum": float(np.max(trajectory_trace)),
            "trajectory_trace_cv": float(
                np.std(trajectory_trace, ddof=1) / np.mean(trajectory_trace)
            ),
            "purity": float(purity[index]),
            "principal_fraction": float(principal[index]),
            "first_last_matrix_distance": float(matrix_drift[index]),
            "max_relative_matrix_distance": float(
                np.nanmax(relative_matrix[index])
            ),
            "first_last_occupation_distance": float(occupation_drift[index]),
            "max_relative_occupation_distance": float(
                np.nanmax(relative_occupation[index])
            ),
        }
        for mode in range(3):
            row[f"g2_mode_{mode}"] = float(g2[index, mode, mode])
        rows.append(row)
    return rows


def _branch_rows(bundle: Any, entry: dict[str, Any]) -> list[dict[str, Any]]:
    coherence = bundle.products["coherence_matrix"]
    moments = bundle.products["moment_statistics"]
    passage = bundle.products["first_passage"]
    trace = float(_materialize(coherence, "trace"))
    per_matrix = _materialize(coherence, "per_trajectory_matrix")
    trajectory_trace = np.real(np.trace(per_matrix, axis1=-2, axis2=-1))
    hit = np.asarray(_materialize(passage, "hit"), dtype=bool)
    hit_times = np.asarray(_materialize(passage, "first_hit_time"), dtype=float)
    g2 = _materialize(moments, "g2")
    scan_param = entry.get("scan_param", "omega_c")
    scan_value = entry.get("scan_value", entry.get("omega_c"))
    output = []
    for mode in range(3):
        products = _allan_analyser(entry, mode).execute(
            bundle.products,
            backend=NumpyBackend(),
            label="allan_scaling",
            metadata=bundle.metadata,
        )
        allan = _product_payload(products["allan_scaling"], 0)["rows"][0]
        output.append(
            {
                "root_family": entry["family"],
                "point": entry["point"],
                "job": entry["job"],
                "mode": mode,
                "scan_param": scan_param,
                "scan_value": scan_value,
                "epsilon": abs(scan_value - entry["critical"]),
                "cam_trace": entry["cam_trace"],
                "sde_trace": trace,
                "sde_over_cam_trace": trace / entry["cam_trace"],
                "trajectory_trace_minimum": float(np.min(trajectory_trace)),
                "trajectory_trace_maximum": float(np.max(trajectory_trace)),
                "n_hit": int(np.count_nonzero(hit)),
                "hit_fraction": float(np.mean(hit)),
                "first_hit_time_minimum": float(np.nanmin(hit_times)),
                "first_hit_time_maximum": float(np.nanmax(hit_times)),
                "purity": float(_materialize(coherence, "purity")),
                "principal_fraction": float(
                    _materialize(coherence, "principal_fraction")
                ),
                "g2": float(g2[mode, mode]),
                "allan_accepted": bool(allan["accepted"]),
                "allan_reason": allan["reason"],
                "allan_tau_start": allan["tau_start"],
                "allan_tau_end": allan["tau_end"],
                "allan_slope": allan["allan_slope"],
                "allan_intensity": allan["allan_intensity"],
                "allan_intensity_sem": allan["allan_intensity_sem"],
            }
        )
    return output


def _controlled_cam(row: dict[str, str]) -> bool:
    return (
        row["status"] == "complete"
        and row["confining"].lower() == "true"
        and float(row["endpoint_density_ratio"]) <= 0.1
        and float(row["edge_mass_fraction"]) <= 0.02
        and float(row["reconstructed_intensity"]) > 0.0
    )


def _cam_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    output = []
    families = {
        entry["family"] for entry in (*JOBS, *FALLBACK_JOBS)
    }
    for family in sorted(families):
        for mode in range(3):
            readout = f"mode_{mode}"
            groups: dict[tuple[int, float], list[dict[str, str]]] = {}
            for row in rows:
                if row["root_family"] != family or row["readout"] != readout:
                    continue
                key = (int(row["epsilon_side"]), float(row["abs_epsilon"]))
                groups.setdefault(key, []).append(row)
            for (side, epsilon), group in sorted(groups.items()):
                controlled = [row for row in group if _controlled_cam(row)]
                latest = (
                    max(controlled, key=lambda row: int(row["order"]))
                    if controlled
                    else None
                )
                reference = group[0]
                output.append(
                    {
                        "root_family": family,
                        "mode": mode,
                        "epsilon_side": side,
                        "epsilon": epsilon,
                        "state_trace": float(reference["state_trace"]),
                        "state_minimum_eigenvalue": float(
                            reference["state_minimum_eigenvalue"]
                        ),
                        "full_ou_intensity": float(reference["full_ou_intensity"]),
                        "controlled_order": int(latest["order"]) if latest else None,
                        "controlled_nonlinear_intensity": (
                            float(latest["reconstructed_intensity"])
                            if latest
                            else math.nan
                        ),
                    }
                )
    return output


def _plot_allan(
    bundles: dict[str, Any], output: Path
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), sharex=False)
    colors = plt.get_cmap("viridis")
    for row_index, entry in enumerate(JOBS):
        bundle = bundles[entry["family"]]
        product = bundle.products["allan_variance"]
        scan = _materialize(product, "parameter.omega_c")
        epsilons = np.abs(scan - float(entry["critical"]))
        for mode in range(3):
            axis = axes[row_index, mode]
            tau = _materialize(product, f"mode_results.{mode}.allan.tau")
            variance = _materialize(
                product,
                f"mode_results.{mode}.allan.nonoverlap_angular_frequency_variance",
            )
            for index, epsilon in enumerate(epsilons):
                local_tau = tau if tau.ndim == 1 else tau[index]
                axis.loglog(
                    local_tau,
                    local_tau * variance[index],
                    color=colors(index / max(len(epsilons) - 1, 1)),
                    linewidth=1.35,
                    label=f"{epsilon:.2g}",
                )
            axis.set_title(f"{entry['family']} / mode {mode}")
            axis.grid(True, which="both", alpha=0.2)
            if mode == 0:
                axis.set_ylabel(r"$\tau\,\sigma_\omega^2(\tau)$")
            if row_index == 1:
                axis.set_xlabel(r"$\tau$")
            if mode == 2:
                axis.legend(title=r"$|\epsilon|$", fontsize=7, title_fontsize=8)
    figure.suptitle("Kerr-full Q0: all-mode Allan intensity")
    figure.tight_layout()
    figure.savefig(output / "q0_allan_intensity.png", dpi=180)
    plt.close(figure)


def _plot_psd(bundles: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), sharex=True, sharey=True)
    colors = plt.get_cmap("plasma")
    for row_index, entry in enumerate(JOBS):
        product = bundles[entry["family"]].products["psd"]
        frequency = _materialize(product, "axis")
        spectrum = _materialize(product, "psd")
        scan = _materialize(product, "parameter.omega_c")
        epsilons = np.abs(scan - float(entry["critical"]))
        for mode in range(3):
            axis = axes[row_index, mode]
            for index, epsilon in enumerate(epsilons):
                values = np.asarray(spectrum[index, :, mode], dtype=float)
                peak_index = int(np.nanargmax(values))
                centered = frequency - frequency[peak_index]
                normalized = values / values[peak_index]
                selected = np.abs(centered) <= 2.0
                axis.semilogy(
                    centered[selected],
                    normalized[selected],
                    color=colors(index / max(len(epsilons) - 1, 1)),
                    linewidth=1.1,
                    label=f"{epsilon:.2g}",
                )
            axis.set_ylim(1.0e-7, 2.0)
            axis.set_title(f"{entry['family']} / mode {mode}")
            axis.grid(True, which="both", alpha=0.2)
            if mode == 0:
                axis.set_ylabel("normalized PSD")
            if row_index == 1:
                axis.set_xlabel(r"$\omega-\omega_{peak}$")
            if mode == 2:
                axis.legend(title=r"$|\epsilon|$", fontsize=7, title_fontsize=8)
    figure.suptitle("Kerr-full Q0: carrier-centered all-mode PSD")
    figure.tight_layout()
    figure.savefig(output / "q0_psd.png", dpi=180)
    plt.close(figure)


def _plot_cam_sde(
    allan_rows: list[dict[str, Any]], cam_rows: list[dict[str, Any]], output: Path
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), sharex=False, sharey=False)
    for row_index, entry in enumerate(JOBS):
        side = 1 if entry["side"] == "positive" else -1
        for mode in range(3):
            axis = axes[row_index, mode]
            observed = [
                row
                for row in allan_rows
                if row["root_family"] == entry["family"]
                and row["mode"] == mode
                and row["accepted"]
            ]
            theory = [
                row
                for row in cam_rows
                if row["root_family"] == entry["family"]
                and row["mode"] == mode
                and row["epsilon_side"] == side
            ]
            if theory:
                axis.loglog(
                    [row["epsilon"] for row in theory],
                    [row["full_ou_intensity"] for row in theory],
                    "--",
                    color="#555555",
                    linewidth=1.2,
                    label="full linear OU",
                )
                controlled = [
                    row
                    for row in theory
                    if np.isfinite(row["controlled_nonlinear_intensity"])
                ]
                if controlled:
                    axis.loglog(
                        [row["epsilon"] for row in controlled],
                        [row["controlled_nonlinear_intensity"] for row in controlled],
                        "s-",
                        color="#c44e52",
                        markersize=4,
                        linewidth=1.1,
                        label="highest controlled CAM",
                    )
            if observed:
                axis.errorbar(
                    [row["epsilon"] for row in observed],
                    [row["allan_intensity"] for row in observed],
                    yerr=[row["allan_intensity_sem"] for row in observed],
                    fmt="o",
                    color="#007f5f",
                    markersize=5,
                    capsize=2,
                    label="Q0 SDE white-FM",
                )
            axis.set_title(f"{entry['family']} / mode {mode}")
            axis.grid(True, which="both", alpha=0.2)
            if mode == 0:
                axis.set_ylabel(r"$N_A$")
            if row_index == 1:
                axis.set_xlabel(r"$|\epsilon|$")
            if mode == 2:
                axis.legend(fontsize=7)
    figure.suptitle("Kerr-full Q0: frozen CAM and SDE Allan comparison")
    figure.tight_layout()
    figure.savefig(output / "q0_cam_sde_allan.png", dpi=180)
    plt.close(figure)


def _plot_branch_survival(rows: list[dict[str, Any]], output: Path) -> None:
    points = [row for row in rows if row["mode"] == 0]
    labels = [f"{row['root_family']}\n{row['point']}" for row in points]
    ratios = [row["sde_over_cam_trace"] for row in points]
    colors = [
        "#007f5f" if ratio >= 0.8 else "#e6a700" if ratio >= 0.4 else "#c44e52"
        for ratio in ratios
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(labels, ratios, color=colors, width=0.62)
    axis.axhline(0.5, color="#555555", linestyle="--", linewidth=1.2)
    axis.set_yscale("log")
    axis.set_ylim(min(ratios) / 2.0, 2.0)
    axis.set_ylabel("observed SDE trace / target CAM trace")
    axis.set_title("Q0 branch-initialized survival after warm-up")
    axis.grid(True, axis="y", which="both", alpha=0.22)
    for index, row in enumerate(points):
        axis.text(
            index,
            ratios[index] * 1.25,
            f"downcross {row['n_hit']}/8",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(output / "q0_branch_survival.png", dpi=180)
    plt.close(figure)


def _plot_fallback_allan(
    bundle: Any, cam_rows: list[dict[str, Any]], output: Path
) -> None:
    product = bundle.products["allan_variance"]
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 3.9), sharex=True)
    for mode, axis in enumerate(axes):
        tau = _materialize(product, f"mode_results.{mode}.allan.tau")
        variance = _materialize(
            product,
            f"mode_results.{mode}.allan.nonoverlap_angular_frequency_variance",
        )
        axis.loglog(tau, tau * variance, color="#007f5f", linewidth=1.7)
        theory = next(
            row
            for row in cam_rows
            if row["root_family"] == "kf3f_032"
            and row["mode"] == mode
            and row["epsilon_side"] == 1
            and math.isclose(row["epsilon"], 0.1)
        )
        axis.axhline(
            theory["full_ou_intensity"],
            color="#555555",
            linestyle="--",
            linewidth=1.2,
            label="full linear OU",
        )
        axis.axhline(
            theory["controlled_nonlinear_intensity"],
            color="#c44e52",
            linestyle="-.",
            linewidth=1.2,
            label=f"controlled CAM O{theory['controlled_order']}",
        )
        axis.set_title(f"mode {mode}")
        axis.set_xlabel(r"$\tau$")
        axis.grid(True, which="both", alpha=0.22)
        if mode == 0:
            axis.set_ylabel(r"$\tau\,\sigma_\omega^2(\tau)$")
        if mode == 2:
            axis.legend(fontsize=8)
    figure.suptitle(r"kf3f_032 fallback at $|\epsilon|=0.1$")
    figure.tight_layout()
    figure.savefig(output / "q0_fallback_allan.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument(
        "--branch-session", type=Path, default=DEFAULT_BRANCH_SESSION
    )
    parser.add_argument(
        "--fallback-session", type=Path, default=DEFAULT_FALLBACK_SESSION
    )
    parser.add_argument("--cam-orders", type=Path, default=DEFAULT_CAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    bundles = {
        entry["family"]: load_bundle(args.session.resolve() / entry["job"])
        for entry in JOBS
    }
    branch_bundles = {
        entry["job"]: load_bundle(args.branch_session.resolve() / entry["job"])
        for entry in BRANCH_JOBS
    }
    branch_bundles.update(
        {
            entry["job"]: load_bundle(
                args.fallback_session.resolve() / entry["job"]
            )
            for entry in FALLBACK_JOBS
        }
    )
    allan_rows = [
        row
        for entry in JOBS
        for mode in range(3)
        for row in _allan_rows(bundles[entry["family"]], entry, mode)
    ]
    diagnostic_rows = [
        row
        for entry in JOBS
        for row in _diagnostic_rows(bundles[entry["family"]], entry)
    ]
    cam_rows = _cam_rows(args.cam_orders.resolve())
    branch_rows = [
        row
        for entry in (*BRANCH_JOBS, *FALLBACK_JOBS)
        for row in _branch_rows(branch_bundles[entry["job"]], entry)
    ]
    closure_rows = []
    for observed in diagnostic_rows:
        side = next(
            1 if entry["side"] == "positive" else -1
            for entry in JOBS
            if entry["family"] == observed["root_family"]
        )
        matches = [
            row
            for row in cam_rows
            if row["root_family"] == observed["root_family"]
            and row["mode"] == 0
            and row["epsilon_side"] == side
            and math.isclose(
                row["epsilon"], observed["epsilon"], rel_tol=1.0e-10, abs_tol=1.0e-14
            )
        ]
        if not matches:
            continue
        cam = matches[0]
        closure_rows.append(
            {
                **observed,
                "cam_state_trace": cam["state_trace"],
                "sde_over_cam_trace": observed["trace"] / cam["state_trace"],
                "cam_state_minimum_eigenvalue": cam["state_minimum_eigenvalue"],
            }
        )
    _write_csv(output / "q0_allan_points.csv", allan_rows)
    _write_csv(output / "q0_diagnostics.csv", diagnostic_rows)
    _write_csv(output / "q0_cam_points.csv", cam_rows)
    _write_csv(output / "q0_closure_points.csv", closure_rows)
    _write_csv(output / "q0_branch_followup.csv", branch_rows)
    _plot_allan(bundles, output)
    _plot_psd(bundles, output)
    _plot_cam_sde(allan_rows, cam_rows, output)
    _plot_branch_survival(branch_rows, output)
    _plot_fallback_allan(
        branch_bundles[FALLBACK_JOBS[0]["job"]], cam_rows, output
    )

    summary = {
        "schema": "qphase.kerr-full-frequency-q0-summary/1",
        "session": str(args.session.resolve()),
        "branch_session": str(args.branch_session.resolve()),
        "fallback_session": str(args.fallback_session.resolve()),
        "jobs": {
            entry["family"]: {
                "job": entry["job"],
                "side": entry["side"],
                "scan_points": bundles[entry["family"]].shape[0],
                "first_passage_hits": sum(
                    row["n_hit"]
                    for row in diagnostic_rows
                    if row["root_family"] == entry["family"]
                ),
                "accepted_white_fm_points_by_mode": {
                    str(mode): sum(
                        bool(row["accepted"])
                        for row in allan_rows
                        if row["root_family"] == entry["family"]
                        and row["mode"] == mode
                    )
                    for mode in range(3)
                },
                "maximum_relative_matrix_distance": max(
                    row["max_relative_matrix_distance"]
                    for row in diagnostic_rows
                    if row["root_family"] == entry["family"]
                ),
                "maximum_relative_occupation_distance": max(
                    row["max_relative_occupation_distance"]
                    for row in diagnostic_rows
                    if row["root_family"] == entry["family"]
                ),
                "matched_closure_trace_ratios": [
                    row["sde_over_cam_trace"]
                    for row in closure_rows
                    if row["root_family"] == entry["family"]
                ],
            }
            for entry in JOBS
        },
        "branch_followup": {
            entry["job"]: {
                "root_family": entry["family"],
                "point": entry["point"],
                "hit_fraction": next(
                    row["hit_fraction"]
                    for row in branch_rows
                    if row["job"] == entry["job"]
                ),
                "first_hit_time_maximum": next(
                    row["first_hit_time_maximum"]
                    for row in branch_rows
                    if row["job"] == entry["job"]
                ),
                "sde_over_cam_trace": next(
                    row["sde_over_cam_trace"]
                    for row in branch_rows
                    if row["job"] == entry["job"]
                ),
            }
            for entry in (*BRANCH_JOBS, *FALLBACK_JOBS)
        },
    }
    (output / "q0_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
