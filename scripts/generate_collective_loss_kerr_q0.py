"""Generate scheduler Q0 jobs from ranked collective-loss CAM candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

MODEL = "collective_loss_kerr_3mode"
PARAMETERS = (
    "omega_b",
    "omega_c",
    "chi",
    "g_ab",
    "g_ac",
    "g_bc",
    "pump_a",
    "kappa_dark",
)


def _read_one(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError("selection CSV must contain exactly one candidate")
    return rows[0]


def _candidate_row(ranking: dict[str, str]) -> dict[str, str]:
    run_dir = Path(ranking["run_dir"])
    stem = run_dir.name
    path = run_dir / f"{stem}_candidates.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    case = int(ranking["case"])
    index = int(ranking["candidate_index"])
    return next(
        row
        for row in rows
        if int(row["case"]) == case and int(row["candidate"]) == index
    )


def _coherent_amplitude(candidate: dict[str, str]) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.complex128)
    for mode in range(3):
        matrix[mode, mode] = float(candidate[f"r_diag_{mode}"])
    for left, right in ((0, 1), (0, 2), (1, 2)):
        value = complex(
            float(candidate[f"r_re_{left}_{right}"]),
            float(candidate[f"r_im_{left}_{right}"]),
        )
        matrix[left, right] = value
        matrix[right, left] = value.conjugate()
    values, vectors = np.linalg.eigh(matrix - 0.5 * np.eye(3))
    amplitude = np.sqrt(max(float(values[-1]), 0.0)) * vectors[:, -1]
    pivot = next((value for value in amplitude if abs(value) > 1e-12), 1.0 + 0.0j)
    amplitude *= np.exp(-1j * np.angle(pivot))
    return amplitude


def _complex_text(value: complex) -> str:
    return f"{value.real:.16g}{value.imag:+.16g}j"


def _job_config(row: dict[str, str]) -> dict[str, Any]:
    candidate = _candidate_row(row)
    amplitude = _coherent_amplitude(candidate)
    critical = float(row["omega_c"])
    side = int(row["epsilon_side"])
    epsilon_noise = float(row["epsilon_noise"])
    epsilon_high = float(row["epsilon_asym_frequency"])
    epsilon_low = max(epsilon_high / 100.0, epsilon_noise / 2.0, 1e-5)
    if epsilon_low >= epsilon_high:
        epsilon_low = epsilon_high / 10.0
    epsilon = np.geomspace(epsilon_low, epsilon_high, 5)
    values = [critical + side * float(value) for value in epsilon]
    label = row["label"]
    model = {"omega_a": 0.0, "kappa_bright": 1.0}
    model.update({name: float(row[name]) for name in PARAMETERS})
    norm = float(np.linalg.norm(amplitude))
    state_limit = max(100.0, 5.0 * norm)
    passage_limit = max(80.0, 3.0 * norm)
    sim_name = f"{MODEL}_{label}_q0_sim"
    jobs = [
        {
            "name": sim_name,
            "save": True,
            "scan": {
                "axes": {
                    "omega_c": {
                        "target": f"model.{MODEL}.omega_c",
                        "values": values,
                    }
                }
            },
            "engine": {
                "sde": {
                    "t0": 50000.0,
                    "t1": 200000.0,
                    "dt": 0.04,
                    "n_traj": 8,
                    "seed": int(row.get("seed", "20260820")),
                    "ic": [[_complex_text(value) for value in amplitude]],
                    "adaptive": False,
                    "save_stride": 10,
                    "record_modes": [0],
                    "keep_traj": False,
                    "trajectory_batching": "auto",
                    "max_state_norm": state_limit,
                    "state_check_interval_steps": 1024,
                }
            },
            "backend": {"cupy": {"float_dtype": "float64", "device": "cuda"}},
            "integrator": {
                "cayley_maruyama": {
                    "fused": "required",
                    "chunk_steps": 256,
                    "max_modes": 16,
                }
            },
            "model": {MODEL: model},
            "observer": {
                "first_passage": {
                    "rule": "state_norm",
                    "threshold": passage_limit,
                    "direction": "above",
                    "action": "record",
                    "check_interval_steps": 1024,
                    "debounce_checks": 2,
                }
            },
            "analyser": {
                "psd": {
                    "modes": [0],
                    "kind": "complex",
                    "convention": "symmetric",
                    "expected_freq_max": 6.0,
                    "find_peaks": True,
                    "estimator": {"periodogram": {"fft_chunk_trajectories": 2}},
                },
                "allan_variance": {
                    "modes": [0],
                    "points": 56,
                    "min_windows": 8,
                    "min_independent_windows": 4,
                },
            },
        },
        {
            "name": f"{MODEL}_{label}_q0_fit",
            "input": {"from": sim_name, "mode": "dataset"},
            "save": True,
            "engine": {"sde": {"mode": "analyze"}},
            "analyser": {
                "lorentz_fitter": {
                    "scan_param": "omega_c",
                    "mode": 0,
                    "freq_min": -6.0,
                    "freq_max": 6.0,
                    "fit_window": 0.2,
                    "uncertainty": "auto",
                    "clip_by_std": False,
                    "export": ["fit_results.csv", "psd_merged.csv"],
                }
            },
        },
        {
            "name": f"{MODEL}_{label}_q0_allan",
            "input": {"from": sim_name, "mode": "dataset"},
            "save": True,
            "engine": {"sde": {"mode": "analyze"}},
            "analyser": {
                "allan_scaling": {
                    "scan_param": "omega_c",
                    "critical_value": critical,
                    "side": "positive" if side > 0 else "negative",
                    "mode": 0,
                    "window_selection": "latest",
                    "white_slope_min": -1.2,
                    "white_slope_max": -0.8,
                    "min_local_r2": 0.9,
                    "min_tau_decades": 0.25,
                    "min_averaging_samples": 16,
                    "max_relative_sem": 0.5,
                    "min_independent_windows": 8,
                    "min_scaling_points": 5,
                    "target_scaling_decades": 1.0,
                    "max_scaling_reduced_chi2": 30.0,
                    "bootstrap_samples": 1000,
                    "bootstrap_seed": int(row.get("seed", "20260820")),
                    "min_frequency_rss_improvement": 0.05,
                    "normal_form": {"n": 3, "k": 1, "m": 0, "observable_order": 1},
                }
            },
        },
    ]
    return {"jobs": jobs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = _job_config(_read_one(args.selection))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


if __name__ == "__main__":
    main()
