"""Generate prioritized control-pair campaigns for the reservoir Kerr trimer."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import qmc

SEED = 20260808
PREFIXES = {"S0": 4, "S1": 32}
CHI_STRATA = (1.0e-4, 1.0e-3, 1.0e-2)


@dataclass(frozen=True)
class Dimension:
    name: str
    lower: float
    upper: float
    logarithmic: bool = True

    def transform(self, unit: np.ndarray) -> np.ndarray:
        if self.logarithmic:
            low, high = np.log10((self.lower, self.upper))
            return np.power(10.0, low + unit * (high - low))
        return self.lower + unit * (self.upper - self.lower)


@dataclass(frozen=True)
class Scheme:
    name: str
    controls: tuple[str, str]
    outer: tuple[Dimension, ...]
    stratify_chi: bool = False


DELTA = Dimension("delta", 0.05, 5.0)
OMEGA_R = Dimension("omega_r", -5.0, 5.0, logarithmic=False)
G = Dimension("g", 0.01, 5.0)
G_R = Dimension("g_r", 0.01, 5.0)
PUMP = Dimension("pump_r", 0.01, 0.995, logarithmic=False)
LOCAL = Dimension("kappa_local", 1.0e-4, 2.0)

SCHEMES = (
    Scheme(
        "pump_reservoir_coupling",
        ("pump_r", "g_r"),
        (DELTA, OMEGA_R, G, LOCAL),
        True,
    ),
    Scheme(
        "pump_direct_coupling",
        ("pump_r", "g"),
        (DELTA, OMEGA_R, G_R, LOCAL),
        True,
    ),
    Scheme(
        "pump_nonlinearity",
        ("pump_r", "chi"),
        (DELTA, OMEGA_R, G, G_R, LOCAL),
    ),
    Scheme(
        "reservoir_coupling_local_loss",
        ("g_r", "kappa_local"),
        (DELTA, OMEGA_R, G, PUMP),
        True,
    ),
    Scheme(
        "nonlinear_local_loss",
        ("chi", "kappa_local"),
        (DELTA, OMEGA_R, G, G_R, PUMP),
    ),
    Scheme(
        "detuning_nonlinearity",
        ("omega_r", "chi"),
        (DELTA, G, G_R, PUMP, LOCAL),
    ),
    Scheme(
        "detuning_direct_coupling",
        ("omega_r", "g"),
        (DELTA, G_R, PUMP, LOCAL),
        True,
    ),
    Scheme(
        "detuning_reservoir_coupling",
        ("omega_r", "g_r"),
        (DELTA, G, PUMP, LOCAL),
        True,
    ),
    Scheme(
        "detuning_local_loss",
        ("omega_r", "kappa_local"),
        (DELTA, G, G_R, PUMP),
        True,
    ),
    Scheme(
        "nonlinear_direct_coupling",
        ("chi", "g"),
        (DELTA, OMEGA_R, G_R, PUMP, LOCAL),
    ),
    Scheme(
        "nonlinear_reservoir_coupling",
        ("chi", "g_r"),
        (DELTA, OMEGA_R, G, PUMP, LOCAL),
    ),
    Scheme(
        "direct_reservoir_coupling",
        ("g", "g_r"),
        (DELTA, OMEGA_R, PUMP, LOCAL),
        True,
    ),
    Scheme(
        "direct_coupling_local_loss",
        ("g", "kappa_local"),
        (DELTA, OMEGA_R, G_R, PUMP),
        True,
    ),
)

RANGES: dict[str, dict[str, Any]] = {
    "omega_r": {"min": -5.0, "max": 5.0, "scale": 1.0, "domain": "real"},
    "chi": {
        "min": 1.0e-5,
        "max": 0.1,
        "scale": 1.0e-2,
        "domain": "nonnegative",
        "sampling": "log",
    },
    "g": {
        "min": 5.0e-3,
        "max": 5.0,
        "scale": 0.2,
        "domain": "nonnegative",
        "sampling": "log",
    },
    "g_r": {
        "min": 5.0e-3,
        "max": 5.0,
        "scale": 0.2,
        "domain": "nonnegative",
        "sampling": "log",
    },
    "pump_r": {
        "min": 1.0e-3,
        "max": 0.999,
        "scale": 1.0,
        "domain": "nonnegative",
    },
    "kappa_local": {
        "min": 1.0e-4,
        "max": 2.0,
        "scale": 0.1,
        "domain": "nonnegative",
        "sampling": "log",
    },
}

DEFAULTS = {
    "omega_r": 0.0,
    "omega_0": 0.0,
    "delta": 1.0,
    "chi": 1.0e-3,
    "g": 0.2,
    "g_r": 0.2,
    "kappa_r": 1.0,
    "pump_r": 0.5,
    "kappa_local": 0.1,
}


def _solver(stage: str, scheme: Scheme) -> dict[str, Any]:
    calibration = stage == "S0"
    return {
        "controls": {name: RANGES[name] for name in scheme.controls},
        "perturbation": {"parameter": "delta", "scale": 1.0, "side": "both"},
        "target": {"equilibrium_multiplicity": {"order": 3}},
        "strategy": {
            "auto": {
                "retained_dimension": 1,
                "max_candidates": 4 if calibration else 9,
                "condition_limit": 1.0e11,
            }
        },
        "discovery": {
            "seeds": {
                "samples_per_control": 3 if calibration else 4,
                "max_starts": 1024 if calibration else 2048,
                "order_parameter_samples": 31 if calibration else 41,
            }
        },
        "classifier": {
            "scaling_signature": {
                "max_total_order": 8,
                "coefficient_tolerance": 1.0e-14,
                "max_exponent": 0.999,
            }
        },
        "refinement": {
            "tolerance": 1.0e-9 if calibration else 1.0e-11,
            "max_iterations": 60 if calibration else 120,
        },
        "verification": {
            "initial_digits": 40 if calibration else 60,
            "max_digits": 120 if calibration else 240,
        },
        "audit": {"near_miss_per_reason": 8, "near_miss_total": 64},
    }


def _rows(scheme: Scheme) -> list[dict[str, Any]]:
    seed = SEED + sum(map(ord, scheme.name))
    unit = qmc.Sobol(d=len(scheme.outer), scramble=True, seed=seed).random_base2(m=5)
    values = {
        dimension.name: dimension.transform(unit[:, index])
        for index, dimension in enumerate(scheme.outer)
    }
    rows: list[dict[str, Any]] = []
    for chi in CHI_STRATA if scheme.stratify_chi else (None,):
        for index in range(PREFIXES["S1"]):
            row: dict[str, Any] = {
                "scheme": scheme.name,
                "stratum": (
                    f"{scheme.name}_chi_{chi:.0e}"
                    if chi is not None
                    else scheme.name
                ),
                "sobol_index": index,
            }
            row.update({name: float(data[index]) for name, data in values.items()})
            if chi is not None:
                row["chi"] = chi
            rows.append(row)
    return rows


def _job(scheme: Scheme, stage: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["sobol_index"] < PREFIXES[stage]]
    scanned = tuple(dimension.name for dimension in scheme.outer)
    if scheme.stratify_chi:
        scanned = ("chi", *scanned)
    plugins: dict[str, Any] = {
        "backend": {"numpy": {"float_dtype": "float64"}},
        "model": {"reservoir_kerr_3mode": DEFAULTS},
        "cam_solver": {"bifurcation": _solver(stage, scheme)},
    }
    if stage != "S0":
        plugins["cam_postprocessor"] = {
            "local_response_validation": {
                "epsilon_min": 1.0e-10,
                "epsilon_max": 1.0e-2,
                "epsilon_points": 13,
                "fit_points": 5,
                "precision_digits": 80,
                "residual_tolerance": 1.0e-30,
            },
            "stochastic_validity": {"probe_epsilon": 1.0e-4},
        }
    return {
        "name": f"reservoir_kerr_3mode_{scheme.name}_{stage.lower()}",
        "save": True,
        "system": {
            "scan_runtime": {
                "checkpoint": {
                    "enabled": True,
                    "interval_chunks": 1,
                    "keep_on_success": False,
                }
            }
        },
        "scan": {
            "combine": "zipped",
            "axes": {
                name: {
                    "target": f"model.reservoir_kerr_3mode.{name}",
                    "values": [float(row[name]) for row in selected],
                }
                for name in scanned
            },
        },
        "engine": {"cam": {"case_failure_policy": "record"}},
        "plugins": plugins,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", choices=tuple(PREFIXES), default=["S0"])
    parser.add_argument("--jobs-dir", type=Path, default=Path("configs/jobs"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    for scheme in SCHEMES:
        rows = _rows(scheme)
        manifest = args.reports_dir / (
            f"reservoir_kerr_3mode_{scheme.name}_manifest_2026-08-08.csv"
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for stage in args.stages:
            output = args.jobs_dir / (
                f"reservoir_kerr_3mode_{scheme.name}_{stage.lower()}.yaml"
            )
            with output.open("w", encoding="utf-8", newline="\n") as handle:
                yaml.safe_dump(_job(scheme, stage, rows), handle, sort_keys=False)


if __name__ == "__main__":
    main()
