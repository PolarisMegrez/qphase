"""Generate complete pair-control campaigns for the collective-loss VDP dimer."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import qmc

SEED = 20260809
STAGE_POINTS = {"S0": 16, "S1": 32}
MODEL = "collective_vdp_2mode"


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


DIMENSIONS = {
    "omega_a": Dimension("omega_a", -5.0, 5.0, logarithmic=False),
    "Gamma": Dimension("Gamma", 1.0e-5, 0.1),
    "g": Dimension("g", 1.0e-3, 5.0),
    "pump_a": Dimension("pump_a", 0.2, 5.0),
    "kappa_dark": Dimension("kappa_dark", 1.0e-4, 2.0),
}

CONTROL_RANGES: dict[str, dict[str, Any]] = {
    "omega_a": {"min": -5.0, "max": 5.0, "scale": 1.0, "domain": "real"},
    "Gamma": {
        "min": 1.0e-5,
        "max": 0.1,
        "scale": 0.01,
        "domain": "nonnegative",
        "sampling": "log",
    },
    "g": {
        "min": 1.0e-3,
        "max": 5.0,
        "scale": 0.5,
        "domain": "nonnegative",
        "sampling": "log",
    },
    "pump_a": {
        "min": 0.2,
        "max": 5.0,
        "scale": 1.0,
        "domain": "nonnegative",
        "sampling": "log",
    },
    "kappa_dark": {
        "min": 1.0e-4,
        "max": 2.0,
        "scale": 0.02,
        "domain": "nonnegative",
        "sampling": "log",
    },
}

CONTROL_PARAMETERS = tuple(DIMENSIONS)
SCHEME_NAMES = {
    frozenset(("omega_a", "Gamma")): "detuning_nonlinear_loss",
    frozenset(("omega_a", "g")): "detuning_coupling",
    frozenset(("omega_a", "pump_a")): "detuning_pump",
    frozenset(("omega_a", "kappa_dark")): "detuning_dark_loss",
    frozenset(("Gamma", "g")): "nonlinear_loss_coupling",
    frozenset(("Gamma", "pump_a")): "nonlinear_loss_pump",
    frozenset(("Gamma", "kappa_dark")): "nonlinear_loss_dark_loss",
    frozenset(("g", "pump_a")): "coupling_pump",
    frozenset(("g", "kappa_dark")): "coupling_dark_loss",
    frozenset(("pump_a", "kappa_dark")): "pump_dark_loss",
}
SCHEMES = tuple(
    Scheme(SCHEME_NAMES[frozenset(pair)], pair)
    for pair in combinations(CONTROL_PARAMETERS, 2)
)

DEFAULTS = {
    "omega_a": 0.0,
    "omega_b": 0.0,
    "Gamma": 0.01,
    "g": 0.2,
    "pump_a": 0.5,
    "kappa_bright": 1.0,
    "kappa_dark": 0.02,
}


def _solver(stage: str, scheme: Scheme) -> dict[str, Any]:
    calibration = stage == "S0"
    return {
        "controls": {name: CONTROL_RANGES[name] for name in scheme.controls},
        "perturbation": {"parameter": "omega_b", "scale": 1.0, "side": "both"},
        "target": {"equilibrium_multiplicity": {"order": 3}},
        "strategy": {
            "auto": {
                "retained_dimension": 1,
                "max_candidates": 4,
                "condition_limit": 1.0e11,
            }
        },
        "discovery": {
            "seeds": {
                "samples_per_control": 3,
                "max_starts": 1024,
                "order_parameter_samples": 31,
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
            "max_iterations": 60 if calibration else 100,
        },
        "verification": {
            "initial_digits": 40 if calibration else 60,
            "max_digits": 120 if calibration else 180,
        },
        "audit": {"near_miss_per_reason": 8, "near_miss_total": 64},
    }


def _rows(scheme: Scheme) -> list[dict[str, Any]]:
    outer = tuple(
        dimension
        for name, dimension in DIMENSIONS.items()
        if name not in scheme.controls
    )
    seed = SEED + sum(map(ord, scheme.name))
    unit = qmc.Sobol(d=len(outer), scramble=True, seed=seed).random_base2(m=5)
    values = {
        dimension.name: dimension.transform(unit[:, index])
        for index, dimension in enumerate(outer)
    }
    rows = []
    for index in range(STAGE_POINTS["S1"]):
        row: dict[str, Any] = {
            "scheme": scheme.name,
            "sobol_index": index,
            "point_id": f"{scheme.name}_{index:03d}",
        }
        row.update({name: float(data[index]) for name, data in values.items()})
        rows.append(row)
    return rows


def _job(scheme: Scheme, stage: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    start = 0 if stage == "S0" else STAGE_POINTS["S0"]
    selected = rows[start : STAGE_POINTS[stage]]
    scanned = tuple(name for name in DIMENSIONS if name not in scheme.controls)
    plugins: dict[str, Any] = {
        "backend": {"numpy": {"float_dtype": "float64"}},
        "model": {MODEL: DEFAULTS},
        "cam_solver": {"bifurcation": _solver(stage, scheme)},
    }
    if stage != "S0":
        plugins["cam_postprocessor"] = {
            "local_response_validation": {
                "epsilon_min": 1.0e-10,
                "epsilon_max": 0.1,
                "epsilon_points": 19,
                "fit_points": 7,
                "precision_digits": 80,
                "residual_tolerance": 1.0e-30,
            },
            "stochastic_validity": {"probe_epsilon": 1.0e-4},
        }
    return {
        "name": f"{MODEL}_{scheme.name}_{stage.lower()}",
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
                    "target": f"model.{MODEL}.{name}",
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
    parser.add_argument(
        "--stages", nargs="+", choices=tuple(STAGE_POINTS), default=["S0"]
    )
    parser.add_argument(
        "--schemes", nargs="+", choices=tuple(item.name for item in SCHEMES)
    )
    parser.add_argument("--jobs-dir", type=Path, default=Path("configs/jobs"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    selected_schemes = (
        SCHEMES
        if args.schemes is None
        else tuple(item for item in SCHEMES if item.name in args.schemes)
    )
    for scheme in selected_schemes:
        rows = _rows(scheme)
        manifest = args.reports_dir / f"{MODEL}_{scheme.name}_manifest_2026-08-09.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for stage in args.stages:
            output = args.jobs_dir / f"{MODEL}_{scheme.name}_{stage.lower()}.yaml"
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8", newline="\n") as handle:
                yaml.safe_dump(_job(scheme, stage, rows), handle, sort_keys=False)


if __name__ == "__main__":
    main()
