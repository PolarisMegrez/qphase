"""Generate manifold-preserving local refinements from ranked CAM candidates."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import yaml
from generate_collective_loss_kerr_interference_campaign import (
    CONTROL_PARAMETERS,
    MODEL,
    SCHEMES,
    SECTORS,
    _control_range,
)

FIXED_PARAMETERS = (*CONTROL_PARAMETERS,)
DEFAULTS = {"omega_a": 0.0, "kappa_bright": 1.0}


def _number(row: dict[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        return DEFAULTS[name]
    return value


def _oat_values(name: str, center: float) -> list[float]:
    if name == "omega_b":
        span = max(0.25, 0.25 * abs(center))
        return [center - span, center - 0.5 * span, center + 0.5 * span, center + span]
    return [center * factor for factor in (0.5, 0.75, 1.25, 1.5)]


def _local_control(name: str, center: float, sector: str) -> dict[str, Any]:
    config = _control_range(name, SECTORS[sector])
    if name == "omega_b":
        span = max(0.5, abs(center))
        config.update(
            {
                "min": center - span,
                "max": center + span,
                "scale": max(1e-3, abs(center)),
            }
        )
        return config

    sign = -1.0 if center < 0.0 else 1.0
    low, high = sorted((abs(center) / 3.0, abs(center) * 3.0))
    if sign < 0.0:
        config.update({"min": -high, "max": -low, "scale": abs(center)})
    else:
        config.update({"min": low, "max": high, "scale": abs(center)})
    return config


def _job(row: dict[str, str]) -> dict[str, Any]:
    label = row["label"]
    sector = row["sector"]
    scheme_name = row["scheme"].removeprefix(f"interference_{sector}_")
    scheme = next(item for item in SCHEMES if item.name == scheme_name)
    model = {
        "omega_a": 0.0,
        "omega_b": _number(row, "omega_b"),
        "omega_c": _number(row, "omega_c"),
        "chi": _number(row, "chi"),
        "g_ab": _number(row, "g_ab"),
        "g_ac": _number(row, "g_ac"),
        "g_bc": _number(row, "g_bc"),
        "pump_a": _number(row, "pump_a"),
        "kappa_bright": 1.0,
        "kappa_dark": _number(row, "kappa_dark"),
    }
    scanned = [
        name
        for name in FIXED_PARAMETERS
        if name not in scheme.controls and name != "omega_c"
    ]
    axes: dict[str, dict[str, Any]] = {}
    point_count = 1 + 4 * len(scanned)
    for name in scanned:
        values = [model[name]] * point_count
        offset = 1 + 4 * scanned.index(name)
        values[offset : offset + 4] = _oat_values(name, model[name])
        axes[name] = {"target": f"model.{MODEL}.{name}", "values": values}

    return {
        "name": f"{MODEL}_{label}_local_refinement",
        "save": True,
        "system": {
            "scan_runtime": {"checkpoint": {"enabled": True, "interval_chunks": 1}}
        },
        "scan": {"combine": "zipped", "axes": axes},
        "engine": {"cam": {"case_failure_policy": "record"}},
        "backend": {"numpy": {"float_dtype": "float64"}},
        "model": {MODEL: model},
        "cam_solver": {
            "bifurcation": {
                    "controls": {
                        name: _local_control(name, model[name], sector)
                        for name in scheme.controls
                    },
                    "perturbation": {
                        "parameter": "omega_c",
                        "scale": 1.0,
                        "side": "both",
                    },
                    "target": {"equilibrium_multiplicity": {"order": 3}},
                    "strategy": {
                        "auto": {
                            "retained_dimension": 1,
                            "max_candidates": 9,
                            "condition_limit": 1e11,
                        }
                    },
                    "discovery": {
                        "seeds": {
                            "samples_per_control": 4,
                            "max_starts": 2048,
                            "order_parameter_samples": 41,
                        }
                    },
                    "classifier": {
                        "scaling_signature": {
                            "max_total_order": 8,
                            "coefficient_tolerance": 1e-14,
                            "max_exponent": 0.999,
                        }
                    },
                    "refinement": {"tolerance": 1e-11, "max_iterations": 120},
                    "verification": {"initial_digits": 60, "max_digits": 240},
                    "audit": {"near_miss_per_reason": 8, "near_miss_total": 64},
            }
        },
        "cam_postprocessor": {
            "local_response_validation": {
                "epsilon_min": 1e-10,
                "epsilon_max": 0.1,
                "epsilon_points": 19,
                "fit_points": 7,
                "precision_digits": 80,
                "residual_tolerance": 1e-30,
            },
            "stochastic_validity": {"probe_epsilon": 1e-4},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.selection.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("selection CSV contains no candidates")
    jobs = [_job(row) for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump({"jobs": jobs}, handle, sort_keys=False)


if __name__ == "__main__":
    main()
