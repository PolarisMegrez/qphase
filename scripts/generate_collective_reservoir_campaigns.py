"""Generate deterministic layered campaigns for collective Kerr models."""

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
CHI_VALUES = (1.0e-4, 1.0e-3, 1.0e-2)
DELTA_VALUES = (-1.0, 1.0)
PREFIXES = {"S0": 4, "S1": 32, "S2": 128}


@dataclass(frozen=True)
class Dimension:
    name: str
    lower: float
    upper: float
    logarithmic: bool = True

    def transform(self, coordinate: np.ndarray) -> np.ndarray:
        if self.logarithmic:
            low, high = np.log10((self.lower, self.upper))
            return np.power(10.0, low + coordinate * (high - low))
        return self.lower + coordinate * (self.upper - self.lower)


@dataclass(frozen=True)
class Scheme:
    name: str
    outer: tuple[Dimension, ...]
    controls: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Campaign:
    model: str
    defaults: dict[str, float]
    schemes: tuple[Scheme, ...]


CAMPAIGNS = {
    "collective": Campaign(
        model="collective_kerr_2mode",
        defaults={
            "omega_0": 0.0,
            "delta": 0.0,
            "chi": 1.0e-3,
            "g": 0.2,
            "kappa_bright": 1.0,
            "pump_bright": 0.5,
            "kappa_dark": 0.1,
        },
        schemes=(
            Scheme(
                name="loss_controls",
                outer=(Dimension("g", 0.01, 5.0),),
                controls={
                    "pump_bright": {
                        "min": 1.0e-3,
                        "max": 0.999,
                        "scale": 1.0,
                        "domain": "nonnegative",
                    },
                    "kappa_dark": {
                        "min": 1.0e-4,
                        "max": 2.0,
                        "scale": 0.1,
                        "domain": "nonnegative",
                        "sampling": "log",
                    },
                },
            ),
            Scheme(
                name="coupling_control",
                outer=(Dimension("kappa_dark", 1.0e-4, 2.0),),
                controls={
                    "g": {
                        "min": 5.0e-3,
                        "max": 5.0,
                        "scale": 0.2,
                        "domain": "nonnegative",
                        "sampling": "log",
                    },
                    "pump_bright": {
                        "min": 1.0e-3,
                        "max": 0.999,
                        "scale": 1.0,
                        "domain": "nonnegative",
                    },
                },
            ),
        ),
    ),
    "reservoir": Campaign(
        model="reservoir_kerr_3mode",
        defaults={
            "omega_r": 0.0,
            "omega_0": 0.0,
            "delta": 0.0,
            "chi": 1.0e-3,
            "g": 0.2,
            "g_r": 0.2,
            "kappa_r": 1.0,
            "pump_r": 0.5,
            "kappa_local": 0.1,
        },
        schemes=(
            Scheme(
                name="loss_controls",
                outer=(
                    Dimension("omega_r", -5.0, 5.0, logarithmic=False),
                    Dimension("g", 0.01, 5.0),
                    Dimension("g_r", 0.01, 5.0),
                ),
                controls={
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
                },
            ),
            Scheme(
                name="detuning_control",
                outer=(
                    Dimension("g", 0.01, 5.0),
                    Dimension("g_r", 0.01, 5.0),
                    Dimension("kappa_local", 1.0e-4, 2.0),
                ),
                controls={
                    "omega_r": {
                        "min": -5.0,
                        "max": 5.0,
                        "scale": 1.0,
                        "domain": "real",
                    },
                    "pump_r": {
                        "min": 1.0e-3,
                        "max": 0.999,
                        "scale": 1.0,
                        "domain": "nonnegative",
                    },
                },
            ),
        ),
    ),
}


def _solver(
    stage: str,
    controls: dict[str, dict[str, Any]],
    *,
    allow_full_fallback: bool,
) -> dict[str, Any]:
    calibration = stage == "S0"
    return {
        "controls": controls,
        "perturbation": {"parameter": "delta", "scale": 1.0, "side": "both"},
        "target": {"equilibrium_multiplicity": {"order": 3}},
        "strategy": {
            "auto" if allow_full_fallback else "reduced": {
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


def _design(campaign: Campaign, scheme: Scheme) -> dict[str, np.ndarray]:
    unit = qmc.Sobol(
        d=len(scheme.outer),
        scramble=True,
        seed=SEED + sum(map(ord, campaign.model + scheme.name)),
    ).random_base2(m=7)
    return {
        dimension.name: dimension.transform(unit[:, index])
        for index, dimension in enumerate(scheme.outer)
    }


def _rows(campaign: Campaign, scheme: Scheme) -> list[dict[str, Any]]:
    design = _design(campaign, scheme)
    rows: list[dict[str, Any]] = []
    for delta in DELTA_VALUES:
        delta_label = "negative" if delta < 0.0 else "positive"
        for chi in CHI_VALUES:
            for index in range(PREFIXES["S2"]):
                row: dict[str, Any] = {
                    "campaign": campaign.model,
                    "scheme": scheme.name,
                    "stratum": f"{scheme.name}_{delta_label}_chi_{chi:.0e}",
                    "sobol_index": index,
                    "point_id": (
                        f"{scheme.name}_{delta_label}_chi_{chi:.0e}_{index:03d}"
                    ),
                    "delta": delta,
                    "chi": chi,
                }
                row.update(
                    {name: float(values[index]) for name, values in design.items()}
                )
                rows.append(row)
    return rows


def _job(
    campaign: Campaign,
    scheme: Scheme,
    stage: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [row for row in rows if int(row["sobol_index"]) < PREFIXES[stage]]
    scanned = ("delta", "chi", *(dimension.name for dimension in scheme.outer))
    axes = {
        name: {
            "target": f"model.{campaign.model}.{name}",
            "values": [float(row[name]) for row in selected],
        }
        for name in scanned
    }
    plugins: dict[str, Any] = {
        "backend": {"numpy": {"float_dtype": "float64"}},
        "model": {campaign.model: campaign.defaults},
        "cam_solver": {
            "bifurcation": _solver(
                stage,
                scheme.controls,
                allow_full_fallback=campaign.model == "reservoir_kerr_3mode",
            )
        },
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
        "name": f"{campaign.model}_{scheme.name}_{stage.lower()}",
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
        "scan": {"combine": "zipped", "axes": axes},
        "engine": {"cam": {"case_failure_policy": "record"}},
        "plugins": plugins,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", choices=tuple(CAMPAIGNS))
    parser.add_argument("--stages", nargs="+", choices=tuple(PREFIXES), default=["S0"])
    parser.add_argument("--jobs-dir", type=Path, default=Path("configs/jobs"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    campaign = CAMPAIGNS[args.campaign]
    for scheme in campaign.schemes:
        rows = _rows(campaign, scheme)
        manifest = (
            args.reports_dir
            / f"{campaign.model}_{scheme.name}_manifest_2026-08-08.csv"
        )
        _write_csv(manifest, rows)
        for stage in args.stages:
            output = (
                args.jobs_dir / f"{campaign.model}_{scheme.name}_{stage.lower()}.yaml"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8", newline="\n") as handle:
                yaml.safe_dump(
                    _job(campaign, scheme, stage, rows),
                    handle,
                    sort_keys=False,
                )


if __name__ == "__main__":
    main()
