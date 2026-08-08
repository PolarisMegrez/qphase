"""Rank collective-loss Kerr-trimer branches for stochastic validation."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

KEY = ("case", "candidate_index", "branch_index")
PARAMETERS = (
    "omega_a",
    "omega_b",
    "omega_c",
    "chi",
    "g_ab",
    "g_ac",
    "g_bc",
    "pump_a",
    "kappa_bright",
    "kappa_dark",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value or "nan")
    except ValueError:
        return math.nan


def _bool(value: str | None) -> bool:
    return str(value).lower() == "true"


def _key(row: dict[str, str]) -> tuple[int, int, int]:
    return tuple(int(row[name]) for name in KEY)  # type: ignore[return-value]


def _valid(row: dict[str, str]) -> bool:
    return all(
        _bool(row.get(name))
        for name in ("converged", "continuous", "is_physical", "is_stable")
    )


def _contiguous_branch(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    accepted = []
    for row in sorted(rows, key=lambda item: abs(_float(item.get("epsilon")))):
        if not _valid(row):
            break
        accepted.append(row)
    return accepted


def _frequency_window(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], float]:
    ordered = sorted(rows, key=lambda item: abs(_float(item.get("epsilon"))))
    exponent = next(
        (
            _float(row.get("rayleigh_fit_exponent"))
            for row in ordered
            if math.isfinite(_float(row.get("rayleigh_fit_exponent")))
        ),
        math.nan,
    )
    if not (math.isfinite(exponent) and 0.0 < exponent < 1.0):
        return [], exponent

    tolerance = max(0.08, 0.25 * abs(exponent))
    accepted = []
    for row in ordered:
        effective = _float(row.get("rayleigh_effective_exponent"))
        projection_ok = row.get("rayleigh_projection_status") in {None, "", "resolved"}
        exponent_ok = (
            not math.isfinite(effective)
            or abs(effective - exponent) <= tolerance
        )
        if not (_valid(row) and projection_ok and exponent_ok):
            break
        accepted.append(row)
    return accepted, exponent


def _scheme(stem: str) -> str:
    prefix = "collective_loss_kerr_3mode_"
    suffix = "_s1"
    value = stem.removeprefix(prefix)
    return value.removesuffix(suffix)


def summarize(run_dir: Path) -> list[dict[str, Any]]:
    stem = run_dir.name
    responses = _read(run_dir / f"{stem}_responses.csv")
    stochastic = {
        _key(row): row for row in _read(run_dir / f"{stem}_stochastic_validity.csv")
    }
    candidates = {
        (int(row["case"]), int(row["candidate"])): row
        for row in _read(run_dir / f"{stem}_candidates.csv")
    }
    grouped: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in responses:
        grouped[_key(row)].append(row)

    output: list[dict[str, Any]] = []
    for key, branch_rows in grouped.items():
        case, candidate_index, branch_index = key
        frequency_rows, exponent = _frequency_window(branch_rows)
        stable_rows = _contiguous_branch(branch_rows)
        stochastic_row = stochastic.get(key, {})
        candidate = candidates.get((case, candidate_index), {})
        epsilon_asym = (
            max(abs(_float(row.get("epsilon"))) for row in frequency_rows)
            if frequency_rows
            else math.nan
        )
        epsilon_stable = (
            max(abs(_float(row.get("epsilon"))) for row in stable_rows)
            if stable_rows
            else math.nan
        )
        epsilon_noise = _float(stochastic_row.get("epsilon_crossover"))
        w_noise = (
            math.log10(epsilon_asym / epsilon_noise)
            if epsilon_asym > 0.0 and epsilon_noise > 0.0
            else math.nan
        )
        petermann = [
            _float(row.get("hamiltonian_petermann_max"))
            for row in stable_rows
            if math.isfinite(_float(row.get("hamiltonian_petermann_max")))
        ]
        petermann_exponents = [
            _float(row.get("hamiltonian_petermann_fit_exponent"))
            for row in stable_rows
            if math.isfinite(_float(row.get("hamiltonian_petermann_fit_exponent")))
        ]
        source = branch_rows[0]
        row: dict[str, Any] = {
            "scheme": _scheme(stem),
            "run_dir": str(run_dir),
            "case": case,
            "candidate_index": candidate_index,
            "branch_index": branch_index,
            "epsilon_side": int(source["epsilon_side"]),
        }
        for name in PARAMETERS:
            value = candidate.get(name, source.get(name))
            row[name] = _float(value)
        row.update(
            {
                "rayleigh_fit_exponent": exponent,
                "frequency_point_count": len(frequency_rows),
                "stable_point_count": len(stable_rows),
                "epsilon_asym_frequency": epsilon_asym,
                "epsilon_stable_physical": epsilon_stable,
                "epsilon_noise": epsilon_noise,
                "W_noise": w_noise,
                "noncritical_spectral_gap": _float(
                    stochastic_row.get("noncritical_spectral_gap")
                ),
                "projected_noise_intensity": _float(
                    stochastic_row.get("projected_noise_intensity")
                ),
                "critical_mode_condition_number": _float(
                    stochastic_row.get("critical_mode_condition_number")
                ),
                "minimum_physical_eigenvalue": (
                    min(
                        _float(item.get("minimum_physical_eigenvalue"))
                        for item in frequency_rows
                    )
                    if frequency_rows
                    else math.nan
                ),
                "rayleigh_visibility": max(
                    (
                        _float(item.get("rayleigh_visibility"))
                        for item in frequency_rows
                    ),
                    default=math.nan,
                ),
                "hamiltonian_petermann_min": min(petermann)
                if petermann
                else math.nan,
                "hamiltonian_petermann_max": max(petermann)
                if petermann
                else math.nan,
                "hamiltonian_petermann_fit_exponent": petermann_exponents[0]
                if petermann_exponents
                else math.nan,
                "stochastic_status": stochastic_row.get("status", "missing"),
                "q0_eligible": bool(
                    len(frequency_rows) >= 4
                    and stochastic_row.get("status") == "complete"
                    and math.isfinite(exponent)
                    and 0.0 < exponent < 1.0
                ),
            }
        )
        output.append(row)
    return output


def _rank(rows: list[dict[str, Any]], field: str, output: str) -> None:
    eligible = [
        row for row in rows if row["q0_eligible"] and math.isfinite(row[field])
    ]
    eligible.sort(key=lambda row: row[field], reverse=True)
    for rank, row in enumerate(eligible, start=1):
        row[output] = rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [row for run_dir in args.run_dirs for row in summarize(run_dir)]
    for field, output in (
        ("W_noise", "rank_W_noise"),
        ("noncritical_spectral_gap", "rank_spectral_gap"),
        ("rayleigh_visibility", "rank_visibility"),
        ("minimum_physical_eigenvalue", "rank_physical_margin"),
    ):
        for row in rows:
            row[output] = ""
        _rank(rows, field, output)
    rows.sort(
        key=lambda row: (
            not row["q0_eligible"],
            -row["W_noise"] if math.isfinite(row["W_noise"]) else math.inf,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
