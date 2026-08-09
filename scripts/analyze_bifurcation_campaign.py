"""Summarize candidates and near misses from one model's CAM campaign."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _scheme(path: Path, model: str) -> str:
    value = path.stem.removeprefix(f"{model}_")
    for suffix in ("_s0", "_s1", "_s2"):
        value = value.removesuffix(suffix)
    return value


def _stage(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1].upper()


def _rows(path: Path, model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with np.load(path, allow_pickle=True) as archive:
        metadata = tuple(archive["case_metadata"])
        candidate_count = int(archive["candidate_states"].shape[0])
    scheme = _scheme(path, model)
    stage = _stage(path)
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for case_index, case in enumerate(metadata):
        fixed = dict(case.get("fixed_params", {}))
        audit = dict(case.get("audit", {}))
        reasons.update(audit.get("totals", {}).get("rejected_by_reason", {}))
        for near_index, near in enumerate(audit.get("near_misses", ())):
            rejection_reasons = tuple(near.get("rejection_reasons", ()))
            params = {**fixed, **dict(near.get("controls", {}))}
            minimum = float(near.get("min_state_eigenvalue", math.nan))
            rows.append(
                {
                    "scheme": scheme,
                    "stage": stage,
                    "source": str(path),
                    "case": case_index,
                    "near_miss": near_index,
                    **params,
                    "path": near.get("path", "unknown"),
                    "reduction": near.get("reduction"),
                    "search_residual": float(near.get("search_residual", math.nan)),
                    "full_residual": float(near.get("full_residual", math.nan)),
                    "minimum_physical_eigenvalue": minimum,
                    "is_physical_near_miss": bool(
                        minimum >= -1.0e-8 and "non_physical" not in rejection_reasons
                    ),
                    "rejection_reasons": ";".join(rejection_reasons),
                }
            )
    physical = [row for row in rows if row["is_physical_near_miss"]]
    summary = {
        "scheme": scheme,
        "stage": stage,
        "source": str(path),
        "case_count": len(metadata),
        "candidate_count": candidate_count,
        "near_miss_count": len(rows),
        "physical_near_miss_count": len(physical),
        "minimum_physical_full_residual": min(
            (row["full_residual"] for row in physical), default=math.nan
        ),
        "minimum_physical_search_residual": min(
            (row["search_residual"] for row in physical), default=math.nan
        ),
        "top_rejection_reasons": ";".join(
            f"{name}:{count}" for name, count in reasons.most_common(5)
        ),
    }
    return rows, summary


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({name for row in rows for name in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--near-misses", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    summaries = []
    for path in args.inputs:
        rows, summary = _rows(path, args.model)
        all_rows.extend(rows)
        summaries.append(summary)
    all_rows.sort(
        key=lambda row: (
            not row["is_physical_near_miss"],
            row["full_residual"],
            row["search_residual"],
        )
    )
    summaries.sort(
        key=lambda row: (
            -row["candidate_count"],
            row["minimum_physical_full_residual"],
        )
    )
    _write(args.near_misses, all_rows)
    _write(args.summary, summaries)


if __name__ == "__main__":
    main()
