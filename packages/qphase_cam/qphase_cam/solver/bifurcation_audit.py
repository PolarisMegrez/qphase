"""Discovery/rejection audit support for bifurcation searches.

The audit makes negative bifurcation results (zero candidates) explainable
without reading debug logs: which structures were searched, how many starts
were generated, how many the physical domain filtered out, the dominant
rejection reasons, and which regions stayed uncovered.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#: Standard wording attached to empty bifurcation results: the search found
#: no candidate *within the stated numerical coverage*.  It never asserts
#: that no candidate exists.
EMPTY_RESULT_NOTE = "no_candidates_found_within_stated_numerical_coverage"

#: Multi-label semantics of every ``rejected_by_reason`` counter: each
#: rejected candidate increments *all* of its rejection-reason labels, so
#: the per-label counts sum to at least the number of rejected candidates.
REJECTION_COUNTER_SEMANTICS = (
    "multi-label: each rejected candidate increments every applicable "
    "rejection reason, so reason counts sum to >= rejected_count"
)

#: Audit payload schema marker stored under ``metadata["audit"]["schema"]``.
AUDIT_SCHEMA = "cam_bifurcation_audit/1"


class AuditConfig(BaseModel):
    """Bounded near-miss retention limits for the bifurcation audit."""

    model_config = ConfigDict(extra="forbid")

    near_miss_per_reason: int = Field(8, ge=0)
    near_miss_total: int = Field(64, ge=0)


class NearMissStore:
    """Bounded retention of rejected candidates, near-misses first.

    Records are bucketed by rejection-reason label; a multi-label record
    occupies one slot in *each* of its label buckets.  Selection considers
    records in ascending ``full_residual`` order (near-miss first) and keeps
    a record while every one of its label buckets is below
    ``per_reason`` and the global ``total`` cap is not reached.  Only
    quantities already computed for the rejection decision are stored; the
    store never triggers new linear algebra or verification work.
    """

    def __init__(self, *, per_reason: int, total: int) -> None:
        self.per_reason = per_reason
        self.total = total
        self._records: list[dict[str, Any]] = []
        self.dropped = 0

    def add(
        self,
        *,
        path: str,
        reduction: str | None,
        seed_source: str,
        controls: dict[str, float],
        rejection_reasons: tuple[str, ...],
        search_residual: float,
        full_residual: float,
        min_state_eigenvalue: float,
    ) -> None:
        self._records.append(
            {
                "path": path,
                "reduction": reduction,
                "seed_source": seed_source,
                "controls": {name: float(value) for name, value in controls.items()},
                "rejection_reasons": tuple(rejection_reasons) or ("unknown",),
                "search_residual": float(search_residual),
                "full_residual": float(full_residual),
                "min_state_eigenvalue": float(min_state_eigenvalue),
            }
        )

    @staticmethod
    def _sort_key(record: dict[str, Any]) -> tuple[float, float]:
        def finite(value: float) -> float:
            return value if math.isfinite(value) else math.inf

        return (finite(record["full_residual"]), finite(record["search_residual"]))

    def finalize(self) -> list[dict[str, Any]]:
        """Return the bounded near-miss selection and record the drop count."""
        kept: list[dict[str, Any]] = []
        if self.per_reason > 0 and self.total > 0:
            buckets: Counter[str] = Counter()
            for record in sorted(self._records, key=self._sort_key):
                if len(kept) >= self.total:
                    break
                reasons = record["rejection_reasons"]
                if all(buckets[reason] >= self.per_reason for reason in reasons):
                    continue
                kept.append(record)
                buckets.update(reasons)
        self.dropped = len(self._records) - len(kept)
        return kept


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def near_miss_json(records: list[dict[str, Any]]) -> tuple[str, ...]:
    """Serialize near-miss records as a tuple of JSON strings for npz meta."""
    return tuple(json.dumps(_json_safe(record), sort_keys=True) for record in records)


def near_miss_selection_metadata(config: AuditConfig) -> dict[str, Any]:
    """Describe the retention policy next to the stored near-miss records."""
    return {
        "per_reason_limit": config.near_miss_per_reason,
        "total_limit": config.near_miss_total,
        "preference": "smallest_full_residual_first",
        "multi_label_buckets": (
            "each record occupies one slot in every rejection-reason bucket"
        ),
    }
