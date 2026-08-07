"""Discovery/rejection audit support for bifurcation searches.

The audit makes negative bifurcation results (zero candidates) explainable
without reading debug logs: which structures were searched, how many starts
were generated, how many the physical domain filtered out, the dominant
rejection reasons, and which regions stayed uncovered.

Counting model (schema ``cam_bifurcation_audit/2``)
---------------------------------------------------

Every count belongs to exactly one unit, and every conservation formula only
relates counts of the same unit:

* ``candidate start`` — one start vector handed between pipeline stages.
  ``generated_candidate_count``, ``prefilter_pass_count``,
  ``prefilter_rejected_count``, ``refinement_start_count``,
  ``refinement_duplicate_count``, ``accepted_count`` and ``rejected_count``
  all use this unit, so they may be summed across reductions and across the
  reduced/full paths, and ``totals`` aggregates exactly these fields.
* ``generation trial`` — one evaluated seed-generation attempt.  The trial
  granularity is path-specific (polynomial roots and control points for
  fraction-free reductions, sign-change intervals for condensed reductions,
  fixed-point guesses for full domain sampling, upstream states for upstream
  seeding), so ``generation_trial_count`` is reported per path entry and is
  never summed across paths.
* ``workload`` — path-specific effort subfields (``control_point_count``,
  ``polynomial_root_count``, ``brent_interval_count``,
  ``fixed_point_guess_count``, ``upstream_seed_count``).  Each key is its own
  unit; totals merge them per key only, never across keys.

The conservation formulas are listed in ``AUDIT_CONSERVATION`` and shipped in
every audit payload under ``audit["conservation"]``; the per-field units are
shipped under ``audit["field_units"]``.
"""

from __future__ import annotations

import bisect
import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#: Standard wording attached to empty bifurcation results: the search found
#: no candidate *within the stated numerical coverage*.  It never asserts
#: that no candidate exists.
EMPTY_RESULT_NOTE = "no_candidates_found_within_stated_numerical_coverage"

#: Multi-label semantics of every ``rejected_by_reason`` counter: each
#: rejected candidate increments *all* of its rejection-reason labels, so
#: the per-label counts sum to at least the number of rejected candidates.
#: This counter is deliberately separate from the near-miss retention quota:
#: the counter records every rejection, the quota only bounds how many
#: example records are kept (see ``NearMissStore``).
REJECTION_COUNTER_SEMANTICS = (
    "multi-label: each rejected candidate increments every applicable "
    "rejection reason, so reason counts sum to >= rejected_count"
)

#: Audit payload schema marker stored under ``metadata["audit"]["schema"]``.
AUDIT_SCHEMA = "cam_bifurcation_audit/2"

#: Unit of every audit count field, shipped as ``audit["field_units"]``.
#: Fields sharing the ``candidate start`` unit are the only ones aggregated
#: into ``audit["totals"]``; workload fields keep one unit per key.
AUDIT_FIELD_UNITS: dict[str, str] = {
    "generation_trial_count": (
        "evaluated seed-generation attempts; per-path trial granularity, "
        "reported per path entry and never summed across paths"
    ),
    "generated_candidate_count": (
        "candidate starts produced by seed generation (candidate start unit)"
    ),
    "prefilter_pass_count": (
        "candidate starts passing the physical prefilter (candidate start unit)"
    ),
    "prefilter_rejected_count": (
        "candidate starts rejected by the physical prefilter (candidate start unit)"
    ),
    "refinement_start_count": (
        "candidate starts entering refinement (candidate start unit)"
    ),
    "refinement_duplicate_count": (
        "refined starts dropped as duplicates of an earlier start "
        "(candidate start unit)"
    ),
    "accepted_count": (
        "refined starts accepted as verified candidates (candidate start unit)"
    ),
    "rejected_count": "refined starts rejected (candidate start unit)",
    "control_point_count": (
        "control grid points evaluated (workload unit, all grid-based paths)"
    ),
    "polynomial_root_count": (
        "polynomial roots examined (fraction-free reduction workload unit)"
    ),
    "brent_interval_count": (
        "sign-change intervals examined (condensed reduction workload unit)"
    ),
    "fixed_point_guess_count": (
        "fixed-point solves attempted (full domain-sampling workload unit)"
    ),
    "upstream_seed_count": (
        "upstream states evaluated for seeding (upstream workload unit)"
    ),
}

#: Conservation formulas locked by tests; shipped as ``audit["conservation"]``.
AUDIT_CONSERVATION: tuple[str, ...] = (
    "generation_trial_count = generated_candidate_count + sum(seed_skips.values())"
    "  [per path entry, per-path generation trial unit]",
    "generated_candidate_count = prefilter_pass_count + prefilter_rejected_count"
    "  [candidate start unit]",
    "prefilter_rejected_count = sum(prefilter_rejected.values())"
    "  [candidate start unit]",
    "refinement_start_count = prefilter_pass_count  [candidate start unit]",
    "refinement_start_count = accepted_count + rejected_count"
    " + refinement_duplicate_count  [candidate start unit]",
    "sum(rejected_by_reason.values()) >= rejected_count  [multi-label counter]",
    "near_miss_saved + near_miss_dropped = rejected_count  [candidate start unit]",
    "totals aggregate only candidate-start fields; workload fields are merged"
    " per unit key; generation_trial_count is never summed across paths",
)


class AuditConfig(BaseModel):
    """Bounded near-miss retention limits for the bifurcation audit."""

    model_config = ConfigDict(extra="forbid")

    near_miss_per_reason: int = Field(8, ge=0)
    near_miss_total: int = Field(64, ge=0)


class NearMissStore:
    """Strictly bounded retention of rejected candidates, near-misses first.

    Storage is bounded at ``add()`` time; nothing is accumulated for a later
    truncation pass.  A record is retained only if *both* quotas can keep it
    among the best records seen so far:

    * every one of its rejection-reason buckets currently holds fewer than
      ``per_reason`` records, or its worst record is worse than the new one
      (the bucket then evicts its worst record — a strict per-label cap: a
      label count never exceeds ``per_reason``); and
    * the global pool holds fewer than ``total`` records, or its worst record
      is worse than the new one (the pool then evicts its worst record).

    A multi-label record is all-or-nothing: if any of its labels is already
    saturated by better records, the whole record is rejected and none of its
    label buckets grow.  Evicted records are removed from the pool and from
    every bucket.  "Better" means smaller ``full_residual`` first, then
    smaller ``search_residual`` (near-miss first); ties keep the earlier
    record.  Only quantities already computed for the rejection decision are
    stored; the store never triggers new linear algebra or verification work.

    Invariants (asserted by tests): at all times
    ``len(pool) <= total``, ``len(bucket) <= per_reason`` per label, the
    number of stored record references is at most
    ``total + n_labels * per_reason``, and
    ``added == len(pool) + dropped`` so that
    ``near_miss_saved + near_miss_dropped == rejected_count``.
    """

    def __init__(self, *, per_reason: int, total: int) -> None:
        self.per_reason = per_reason
        self.total = total
        # Entries are (sort_key, sequence, record) tuples; the sequence keeps
        # tuple comparison from ever reaching the record dict and makes ties
        # keep the earlier record.  Lists are sorted ascending by sort_key.
        self._pool: list[tuple[tuple[float, float], int, dict[str, Any]]] = []
        self._buckets: dict[
            str, list[tuple[tuple[float, float], int, dict[str, Any]]]
        ] = {}
        self._sequence = 0
        self._added = 0
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
        self._added += 1
        if self.per_reason <= 0 or self.total <= 0:
            self.dropped += 1
            return
        record: dict[str, Any] = {
            "path": path,
            "reduction": reduction,
            "seed_source": seed_source,
            "controls": {name: float(value) for name, value in controls.items()},
            "rejection_reasons": tuple(dict.fromkeys(rejection_reasons))
            or ("unknown",),
            "search_residual": float(search_residual),
            "full_residual": float(full_residual),
            "min_state_eigenvalue": float(min_state_eigenvalue),
        }
        key = self._sort_key(record)
        reasons = record["rejection_reasons"]
        # Pool admission: strict total cap with worst-record eviction.
        if len(self._pool) >= self.total and key >= self._pool[-1][0]:
            self.dropped += 1
            return
        # Bucket admission: strict per-label cap with worst-record eviction;
        # a multi-label record is rejected as a whole when any of its labels
        # is saturated by better-or-equal records.
        buckets = []
        for reason in reasons:
            bucket = self._buckets.setdefault(reason, [])
            if len(bucket) >= self.per_reason and key >= bucket[-1][0]:
                self.dropped += 1
                return
            buckets.append(bucket)
        entry = (key, self._sequence, record)
        self._sequence += 1
        if len(self._pool) >= self.total:
            self._evict(self._pool.pop(), from_pool=False)
        bisect.insort(self._pool, entry)
        for reason, bucket in zip(reasons, buckets, strict=True):
            bisect.insort(bucket, entry)
            if len(bucket) > self.per_reason:
                self._evict(bucket.pop(), from_pool=True, skip_bucket=reason)

    def _evict(
        self,
        entry: tuple[tuple[float, float], int, dict[str, Any]],
        *,
        from_pool: bool,
        skip_bucket: str | None = None,
    ) -> None:
        """Remove an evicted entry from the pool and its other buckets."""
        if from_pool:
            self._discard(self._pool, entry)
        for reason in entry[2]["rejection_reasons"]:
            if reason == skip_bucket:
                continue
            bucket = self._buckets.get(reason)
            if bucket is not None:
                self._discard(bucket, entry)
        self.dropped += 1

    @staticmethod
    def _discard(
        entries: list[tuple[tuple[float, float], int, dict[str, Any]]],
        entry: tuple[tuple[float, float], int, dict[str, Any]],
    ) -> None:
        for index, item in enumerate(entries):
            if item[1] == entry[1]:
                entries.pop(index)
                return

    @staticmethod
    def _sort_key(record: dict[str, Any]) -> tuple[float, float]:
        def finite(value: float) -> float:
            return value if math.isfinite(value) else math.inf

        return (finite(record["full_residual"]), finite(record["search_residual"]))

    def finalize(self) -> list[dict[str, Any]]:
        """Return the bounded near-miss selection, best records first.

        The pool is already the bounded selection, so this is O(pool); the
        ``dropped`` counter is maintained incrementally by ``add()`` and
        satisfies ``added == len(selection) + dropped``.
        """
        return [entry[2] for entry in self._pool]


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
        "admission": (
            "strict: a record is retained only if every rejection-reason "
            "bucket and the total pool can keep it among the best records; "
            "a record with any label saturated by better records is rejected "
            "as a whole and that label never exceeds its limit"
        ),
        "multi_label_buckets": (
            "each retained record occupies one slot in every rejection-reason "
            "bucket; per-label counts never exceed per_reason_limit"
        ),
        "storage_bound": (
            "bounded at add() time: at most total_limit pool entries plus "
            "per_reason_limit entries per reason bucket; rejected candidates "
            "are not accumulated for later truncation"
        ),
        "counter_vs_quota": (
            "rejected_by_reason counts every rejected candidate under all of "
            "its labels; the near-miss quotas only bound how many example "
            "records are retained and never affect the counters"
        ),
    }
