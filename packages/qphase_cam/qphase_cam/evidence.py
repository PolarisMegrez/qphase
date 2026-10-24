"""qphase_cam: search-audit fragments
---------------------------------------------------------
A :class:`SearchAuditFragment` is the typed, self-validating contract a solver
hands to the evidence layer. It is a *resource-package service contract*
(Phase 10C2b), not a plugin: Phase 10D's solver protocol produces fragments and
Phase 10E's EvidenceAssembler consumes them, but neither plugs into a registry
through this module.

Two rules shape it:

* **One counting system.** Every count name, unit and conservation formula is
  the one already shipped by :mod:`qphase_cam.solver.bifurcation_audit`
  (``AUDIT_CONSERVATION``, ``AUDIT_FIELD_UNITS``,
  ``REJECTION_COUNTER_SEMANTICS``, ``EMPTY_RESULT_NOTE``). There is no parallel
  vocabulary here, and the conservation identities are enforced on construction.
* **No unitless totals.** ``generation_trial_count`` and the workload counters
  are per-path quantities. A fragment carries one
  :class:`GenerationPathAudit` per generation path and never pools them; the
  fragment-level ``generation_trial_count`` is only filled for a single-path
  fragment and is ``None`` otherwise.

A fragment also refuses to pretend completeness. ``solver``, the plugin ids, the
distribution versions, the audit API revision, the backend and the dtype are
mandatory, a physical verdict requires a physical check
(``unchecked`` is not ``failed``), and a case that accepted nothing must say
``not_found`` or ``not_searched`` instead of hiding behind a generic status.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .data.outcomes import CAMOutcomeStatus, coerce_outcome_status
from .solver.bifurcation_audit import (
    AUDIT_CONSERVATION,
    AUDIT_FIELD_UNITS,
    AUDIT_SCHEMA,
    EMPTY_RESULT_NOTE,
    REJECTION_COUNTER_SEMANTICS,
)

__all__ = [
    "AUDIT_CONSERVATION",
    "AUDIT_FIELD_UNITS",
    "AUDIT_SCHEMA",
    "EMPTY_RESULT_NOTE",
    "REJECTION_COUNTER_SEMANTICS",
    "FRAGMENT_SCHEMA",
    "GenerationPathAudit",
    "SearchAuditFragment",
    "SearchAuditNearMiss",
    "SearchControlSpec",
    "fragment_from_bifurcation_metadata",
]

#: Schema marker of one fragment contract.
FRAGMENT_SCHEMA: Literal["qphase_cam.search_audit_fragment/1"] = (
    "qphase_cam.search_audit_fragment/1"
)

#: Physical-verdict states a fragment may claim.
PhysicalStatus = Literal["checked", "not_checked"]

#: Statuses that assert a physical/stochastic verdict about a target.
_PHYSICAL_VERDICTS = frozenset(
    {
        CAMOutcomeStatus.NONPHYSICAL,
        CAMOutcomeStatus.UNSTABLE,
        CAMOutcomeStatus.STOCHASTIC_INVALID,
    }
)


class SearchControlSpec(BaseModel):
    """One search control: its bounds and its sampling rule."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    bounds: tuple[float, float]
    sampling: str = ""

    @model_validator(mode="after")
    def _check_bounds(self) -> SearchControlSpec:
        low, high = self.bounds
        if low > high:
            raise ValueError(
                f"control bounds must satisfy low <= high, got {self.bounds}"
            )
        return self


class GenerationPathAudit(BaseModel):
    """Per-path generation and workload bookkeeping, in per-path units.

    ``generation_trial_count`` is the per-path generation-trial unit:
    ``generated_candidate_count + sum(seed_skips.values())``. ``workload`` keeps
    one unit per key (``control_point_count``, ``polynomial_root_count``, ...) and
    is merged per key, never summed across units.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    generation_trial_count: int = Field(ge=0)
    generated_candidate_count: int = Field(ge=0)
    seed_skips: dict[str, int] = Field(default_factory=dict)
    seed_truncated: bool = False
    workload: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_trial_conservation(self) -> GenerationPathAudit:
        if any(value < 0 for value in self.seed_skips.values()):
            raise ValueError(f"path {self.path!r}: seed_skips must be non-negative")
        if any(value < 0 for value in self.workload.values()):
            raise ValueError(f"path {self.path!r}: workload must be non-negative")
        unknown = sorted(set(self.workload) - set(AUDIT_FIELD_UNITS))
        if unknown:
            raise ValueError(
                f"path {self.path!r}: workload keys without a declared unit: {unknown}"
            )
        expected = self.generated_candidate_count + sum(self.seed_skips.values())
        if self.generation_trial_count != expected:
            raise ValueError(
                f"path {self.path!r}: generation_trial_count must equal "
                f"generated_candidate_count + sum(seed_skips) = {expected}, got "
                f"{self.generation_trial_count}"
            )
        return self


class SearchAuditNearMiss(BaseModel):
    """One retained rejected candidate, bounded by the near-miss quotas."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    path: str = Field(min_length=1)
    reduction: str | None = None
    seed_source: str = ""
    controls: dict[str, float] = Field(default_factory=dict)
    rejection_reasons: tuple[str, ...] = Field(min_length=1)
    search_residual: float
    full_residual: float
    min_state_eigenvalue: float | None = None


class SearchAuditFragment(BaseModel):
    """Typed audit of one search case.

    A fragment is valid only when its candidate-start counts conserve, its
    accepted identities match ``accepted_count``, its near-miss accounting adds
    up, and its status does not contradict what actually ran.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fragment_schema: Literal["qphase_cam.search_audit_fragment/1"] = FRAGMENT_SCHEMA
    case_identity: str = Field(min_length=1)
    search_identity: str = Field(min_length=1)
    status: CAMOutcomeStatus
    physical_status: PhysicalStatus
    search_design: dict[str, Any] = Field(min_length=1)
    controls: dict[str, SearchControlSpec] = Field(min_length=1)
    coverage: dict[str, Any] = Field(min_length=1)

    # -- candidate-start unit: the only counts that may be aggregated ---------
    generated_candidate_count: int = Field(ge=0)
    prefilter_pass_count: int = Field(ge=0)
    prefilter_rejected_count: int = Field(ge=0)
    refinement_start_count: int = Field(ge=0)
    refinement_duplicate_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    prefilter_rejected: dict[str, int] = Field(default_factory=dict)
    rejected_by_reason: dict[str, int] = Field(default_factory=dict)
    near_miss_saved: int = Field(ge=0)
    near_miss_dropped: int = Field(ge=0)
    unique_candidate_count: int | None = Field(default=None, ge=0)

    # -- per-path units: never pooled -----------------------------------------
    paths: tuple[GenerationPathAudit, ...] = ()
    generation_trial_count: int | None = Field(default=None, ge=0)

    # -- evidence payload -----------------------------------------------------
    accepted_candidate_identities: tuple[str, ...] = ()
    near_misses: tuple[SearchAuditNearMiss, ...] = ()

    # -- provenance -----------------------------------------------------------
    solver: str = Field(min_length=1)
    plugins: dict[str, str] = Field(min_length=1)
    distribution: dict[str, str] = Field(min_length=1)
    api_revision: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_conservation(self) -> SearchAuditFragment:
        label = self.case_identity
        if self.generated_candidate_count != (
            self.prefilter_pass_count + self.prefilter_rejected_count
        ):
            raise ValueError(
                f"case {label!r}: generated_candidate_count must equal "
                "prefilter_pass_count + prefilter_rejected_count"
            )
        if self.refinement_start_count != (
            self.accepted_count + self.rejected_count + self.refinement_duplicate_count
        ):
            raise ValueError(
                f"case {label!r}: refinement_start_count must equal accepted_count "
                "+ rejected_count + refinement_duplicate_count"
            )
        if sum(self.prefilter_rejected.values()) != self.prefilter_rejected_count:
            raise ValueError(
                f"case {label!r}: prefilter_rejected must sum to "
                "prefilter_rejected_count"
            )
        if any(value < 0 for value in self.rejected_by_reason.values()):
            raise ValueError(f"case {label!r}: rejected_by_reason must be non-negative")
        if sum(self.rejected_by_reason.values()) < self.rejected_count:
            raise ValueError(
                f"case {label!r}: rejected_by_reason is a multi-label counter and "
                "must sum to at least rejected_count"
            )
        if self.near_miss_saved + self.near_miss_dropped != self.rejected_count:
            raise ValueError(
                f"case {label!r}: near_miss_saved + near_miss_dropped must equal "
                "rejected_count"
            )
        return self

    @model_validator(mode="after")
    def _check_evidence_payload(self) -> SearchAuditFragment:
        label = self.case_identity
        if len(self.accepted_candidate_identities) != self.accepted_count:
            raise ValueError(
                f"case {label!r}: {len(self.accepted_candidate_identities)} accepted "
                f"identities for accepted_count={self.accepted_count}"
            )
        identities = list(self.accepted_candidate_identities)
        for position, value in enumerate(identities):
            if not value.strip():
                raise ValueError(
                    f"case {label!r}: accepted identity[{position}] is empty"
                )
        if len(set(identities)) != len(identities):
            raise ValueError(
                f"case {label!r}: accepted candidate identities must be unique"
            )
        if len(self.near_misses) != self.near_miss_saved:
            raise ValueError(
                f"case {label!r}: {len(self.near_misses)} near-miss records for "
                f"near_miss_saved={self.near_miss_saved}"
            )
        if (
            self.unique_candidate_count is not None
            and self.unique_candidate_count > self.accepted_count
        ):
            raise ValueError(
                f"case {label!r}: unique_candidate_count must not exceed accepted_count"
            )
        return self

    @model_validator(mode="after")
    def _check_path_units(self) -> SearchAuditFragment:
        label = self.case_identity
        names = [entry.path for entry in self.paths]
        if len(set(names)) != len(names):
            raise ValueError(f"case {label!r}: path entries must be unique")
        if len(self.paths) > 1 and self.generation_trial_count is not None:
            raise ValueError(
                f"case {label!r}: generation trials are per-path and must not be "
                "pooled; leave generation_trial_count unset for a multi-path case"
            )
        if (
            len(self.paths) == 1
            and self.generation_trial_count is not None
            and self.generation_trial_count != self.paths[0].generation_trial_count
        ):
            raise ValueError(
                f"case {label!r}: generation_trial_count disagrees with the single "
                "path entry"
            )
        return self

    @model_validator(mode="after")
    def _check_status_semantics(self) -> SearchAuditFragment:
        label = self.case_identity
        if self.physical_status == "not_checked" and self.status in _PHYSICAL_VERDICTS:
            raise ValueError(
                f"case {label!r}: status {self.status.value!r} claims a physical "
                "verdict but physical_status is 'not_checked'; an unchecked "
                "physicality is not a failure"
            )
        if self.status is CAMOutcomeStatus.NOT_SEARCHED:
            examined = (
                self.generated_candidate_count
                + self.refinement_start_count
                + self.accepted_count
                + self.near_miss_saved
            )
            if examined:
                raise ValueError(
                    f"case {label!r}: status 'not_searched' cannot carry searched "
                    "counts; that absence is not evidence"
                )
        if self.status is CAMOutcomeStatus.NOT_FOUND:
            if self.accepted_count:
                raise ValueError(
                    f"case {label!r}: status 'not_found' cannot carry accepted "
                    "candidates"
                )
            if not (self.generated_candidate_count or self.refinement_start_count):
                raise ValueError(
                    f"case {label!r}: status 'not_found' requires that something was "
                    "actually searched; use 'not_searched' otherwise"
                )
        if self.status is CAMOutcomeStatus.VALID and not self.accepted_count:
            raise ValueError(
                f"case {label!r}: status 'valid' requires at least one accepted "
                "candidate; a completed search without one is 'not_found'"
            )
        return self

    @model_validator(mode="after")
    def _check_provenance(self) -> SearchAuditFragment:
        label = self.case_identity
        for field in ("solver", "api_revision", "backend", "dtype"):
            if not str(getattr(self, field)).strip():
                raise ValueError(f"case {label!r}: {field} must not be blank")
        for field in ("plugins", "distribution"):
            values = getattr(self, field)
            blank = sorted(key for key, value in values.items() if not value.strip())
            if blank:
                raise ValueError(
                    f"case {label!r}: {field} entries must not be blank: {blank}"
                )
        return self

    # -- derived views --------------------------------------------------------

    @property
    def accepted(self) -> bool:
        """Whether the case accepted at least one candidate."""
        return self.accepted_count > 0

    def to_status(self) -> CAMOutcomeStatus:
        """Return the case status."""
        return self.status


# --- construction from the current solver metadata -------------------------


def _search_controls(metadata: Mapping[str, Any]) -> dict[str, SearchControlSpec]:
    """Return the control specs of one search from its metadata."""
    names = [str(name) for name in metadata.get("control_names", ())]
    if not names:
        raise ValueError(
            "metadata['control_names'] is required to map the search controls"
        )
    domain = metadata.get("domain_coverage") or {}
    bounds = list(domain.get("control_bounds") or ())
    sampling = dict(domain.get("control_sampling") or {})
    if len(bounds) != len(names):
        raise ValueError(
            f"metadata carries {len(bounds)} control bounds for {len(names)} "
            "control names; cannot map the controls"
        )
    specs: dict[str, SearchControlSpec] = {}
    for name, pair in zip(names, bounds, strict=True):
        values = [float(value) for value in pair]
        if len(values) != 2:
            raise ValueError(f"control {name!r} bounds must be a [low, high] pair")
        specs[name] = SearchControlSpec(
            bounds=(values[0], values[1]),
            sampling=str(sampling.get(name, "")),
        )
    return specs


def _search_design(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the search design of one case from its metadata."""
    design = {
        "target": metadata.get("target"),
        "order": metadata.get("order"),
        "strategy": metadata.get("strategy"),
        "state_ids": list(metadata.get("state_ids") or ()),
        "perturbation_parameter": metadata.get("perturbation_parameter"),
        "perturbation_scale": metadata.get("perturbation_scale"),
        "perturbation_side": metadata.get("perturbation_side"),
        "reduced_degree": metadata.get("reduced_degree"),
    }
    cleaned = {key: value for key, value in design.items() if value is not None}
    if not cleaned:
        raise ValueError("metadata carries no search design fields")
    return cleaned


def _search_coverage(
    metadata: Mapping[str, Any], audit: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the coverage/truncation/materialization semantics of one case."""
    coverage = {
        "coverage": metadata.get("coverage"),
        "structural_coverage": metadata.get("structural_coverage"),
        "numerical_coverage": metadata.get("numerical_coverage"),
        "singular_coverage": metadata.get("singular_coverage"),
        "domain_coverage": metadata.get("domain_coverage") or {},
        "seed_truncated": _pooled_seed_truncated(audit),
        "result_note": audit.get("result_note"),
        "near_miss_selection": audit.get("near_miss_selection") or {},
        "rejection_counter_semantics": audit.get("rejection_counter_semantics")
        or REJECTION_COUNTER_SEMANTICS,
        "empty_result_note": EMPTY_RESULT_NOTE,
    }
    if not any(value for key, value in coverage.items() if key != "empty_result_note"):
        raise ValueError("metadata carries no coverage information")
    return coverage


def _pooled_seed_truncated(audit: Mapping[str, Any]) -> bool:
    """Return whether any path entry truncated its seed enumeration."""
    totals = audit.get("totals")
    if isinstance(totals, Mapping) and "seed_truncated" in totals:
        return bool(totals["seed_truncated"])
    return bool(audit.get("seed_truncated", False))


def _counter_totals(audit: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the candidate-start counters of one audit payload.

    The reduced/auto payload aggregates them under ``totals``; a standalone full
    payload carries them at the top level. A payload with neither cannot be
    mapped and raises.
    """
    totals = audit.get("totals")
    if isinstance(totals, Mapping):
        return totals
    if "generated_candidate_count" in audit:
        return audit
    raise ValueError(
        "audit payload carries no candidate-start counters (neither 'totals' nor "
        "top-level counts)"
    )


def _path_entries(
    metadata: Mapping[str, Any], audit: Mapping[str, Any]
) -> tuple[GenerationPathAudit, ...]:
    """Return one entry per generation path of one case."""
    raw: list[Mapping[str, Any]] = []
    reductions = audit.get("reductions")
    if isinstance(reductions, Sequence) and not isinstance(reductions, (str, bytes)):
        raw.extend(item for item in reductions if isinstance(item, Mapping))
    fallback = metadata.get("full_fallback")
    if isinstance(fallback, Mapping):
        fallback_audit = fallback.get("audit")
        if isinstance(fallback_audit, Mapping):
            raw.append(fallback_audit)
    if not raw and "generation_trial_count" in audit:
        raw.append(audit)
    if not raw:
        raise ValueError(
            "audit payload carries no generation path entries; per-path trial "
            "counts cannot be mapped"
        )
    entries: list[GenerationPathAudit] = []
    for item in raw:
        path = item.get("path") or item.get("method") or item.get("seed_source")
        if not path:
            raise ValueError(
                "a generation path entry carries no path/method/seed_source id"
            )
        entries.append(
            GenerationPathAudit(
                path=str(path),
                generation_trial_count=int(item["generation_trial_count"]),
                generated_candidate_count=int(item["generated_candidate_count"]),
                seed_skips={
                    str(key): int(value)
                    for key, value in (item.get("seed_skips") or {}).items()
                },
                seed_truncated=bool(item.get("seed_truncated", False)),
                workload={
                    str(key): int(value)
                    for key, value in (item.get("workload") or {}).items()
                },
            )
        )
    return tuple(entries)


def _near_misses(audit: Mapping[str, Any]) -> tuple[SearchAuditNearMiss, ...]:
    """Return the retained near misses of one case."""
    records = audit.get("near_misses")
    if records is None:
        return ()
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("audit['near_misses'] must be a sequence of records")
    return tuple(
        SearchAuditNearMiss(
            path=str(record["path"]),
            reduction=(
                None if record.get("reduction") is None else str(record["reduction"])
            ),
            seed_source=str(record.get("seed_source", "")),
            controls={
                str(key): float(value)
                for key, value in (record.get("controls") or {}).items()
            },
            rejection_reasons=tuple(
                str(reason) for reason in record.get("rejection_reasons", ())
            ),
            search_residual=float(record["search_residual"]),
            full_residual=float(record["full_residual"]),
            min_state_eigenvalue=_optional_float(record.get("min_state_eigenvalue")),
        )
        for record in records
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def _installed_distribution() -> dict[str, str]:
    """Return the installed distribution versions of the CAM stack."""
    versions: dict[str, str] = {}
    for name in ("qphase", "qphase-cam", "qpsgen"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - always installed here
            continue
    if not versions:
        raise ValueError("no installed distribution version could be read")
    return versions


def _derive_status(audit: Mapping[str, Any], accepted_count: int, *, examined: int):
    """Derive the case status from what the search actually did."""
    note = audit.get("result_note")
    if note is not None and str(note) == EMPTY_RESULT_NOTE:
        return CAMOutcomeStatus.NOT_FOUND
    if accepted_count:
        return CAMOutcomeStatus.VALID
    if examined:
        return CAMOutcomeStatus.NOT_FOUND
    return CAMOutcomeStatus.NOT_SEARCHED


def fragment_from_bifurcation_metadata(
    metadata: Mapping[str, Any],
    *,
    case_identity: str,
    search_identity: str,
    accepted_candidate_identities: Sequence[str],
    backend: str,
    dtype: str,
    plugins: Mapping[str, str] | None = None,
    distribution: Mapping[str, str] | None = None,
    status: CAMOutcomeStatus | str | None = None,
    physical_status: PhysicalStatus | None = None,
    search_design: Mapping[str, Any] | None = None,
    coverage: Mapping[str, Any] | None = None,
    api_revision: str | None = None,
    solver: str = "bifurcation",
    notes: Sequence[str] = (),
) -> SearchAuditFragment:
    """Build one fragment from the metadata of one solver case.

    This is the explicit mapping Phase 10D will call. It reads the audit payload
    the solver already produces — never a post-hoc reconstruction — and raises
    when a required piece is missing instead of filling it with a default:
    unknown audit schema, missing counters, missing path entries, missing
    coverage or missing physical status all fail.

    ``backend``/``dtype`` are arguments because the audit metadata does not record
    them; the plugin ids default to the target and strategy the metadata names,
    so a caller must add the discovery/classifier ids it owns.
    """
    audit = metadata.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError(f"case {case_identity!r}: metadata carries no audit payload")
    schema = audit.get("schema")
    if schema is not None and str(schema) != AUDIT_SCHEMA:
        raise ValueError(
            f"case {case_identity!r}: unrecognized audit schema {schema!r}; "
            f"expected {AUDIT_SCHEMA!r}"
        )
    totals = _counter_totals(audit)
    accepted_count = int(totals["accepted_count"])
    names = list(accepted_candidate_identities)
    if len(names) != accepted_count:
        unique = totals.get("unique_candidate_count")
        hint = (
            " A multi-path case is deduplicated before packing, so several "
            "accepted starts can collapse into one candidate row; one identity "
            "per accepted start needs the solver-owned identity of Phase 10D."
            if unique is not None and int(unique) < accepted_count
            else ""
        )
        raise ValueError(
            f"case {case_identity!r}: {len(names)} accepted identities for "
            f"audit accepted_count={accepted_count}.{hint}"
        )

    resolved_plugins = {
        str(key): str(value)
        for key, value in (
            metadata.get("target") and {"target": metadata["target"]} or {}
        ).items()
    }
    if metadata.get("strategy"):
        resolved_plugins["strategy"] = str(metadata["strategy"])
    resolved_plugins.update(
        {str(key): str(value) for key, value in (plugins or {}).items()}
    )
    if not resolved_plugins:
        raise ValueError(
            f"case {case_identity!r}: no plugin ids are available; pass plugins="
        )

    declared_physical: str | None = physical_status
    if declared_physical is None:
        raw = audit.get("physical_status")
        if raw is None:
            raise ValueError(
                f"case {case_identity!r}: audit carries no physical_status; an "
                "unchecked physicality must be stated explicitly"
            )
        declared_physical = str(raw)
    if declared_physical not in ("checked", "not_checked"):
        raise ValueError(
            f"case {case_identity!r}: unknown physical_status {declared_physical!r}"
        )
    resolved_physical = cast("PhysicalStatus", declared_physical)

    near_misses = _near_misses(audit)
    examined = int(totals.get("generated_candidate_count", 0)) + int(
        totals.get("refinement_start_count", 0)
    )
    resolved_status = (
        coerce_outcome_status(status)
        if status is not None
        else _derive_status(audit, accepted_count, examined=examined)
    )
    resolution = {
        "search_design": (
            dict(search_design)
            if search_design is not None
            else _search_design(metadata)
        ),
        "coverage": (
            dict(coverage)
            if coverage is not None
            else _search_coverage(metadata, audit)
        ),
    }
    return SearchAuditFragment(
        case_identity=str(case_identity),
        search_identity=str(search_identity),
        status=resolved_status,
        physical_status=resolved_physical,
        search_design=resolution["search_design"],
        controls=_search_controls(metadata),
        coverage=resolution["coverage"],
        generated_candidate_count=int(totals["generated_candidate_count"]),
        prefilter_pass_count=int(totals["prefilter_pass_count"]),
        prefilter_rejected_count=int(totals["prefilter_rejected_count"]),
        refinement_start_count=int(totals["refinement_start_count"]),
        refinement_duplicate_count=int(totals["refinement_duplicate_count"]),
        accepted_count=accepted_count,
        rejected_count=int(totals["rejected_count"]),
        prefilter_rejected={
            str(key): int(value)
            for key, value in (totals.get("prefilter_rejected") or {}).items()
        },
        rejected_by_reason={
            str(key): int(value)
            for key, value in (totals.get("rejected_by_reason") or {}).items()
        },
        near_miss_saved=int(totals["near_miss_saved"]),
        near_miss_dropped=int(totals["near_miss_dropped"]),
        unique_candidate_count=(
            None
            if totals.get("unique_candidate_count") is None
            else int(totals["unique_candidate_count"])
        ),
        paths=_path_entries(metadata, audit),
        generation_trial_count=_single_path_trials(metadata, audit),
        accepted_candidate_identities=tuple(names),
        near_misses=near_misses,
        solver=solver,
        plugins=resolved_plugins,
        distribution={
            str(key): str(value)
            for key, value in (
                distribution if distribution is not None else _installed_distribution()
            ).items()
        },
        api_revision=str(api_revision or schema or AUDIT_SCHEMA),
        backend=str(backend),
        dtype=str(dtype),
        notes=tuple(str(note) for note in notes),
    )


def _single_path_trials(
    metadata: Mapping[str, Any], audit: Mapping[str, Any]
) -> int | None:
    """Return the generation-trial count only when a single path produced it."""
    entries = _path_entries(metadata, audit)
    if len(entries) == 1:
        return entries[0].generation_trial_count
    return None
