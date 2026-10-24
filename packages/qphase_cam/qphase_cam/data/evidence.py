"""qphase_cam: the search-evidence product builder
---------------------------------------------------------
One evidence product is the typed record of what a search actually did: one row
per search case with its candidate-start counters, its coverage and truncation
semantics, the identities it accepted and the bounded near misses it kept.

Three rules shape it:

* **Counts keep their unit.** The typed count columns are the candidate-start
  population only, because that is the only unit that may be aggregated. The
  per-path generation-trial and workload counters are carried verbatim in
  ``path_audit`` and are never summed into an unqualified total.
* **A negative result is a status.** ``not_found`` (searched, accepted nothing),
  ``not_searched`` (never looked) and ``unchecked`` (no physical verdict) are
  distinct :class:`~qphase_cam.outcomes.CAMOutcomeStatus` values; none of
  them collapses into "failed".
* **Incomplete provenance fails.** A complete product needs at least one source
  artifact, unique case identities and conserving counters. A solver intermediate
  is representable only through an explicit ``complete=False`` plus a stated
  reason, which is recorded in the attributes — an intermediate is never written
  as if it were final evidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from qphase.data import Dataset, StatisticsDataset, VariableSchema

from ..evidence import (
    AUDIT_CONSERVATION,
    AUDIT_FIELD_UNITS,
    EMPTY_RESULT_NOTE,
    REJECTION_COUNTER_SEMANTICS,
    GenerationPathAudit,
    SearchAuditFragment,
    SearchAuditNearMiss,
    SearchControlSpec,
)
from ..outcomes import CAMOutcomeStatus
from ._common import (
    closed_product,
    json_text_column,
    string_column,
)
from .declarations import (
    CAM_EVIDENCE_PRODUCT,
    EVIDENCE_COUNT_FIELDS,
    _evidence_variables,
)

__all__ = [
    "COMPLETE_EVIDENCE",
    "INTERMEDIATE_EVIDENCE",
    "SourceArtifactRef",
    "build_evidence_product",
    "fragments_from_candidate_product",
    "fragments_from_evidence_product",
]

#: Completeness marker of a product built from final evidence.
COMPLETE_EVIDENCE = "complete"

#: Completeness marker of a product built from a solver intermediate state.
INTERMEDIATE_EVIDENCE = "intermediate"


class SourceArtifactRef(BaseModel):
    """A reference to the artifact an evidence row was derived from.

    Artifacts are referenced by ``artifact_id`` (plus an optional job name), never
    by file path: a path is a local accident, an artifact id is what the catalog
    can resolve and what a re-run can cite.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(min_length=1)
    job: str | None = None

    @model_validator(mode="after")
    def _check_artifact_id(self) -> SourceArtifactRef:
        if ":" in self.artifact_id:
            raise ValueError(
                f"artifact_id {self.artifact_id!r} must not contain ':' "
                "(reserved as the catalog object-id separator)"
            )
        if any(sep in self.artifact_id for sep in ("/", "\\")):
            raise ValueError(
                f"artifact_id {self.artifact_id!r} looks like a path; evidence "
                "references artifacts by id, not by file location"
            )
        if self.job is not None and not self.job.strip():
            raise ValueError("job must not be blank when given")
        return self


def _coerce_artifacts(values: Sequence[Any]) -> list[SourceArtifactRef]:
    """Return the source artifacts as typed references."""
    refs: list[SourceArtifactRef] = []
    for value in values:
        ref = (
            value
            if isinstance(value, SourceArtifactRef)
            else SourceArtifactRef.model_validate(value)
        )
        refs.append(ref)
    identifiers = [ref.artifact_id for ref in refs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "source artifacts must be unique; duplicated: "
            f"{sorted({name for name in identifiers if identifiers.count(name) > 1})}"
        )
    return refs


def _fragment_cases(
    fragments: Sequence[SearchAuditFragment],
) -> list[SearchAuditFragment]:
    """Return the cases, rejecting duplicates and a non-fragment input."""
    cases: list[SearchAuditFragment] = []
    for fragment in fragments:
        if not isinstance(fragment, SearchAuditFragment):
            raise TypeError(
                "build_evidence_product takes SearchAuditFragment objects; got "
                f"{type(fragment).__name__}. A migration document is not evidence "
                "and never becomes a product."
            )
        cases.append(fragment)
    if not cases:
        raise ValueError("an evidence product needs at least one search case")
    identities = [case.case_identity for case in cases]
    if len(set(identities)) != len(identities):
        raise ValueError(
            "case identities must be unique across the product; duplicated: "
            f"{sorted({name for name in identities if identities.count(name) > 1})}"
        )
    return cases


def _control_axis(cases: Sequence[SearchAuditFragment]) -> list[str]:
    """Return the control names shared by every case."""
    names = list(cases[0].controls)
    for case in cases[1:]:
        if list(case.controls) != names:
            raise ValueError(
                f"case {case.case_identity!r} declares controls {list(case.controls)} "
                f"but case {cases[0].case_identity!r} declares {names}; one evidence "
                "product carries one control set"
            )
    return names


def _offsets(blocks: Sequence[int], name: str) -> np.ndarray:
    """Return one cumulative offset block of length ``n_case + 1``."""
    offsets = np.zeros(len(blocks) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.asarray(blocks, dtype=np.int64))
    if int(offsets[0]) != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError(f"{name} must start at 0 and be monotonic")
    return offsets


def _check_candidate_references(
    identities: Sequence[str], candidate_product: Dataset
) -> None:
    """Fail when an accepted identity is not in the candidate product."""
    available = {
        str(value)
        for value in np.asarray(
            candidate_product.handle("candidate_identity").materialize()
        )
    }
    missing = sorted(set(identities) - available)
    if missing:
        raise ValueError(
            "accepted candidate identities are not present in the candidate "
            f"product: {missing}"
        )


def build_evidence_product(
    *,
    fragments: Sequence[SearchAuditFragment],
    source_artifacts: Sequence[SourceArtifactRef | Mapping[str, Any]],
    search_design: Mapping[str, Any],
    candidate_product: Dataset | None = None,
    complete: bool = True,
    intermediate_reason: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Dataset:
    """Build one search-evidence product from validated fragments.

    ``source_artifacts`` are catalog references, not paths; a complete product
    requires at least one. ``candidate_product`` is optional: when given, every
    accepted identity is checked against its ``candidate_identity`` column, so a
    reference to a candidate that does not exist fails instead of dangling.

    ``complete=False`` marks a solver intermediate. It must state
    ``intermediate_reason`` and is recorded as such in the product attributes, so
    an intermediate can never be mistaken for final evidence; it also permits an
    empty ``source_artifacts`` list.

    Raises
    ------
    ValueError
        If no case is given, case identities repeat, the control sets differ, a
        complete product has no source artifact, an intermediate has no reason,
        or an accepted identity is missing from the candidate product.

    """
    cases = _fragment_cases(fragments)
    artifacts = _coerce_artifacts(source_artifacts)
    if complete and not artifacts:
        raise ValueError(
            "a complete evidence product needs at least one source artifact; pass "
            "complete=False with intermediate_reason for a solver intermediate"
        )
    if not complete:
        reason = (intermediate_reason or "").strip()
        if not reason:
            raise ValueError(
                "complete=False requires an intermediate_reason so the product "
                "cannot be mistaken for final evidence"
            )
    else:
        reason = None
    controls = _control_axis(cases)
    # Two layers: the final (deduplicated) candidate rows, and the accepted
    # starts that produced them. An accepted start is never a candidate row, so
    # the candidate cross-check only ever sees the final identities.
    accepted_identities = [
        identity for case in cases for identity in case.accepted_candidate_identities
    ]
    accepted_start_identities = [
        identity for case in cases for identity in case.accepted_start_identities
    ]
    if candidate_product is not None:
        _check_candidate_references(accepted_identities, candidate_product)

    widths: dict[str, int] = {}

    def text(values: Sequence[str]) -> np.ndarray:
        column = string_column(list(values))
        return column

    def json_column(values: Sequence[Any]) -> np.ndarray:
        return json_text_column(list(values))

    case_identity = text([case.case_identity for case in cases])
    search_identity = text([case.search_identity for case in cases])
    status = text([case.status.value for case in cases])
    physical_status = text([case.physical_status for case in cases])
    search_design_column = json_column([case.search_design for case in cases])
    controls_column = json_column(
        [
            {
                name: {"bounds": list(spec.bounds), "sampling": spec.sampling}
                for name, spec in case.controls.items()
            }
            for case in cases
        ]
    )
    notes_column = json_column([list(case.notes) for case in cases])
    coverage = json_column([case.coverage for case in cases])
    path_audit = json_column(
        [
            [
                {
                    "path": entry.path,
                    "generation_trial_count": entry.generation_trial_count,
                    "generated_candidate_count": entry.generated_candidate_count,
                    "seed_skips": dict(entry.seed_skips),
                    "seed_truncated": entry.seed_truncated,
                    "workload": dict(entry.workload),
                }
                for entry in case.paths
            ]
            for case in cases
        ]
    )
    prefilter_rejected = json_column([case.prefilter_rejected for case in cases])
    rejected_by_reason = json_column([case.rejected_by_reason for case in cases])
    # The producing environment is evidence: it is carried per case, never
    # deduplicated, so a product mixing solvers or backends keeps both.
    execution_provenance = json_column(
        [
            {
                "solver": case.solver,
                "plugins": dict(case.plugins),
                "distribution": dict(case.distribution),
                "api_revision": case.api_revision,
                "backend": case.backend,
                "dtype": case.dtype,
            }
            for case in cases
        ]
    )

    near_misses = [record for case in cases for record in case.near_misses]
    missing_controls = sorted(
        {
            name
            for record in near_misses
            for name in record.controls
            if name not in controls
        }
    )
    if missing_controls:
        raise ValueError(
            "near-miss controls are not declared by the case controls: "
            f"{missing_controls}"
        )

    arrays: dict[str, np.ndarray] = {
        "case_identity": case_identity,
        "search_identity": search_identity,
        "status": status,
        "physical_status": physical_status,
        "search_design": search_design_column,
        "controls": controls_column,
        "notes": notes_column,
        "coverage": coverage,
        "path_audit": path_audit,
        "execution_provenance": execution_provenance,
        "prefilter_rejected": prefilter_rejected,
        "rejected_by_reason": rejected_by_reason,
        **{
            field: np.asarray([getattr(case, field) for case in cases], dtype=np.int64)
            for field in EVIDENCE_COUNT_FIELDS
        },
        "accepted_offsets": _offsets(
            [len(case.accepted_candidate_identities) for case in cases],
            "accepted_offsets",
        ),
        "accepted_start_offsets": _offsets(
            [len(case.accepted_start_identities) for case in cases],
            "accepted_start_offsets",
        ),
        "near_miss_offsets": _offsets(
            [case.near_miss_saved for case in cases], "near_miss_offsets"
        ),
        "accepted_identity": text(accepted_identities),
        "accepted_start_identity": text(accepted_start_identities),
        "control_names": text(controls),
        "near_miss_path": text([record.path for record in near_misses]),
        "near_miss_reduction": text([record.reduction or "" for record in near_misses]),
        "near_miss_seed_source": text([record.seed_source for record in near_misses]),
        "near_miss_rejection_reasons": json_column(
            [list(record.rejection_reasons) for record in near_misses]
        ),
        "near_miss_search_residual": np.asarray(
            [record.search_residual for record in near_misses], dtype=np.float64
        ),
        "near_miss_full_residual": np.asarray(
            [record.full_residual for record in near_misses], dtype=np.float64
        ),
        "near_miss_min_state_eigenvalue": np.asarray(
            [
                np.nan
                if record.min_state_eigenvalue is None
                else record.min_state_eigenvalue
                for record in near_misses
            ],
            dtype=np.float64,
        ),
        "near_miss_control_values": np.asarray(
            [
                [record.controls.get(name, np.nan) for name in controls]
                for record in near_misses
            ],
            dtype=np.float64,
        ).reshape(len(near_misses), len(controls)),
    }
    for name, array in arrays.items():
        if array.dtype.kind == "U":
            widths[name] = int(array.dtype.itemsize // 4)

    n_case = len(cases)
    attributes = {
        **CAM_EVIDENCE_PRODUCT.attributes,
        "case_count": n_case,
        "accepted_start_count": len(accepted_start_identities),
        "accepted_candidate_count": len(accepted_identities),
        "near_miss_count": len(near_misses),
        "control_names": list(controls),
        "search_design": dict(search_design),
        "source_artifacts": [ref.model_dump(mode="json") for ref in artifacts],
        "evidence_completeness": COMPLETE_EVIDENCE
        if complete
        else INTERMEDIATE_EVIDENCE,
        "intermediate_reason": reason,
        "status_vocabulary": [member.value for member in CAMOutcomeStatus],
        "field_units": dict(AUDIT_FIELD_UNITS),
        "conservation": list(AUDIT_CONSERVATION),
        "rejection_counter_semantics": REJECTION_COUNTER_SEMANTICS,
        "empty_result_note": EMPTY_RESULT_NOTE,
    }
    schema = closed_product(
        CAM_EVIDENCE_PRODUCT,
        sizes={
            "case": n_case,
            "case_offset": n_case + 1,
            "accepted": len(accepted_identities),
            "accepted_start": len(accepted_start_identities),
            "near_miss": len(near_misses),
            "control": len(controls),
        },
        variables=_evidence_variables_with_widths(widths),
        coordinates=[],
        attributes=attributes,
    )
    return StatisticsDataset.from_arrays(
        schema,
        arrays,
        owner="engine.cam",
        provenance={
            **dict(provenance or {}),
            "evidence_completeness": attributes["evidence_completeness"],
            "search_identities": sorted({case.search_identity for case in cases}),
            "execution_provenance": [
                {
                    "solver": case.solver,
                    "plugins": dict(case.plugins),
                    "distribution": dict(case.distribution),
                    "api_revision": case.api_revision,
                    "backend": case.backend,
                    "dtype": case.dtype,
                }
                for case in cases
            ],
        },
    )


def _evidence_variables_with_widths(
    widths: Mapping[str, int],
) -> list[VariableSchema]:
    """Return the evidence columns with data-dependent string widths."""
    variables = []
    for variable in _evidence_variables():
        width = widths.get(variable.name)
        if width is None:
            variables.append(variable)
            continue
        variables.append(
            variable.model_copy(update={"dtype": np.dtype(f"U{max(width, 1)}").str})
        )
    return variables


# --- readers ---------------------------------------------------------------


def _column(product: Dataset, name: str) -> np.ndarray:
    """Return one materialized column of an evidence or candidate product."""
    try:
        handle = product.handle(name)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"product has no {name!r} column; a reader needs the full column set"
        ) from exc
    return np.asarray(handle.materialize())


def _json_rows(product: Dataset, name: str) -> list[Any]:
    """Return one JSON column decoded per row."""
    return [json.loads(str(value)) for value in _column(product, name)]


def _controls_from_json(
    value: Any, *, case_identity: str
) -> dict[str, SearchControlSpec]:
    """Rebuild one case's control specs from its canonical JSON column entry."""
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            f"case {case_identity!r}: the controls column carries no control "
            "contract; the product cannot be read back into a fragment"
        )
    specs: dict[str, SearchControlSpec] = {}
    for name, payload in value.items():
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"case {case_identity!r}: control {name!r} is not a mapping"
            )
        bounds = payload.get("bounds")
        if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)):
            raise ValueError(
                f"case {case_identity!r}: control {name!r} carries no bounds pair"
            )
        specs[str(name)] = SearchControlSpec(
            bounds=(float(bounds[0]), float(bounds[1])),
            sampling=str(payload.get("sampling", "")),
        )
    return specs


def _paths_from_json(
    value: Any, *, case_identity: str
) -> tuple[GenerationPathAudit, ...]:
    """Rebuild one case's generation-path entries from its JSON column entry."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(
            f"case {case_identity!r}: path_audit is not a sequence of entries"
        )
    return tuple(
        GenerationPathAudit(
            path=str(entry["path"]),
            generation_trial_count=int(entry["generation_trial_count"]),
            generated_candidate_count=int(entry["generated_candidate_count"]),
            seed_skips={
                str(key): int(item)
                for key, item in (entry.get("seed_skips") or {}).items()
            },
            seed_truncated=bool(entry.get("seed_truncated", False)),
            workload={
                str(key): int(item)
                for key, item in (entry.get("workload") or {}).items()
            },
        )
        for entry in value
    )


def _execution_provenance(value: Any, *, case_identity: str) -> dict[str, Any]:
    """Return the producing environment of one case."""
    if not isinstance(value, Mapping):
        raise ValueError(
            f"case {case_identity!r}: execution_provenance is not a mapping"
        )
    missing = sorted(
        {
            "solver",
            "plugins",
            "distribution",
            "api_revision",
            "backend",
            "dtype",
        }
        - set(value)
    )
    if missing:
        raise ValueError(
            f"case {case_identity!r}: execution_provenance lacks {missing}; a "
            "reader never fills provenance with a default"
        )
    return dict(value)


def fragments_from_evidence_product(
    product: Dataset,
) -> tuple[SearchAuditFragment, ...]:
    """Rebuild every search case of one evidence product as a fragment.

    The reader is symmetric to :func:`build_evidence_product`: it reuses the
    stored counts, the accepted/start identity blocks, the path audit, the
    near-miss block and each case's execution provenance, so a summarized
    evidence product preserves the counts and the provenance of its inputs
    instead of recomputing a scientific value. Missing contract columns (for
    example an evidence product written before the per-case ``controls`` column
    existed) raise rather than being replaced by a guess.

    ``unique_candidate_count`` is derived: it is left unset when every accepted
    start produced its own candidate row, because the product stores the final
    identities and that distinction is exactly "nothing was deduplicated".
    """
    identities = [str(value) for value in _column(product, "case_identity")]
    search_identities = [str(value) for value in _column(product, "search_identity")]
    statuses = [str(value) for value in _column(product, "status")]
    physical = [str(value) for value in _column(product, "physical_status")]
    designs = _json_rows(product, "search_design")
    controls = _json_rows(product, "controls")
    coverages = _json_rows(product, "coverage")
    paths = _json_rows(product, "path_audit")
    environments = _json_rows(product, "execution_provenance")
    notes = _json_rows(product, "notes")
    prefilter_rejected = _json_rows(product, "prefilter_rejected")
    rejected_by_reason = _json_rows(product, "rejected_by_reason")
    counts = {
        name: [int(value) for value in _column(product, name)]
        for name in EVIDENCE_COUNT_FIELDS
    }
    accepted_offsets = [int(value) for value in _column(product, "accepted_offsets")]
    start_offsets = [int(value) for value in _column(product, "accepted_start_offsets")]
    near_miss_offsets = [int(value) for value in _column(product, "near_miss_offsets")]
    accepted = [str(value) for value in _column(product, "accepted_identity")]
    starts = [str(value) for value in _column(product, "accepted_start_identity")]
    control_names = [str(value) for value in _column(product, "control_names")]
    near_miss_path = [str(value) for value in _column(product, "near_miss_path")]
    near_miss_reduction = [
        str(value) for value in _column(product, "near_miss_reduction")
    ]
    near_miss_seed = [str(value) for value in _column(product, "near_miss_seed_source")]
    near_miss_reasons = _json_rows(product, "near_miss_rejection_reasons")
    near_miss_search_residual = [
        float(value) for value in _column(product, "near_miss_search_residual")
    ]
    near_miss_full_residual = [
        float(value) for value in _column(product, "near_miss_full_residual")
    ]
    near_miss_min_eigenvalue = [
        float(value) for value in _column(product, "near_miss_min_state_eigenvalue")
    ]
    near_miss_control_values = np.asarray(
        _column(product, "near_miss_control_values"), dtype=float
    ).reshape(len(near_miss_path), len(control_names))
    if len(identities) + 1 != len(accepted_offsets):
        raise ValueError(
            "the case axis and case_offset axis disagree; the product is corrupt"
        )
    if len(identities) + 1 != len(start_offsets) or len(identities) + 1 != len(
        near_miss_offsets
    ):
        raise ValueError("the evidence offsets do not cover the case axis")

    fragments: list[SearchAuditFragment] = []
    for case in range(len(identities)):
        case_identity = identities[case]
        label = f"case {case_identity!r}"
        environment = _execution_provenance(
            environments[case], case_identity=case_identity
        )
        path_entries = _paths_from_json(paths[case], case_identity=case_identity)
        near_miss_block = []
        for index in range(near_miss_offsets[case], near_miss_offsets[case + 1]):
            values = near_miss_control_values[index]
            near_miss_block.append(
                SearchAuditNearMiss(
                    path=near_miss_path[index],
                    reduction=near_miss_reduction[index] or None,
                    seed_source=near_miss_seed[index],
                    controls={
                        name: float(value)
                        for name, value in zip(control_names, values, strict=True)
                        if np.isfinite(value)
                    },
                    rejection_reasons=tuple(
                        str(reason) for reason in near_miss_reasons[index]
                    ),
                    search_residual=near_miss_search_residual[index],
                    full_residual=near_miss_full_residual[index],
                    min_state_eigenvalue=(
                        None
                        if not np.isfinite(near_miss_min_eigenvalue[index])
                        else near_miss_min_eigenvalue[index]
                    ),
                )
            )
        final = accepted[accepted_offsets[case] : accepted_offsets[case + 1]]
        start_block = starts[start_offsets[case] : start_offsets[case + 1]]
        try:
            fragments.append(
                SearchAuditFragment(
                    case_identity=case_identity,
                    search_identity=search_identities[case],
                    status=CAMOutcomeStatus(statuses[case]),
                    physical_status=physical[case],
                    search_design=designs[case],
                    controls=_controls_from_json(
                        controls[case], case_identity=case_identity
                    ),
                    coverage=coverages[case],
                    generated_candidate_count=counts["generated_candidate_count"][case],
                    prefilter_pass_count=counts["prefilter_pass_count"][case],
                    prefilter_rejected_count=counts["prefilter_rejected_count"][case],
                    refinement_start_count=counts["refinement_start_count"][case],
                    refinement_duplicate_count=counts["refinement_duplicate_count"][
                        case
                    ],
                    accepted_count=counts["accepted_count"][case],
                    rejected_count=counts["rejected_count"][case],
                    prefilter_rejected={
                        str(key): int(value)
                        for key, value in prefilter_rejected[case].items()
                    },
                    rejected_by_reason={
                        str(key): int(value)
                        for key, value in rejected_by_reason[case].items()
                    },
                    near_miss_saved=counts["near_miss_saved"][case],
                    near_miss_dropped=counts["near_miss_dropped"][case],
                    unique_candidate_count=(
                        len(final)
                        if len(final) != counts["accepted_count"][case]
                        else None
                    ),
                    paths=path_entries,
                    generation_trial_count=(
                        path_entries[0].generation_trial_count
                        if len(path_entries) == 1
                        else None
                    ),
                    accepted_start_identities=tuple(start_block),
                    accepted_candidate_identities=tuple(final),
                    near_misses=tuple(near_miss_block),
                    solver=str(environment["solver"]),
                    plugins={
                        str(key): str(value)
                        for key, value in environment["plugins"].items()
                    },
                    distribution={
                        str(key): str(value)
                        for key, value in environment["distribution"].items()
                    },
                    api_revision=str(environment["api_revision"]),
                    backend=str(environment["backend"]),
                    dtype=str(environment["dtype"]),
                    notes=tuple(str(note) for note in notes[case]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{label}: cannot be rebuilt as a fragment: {exc}"
            ) from exc
    if not fragments:
        raise ValueError("the evidence product carries no case rows")
    return tuple(fragments)


def fragments_from_candidate_product(
    product: Dataset,
) -> tuple[SearchAuditFragment, ...]:
    """Rebuild the search cases of one candidate product as fragments.

    A solver stores the fragment it derived for each searched case in the
    case metadata (key ``search_fragment``), so a summary of a candidate-only
    artifact reuses exactly what the solver recorded. A case that failed
    carries no fragment — its evidence was never produced — and that is an
    error instead of a silently shorter product.
    """
    rows = _json_rows(product, "case_metadata")
    if not rows:
        raise ValueError("the candidate product carries no case rows")
    fragments: list[SearchAuditFragment] = []
    missing: list[str] = []
    for case, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"case_metadata[{case}] is not a mapping")
        case_identity = str(row.get("case_identity", "")) or f"case-index:{case}"
        payload = row.get("search_fragment")
        if payload is None:
            missing.append(f"{case_identity} ({row.get('case_status', 'unknown')})")
            continue
        fragment = SearchAuditFragment.model_validate(payload)
        if fragment.case_identity != str(
            row.get("case_identity", fragment.case_identity)
        ):
            raise ValueError(
                f"case_metadata[{case}] identity {row.get('case_identity')!r} "
                f"disagrees with its search fragment {fragment.case_identity!r}"
            )
        if str(row.get("search_identity", fragment.search_identity)) != (
            fragment.search_identity
        ):
            raise ValueError(
                f"case {fragment.case_identity!r} search identity disagrees with "
                "its case metadata"
            )
        fragments.append(fragment)
    if missing:
        raise ValueError(
            "the candidate product carries no search fragment for "
            f"{missing}; those cases produced no evidence, so the summary cannot "
            "be reconstructed from candidates alone"
        )
    return tuple(fragments)
