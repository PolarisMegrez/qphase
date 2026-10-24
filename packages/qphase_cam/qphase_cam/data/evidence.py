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
    SearchAuditFragment,
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
