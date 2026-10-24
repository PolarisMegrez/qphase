"""qphase_cam: outcome and target vocabularies
---------------------------------------------------------
The controlled vocabularies that let a CAM product express a *negative* result
without collapsing every failure into one boolean.

``CAMOutcomeStatus`` answers "what happened to this target?", so a consumer can
tell "we looked and it is not there" (``not_found``) from "we never looked at
that region" (``not_searched``), from "it is there but fails a physical or
statistical check" (``nonphysical``/``unstable``/``stochastic_invalid``). A
single ``rejected`` flag cannot carry that distinction, and a search that
covered its design space and accepted nothing must still be able to say so —
which is the negative-result semantics of Phase 10C2.

``CAMTargetKind`` answers "what is this identity attached to?", so a
``target_identity`` is never ambiguous between a solved state, a candidate, a
continuation path or a scan case.

Both are resource-defined vocabularies, not core enumerations: they live in the
CAM package, they are used by the annotation product (Phase 10C2a) and will be
used by the evidence product (Phase 10C2b).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

__all__ = [
    "CAMTargetKind",
    "CAMOutcomeStatus",
    "coerce_outcome_status",
    "coerce_target_kind",
]


class CAMOutcomeStatus(StrEnum):
    """What happened to a CAM target.

    ``valid``
        The check ran and passed: the target is usable as-is.
    ``not_found``
        The region was searched and the target is not there.
    ``not_searched``
        The region was not searched, so the absence is *not* evidence.
    ``not_converged``
        An attempt was made and the solver did not converge.
    ``nonphysical``
        Converged, but the state violates a physicality requirement.
    ``unstable``
        Converged, but the linear stability requirement fails.
    ``stochastic_invalid``
        Converged and stable, but the stochastic-validity check fails.
    ``unidentifiable``
        Present, but its identity/parameters cannot be determined.
    ``uncontrolled_approximation``
        Only a truncated/uncontrolled approximation of the target exists.
    ``threshold_rejected``
        Rejected by an explicit, configured acceptance threshold.
    ``unchecked``
        Nothing was evaluated; the row records intent, not a verdict.
    """

    VALID = "valid"
    NOT_FOUND = "not_found"
    NOT_SEARCHED = "not_searched"
    NOT_CONVERGED = "not_converged"
    NONPHYSICAL = "nonphysical"
    UNSTABLE = "unstable"
    STOCHASTIC_INVALID = "stochastic_invalid"
    UNIDENTIFIABLE = "unidentifiable"
    UNCONTROLLED_APPROXIMATION = "uncontrolled_approximation"
    THRESHOLD_REJECTED = "threshold_rejected"
    UNCHECKED = "unchecked"


class CAMTargetKind(StrEnum):
    """What a ``target_identity`` refers to.

    ``state``
        One fixed-capacity solution slot (``CAM_STATE_PRODUCT``).
    ``candidate``
        One bifurcation/search candidate (``CAM_CANDIDATE_PRODUCT``).
    ``path``
        One continuation path (``CAM_CONTINUATION_PRODUCT``).
    ``case``
        One scan case, i.e. one point of a flattened parameter grid.
    """

    STATE = "state"
    CANDIDATE = "candidate"
    PATH = "path"
    CASE = "case"


def coerce_outcome_status(value: Any) -> CAMOutcomeStatus:
    """Return the outcome status for ``value``, rejecting anything unknown."""
    if isinstance(value, CAMOutcomeStatus):
        return value
    try:
        return CAMOutcomeStatus(str(value))
    except ValueError as exc:
        allowed = ", ".join(member.value for member in CAMOutcomeStatus)
        raise ValueError(
            f"unknown CAM outcome status {value!r}; expected one of: {allowed}"
        ) from exc


def coerce_target_kind(value: Any) -> CAMTargetKind:
    """Return the target kind for ``value``, rejecting anything unknown."""
    if isinstance(value, CAMTargetKind):
        return value
    try:
        return CAMTargetKind(str(value))
    except ValueError as exc:
        allowed = ", ".join(member.value for member in CAMTargetKind)
        raise ValueError(
            f"unknown CAM target kind {value!r}; expected one of: {allowed}"
        ) from exc
