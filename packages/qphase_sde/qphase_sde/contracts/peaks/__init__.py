"""qphase_sde: Peak Contracts
---------------------------------------------------------
Peak candidate, fit-parameter and path contracts: the declarations and
product schemas in :mod:`models`, the payload validators in :mod:`validators`.

Public API
----------
``PeakCandidate`` / ``PeakPathResult`` / ``PeakStatus`` / ``UncertaintyScope``
    Record and enumeration declarations.
``PEAK_PRODUCT`` / ``PEAK_FIT_PRODUCT`` / ``PEAK_PATH_PRODUCT``
    Frozen product schemas.
``CandidateTablePayload``
    Grouped candidate-table payload returned by finders.
``validate_candidate_table`` / ``validate_fit_table`` /
``validate_path_table`` / ``validate_confidence_bounds``
    Payload validators.
"""

from .models import (
    OPTIONAL_CAPABILITY_FIELDS,
    PEAK_CANDIDATE_STATUS_MASK,
    PEAK_FIT_PRODUCT,
    PEAK_PATH_PRODUCT,
    PEAK_PATH_STATUS_MASK,
    PEAK_PRODUCT,
    PEAK_STATUS_MASK,
    CandidateTablePayload,
    PeakCandidate,
    PeakPathResult,
    PeakStatus,
    UncertaintyScope,
)
from .validators import (
    CHANNEL_MEASUREMENT_KINDS,
    validate_candidate_table,
    validate_confidence_bounds,
    validate_fit_table,
    validate_path_table,
)

__all__ = [
    "CHANNEL_MEASUREMENT_KINDS",
    "OPTIONAL_CAPABILITY_FIELDS",
    "PEAK_CANDIDATE_STATUS_MASK",
    "PEAK_FIT_PRODUCT",
    "PEAK_PATH_PRODUCT",
    "PEAK_PATH_STATUS_MASK",
    "PEAK_PRODUCT",
    "PEAK_STATUS_MASK",
    "CandidateTablePayload",
    "PeakCandidate",
    "PeakPathResult",
    "PeakStatus",
    "UncertaintyScope",
    "validate_candidate_table",
    "validate_confidence_bounds",
    "validate_fit_table",
    "validate_path_table",
]
