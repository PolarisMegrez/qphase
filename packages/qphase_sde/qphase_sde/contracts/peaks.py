"""qphase_sde: Peak Candidate Contracts (2.0)
---------------------------------------------------------
Freezes the unified peak-candidate representation returned by every
``peak_finder`` implementation (prominence, Lorentz, rational, scale-space).
Candidates are rows of a typed table — ragged per-scan-point candidate lists
are stored as typed columns plus row offsets, never as object arrays.

Uncertainty scope is explicit: ``conditional`` (error at the detected
location), ``sampling`` (across independent realizations) and
``path_model_selection`` (path/model ambiguity) are distinct scopes. Missing
capabilities must surface as ``status``/``quality`` flags — never as silent
NaN fields presented as schema-complete.

Path information never enters the point-candidate body; it is expressed by
``PeakPathResult`` through candidate-row references.

Public API
----------
UncertaintyScope
    Explicit uncertainty scopes.
PeakCandidate
    One row of the candidate table.
PeakPathResult
    Scan-axis path built over candidate rows.
PEAK_PRODUCT
    Open product schema template of the peak product.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from qphase.data import AxisSchema, DataKind, ProductSchema, VariableSchema

from .quantities import SDEQuantity

__all__ = [
    "PEAK_PRODUCT",
    "PeakCandidate",
    "PeakPathResult",
    "UncertaintyScope",
]


class UncertaintyScope(str, Enum):
    """Explicit scope of a reported uncertainty."""

    CONDITIONAL = "conditional"
    SAMPLING = "sampling"
    PATH_MODEL_SELECTION = "path_model_selection"


class PeakCandidate(BaseModel):
    """One peak-candidate row of the unified peak table.

    ``conditional_location_std``/``confidence_*`` carry the conditional scope;
    sampling and path/model-selection scopes attach through separate fields or
    the optional fit payload.
    """

    model_config = ConfigDict(extra="forbid")

    location: float
    intensity: float
    conditional_location_std: float | None = None
    confidence_lower: float | None = None
    confidence_upper: float | None = None
    width: float | None = None
    prominence: float | None = None
    curvature: float | None = None
    support: float | None = Field(
        default=None,
        description="Support size (bandwidth/samples) of the detection.",
    )
    quality: float | None = None
    status: Literal["ok", "ambiguous", "degenerate", "rejected"] = "ok"
    fit_payload: dict[str, Any] | None = Field(
        default=None,
        description="Optional model-specific fit parameters/covariance; must "
        "remain JSON-serializable.",
    )


class PeakPathResult(BaseModel):
    """A scan-axis peak path referencing candidate rows of the peak table."""

    model_config = ConfigDict(extra="forbid")

    path_id: str
    candidate_rows: list[int] = Field(
        description="Row indices into the peak candidate table."
    )
    scan_positions: list[int] = Field(
        default_factory=list,
        description="Scan-axis positions traversed by the path.",
    )
    uncertainty_scope: UncertaintyScope = UncertaintyScope.PATH_MODEL_SELECTION
    quality: float | None = None
    status: Literal["ok", "ambiguous", "broken", "rejected"] = "ok"


#: Open schema template of the peak product: a typed candidate table with a
#: ragged ``candidate`` axis closed once candidates are counted.
PEAK_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=[
        AxisSchema(name="scan", independent=True),
        AxisSchema(name="candidate", monotonic=False),
    ],
    variables=[
        VariableSchema(
            name="location",
            dtype="float64",
            value_domain="real",
            dims=("candidate",),
            quantity=SDEQuantity.PEAK_LOCATION.value,
        ),
        VariableSchema(
            name="intensity",
            dtype="float64",
            value_domain="real",
            dims=("candidate",),
            quantity=SDEQuantity.INTENSITY.value,
        ),
    ],
    attributes={
        "table": "peak_candidates",
        "ragged_axis": "candidate",
        "offsets_variable": "candidate_offsets",
    },
)
