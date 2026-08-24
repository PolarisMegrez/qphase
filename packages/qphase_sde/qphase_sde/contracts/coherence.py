"""qphase_sde: Coherence Frequency Contracts (2.0)
---------------------------------------------------------
Freezes the unified coherence-frequency estimate returned by every
``coherence_frequency`` estimator (short-delay, band-limited, finite-delay,
pole model). Estimates are rows of a typed statistics table; lag, bandwidth
and model diagnostics are explicit fields so that estimator changes are always
visible in the schema and provenance.

Public API
----------
CoherenceFrequencyEstimate
    One coherence-frequency estimate row.
COHERENCE_FREQUENCY_PRODUCT
    Open product schema template of the coherence-frequency product.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ProductSchema,
    UncertaintySchema,
    VariableSchema,
)

from .quantities import SDEQuantity

__all__ = [
    "COHERENCE_FREQUENCY_PRODUCT",
    "CoherenceFrequencyEstimate",
]


class CoherenceFrequencyEstimate(BaseModel):
    """One coherence-frequency estimate with explicit diagnostics."""

    model_config = ConfigDict(extra="forbid")

    frequency: float
    bandwidth: float | None = None
    lag: float | None = Field(
        default=None, description="Delay/lag the estimate was computed at."
    )
    estimator: str = Field(
        description="Estimator identifier, e.g. 'band_limited'."
    )
    conditional_std: float | None = None
    sampling_std: float | None = None
    confidence_lower: float | None = None
    confidence_upper: float | None = None
    independent_count: int | None = None
    status: Literal["ok", "ambiguous", "degenerate", "rejected"] = "ok"
    diagnostics: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-serializable lag/bandwidth/model diagnostics.",
    )


#: Open schema template of the coherence-frequency product. The sampling
#: uncertainty counts over independent trajectories, never over the scan axis.
COHERENCE_FREQUENCY_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=[
        AxisSchema(name="scan", role=AxisRole.PARAMETER),
        AxisSchema(name="trajectory", role=AxisRole.REALIZATION),
        AxisSchema(name="channel", role=AxisRole.COMPONENT),
    ],
    variables=[
        VariableSchema(
            name="frequency",
            dtype="float64",
            value_domain="real",
            dims=("scan", "channel"),
            quantity=SDEQuantity.COHERENCE_FREQUENCY.value,
        )
    ],
    uncertainties=[
        UncertaintySchema(
            target="frequency",
            kind="sample_std",
            independent_unit="trajectory",
            covariance="real",
            scope="sampling",
        )
    ],
    attributes={"estimator": ""},
)
