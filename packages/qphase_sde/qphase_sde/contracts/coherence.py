"""qphase_sde: Coherence Frequency Contracts (2.0)
---------------------------------------------------------
Freezes the unified coherence-frequency estimate returned by every
``coherence_frequency`` estimator (short-delay, band-limited, finite-delay,
pole model). Estimates are rows of a typed statistics table; the frequency
uncertainty payload, lag, bandwidth and status code are explicit schema
variables so that estimator changes are always visible in the schema and
provenance.

Status codes
------------
The int64 ``status_code`` column layers the bit flags of
:class:`CoherenceFrequencyStatus`, grounded in what the coherence
estimators actually produce:

- computation: ``COMPUTED`` — the row payload was actually computed; a
  cleared bit marks a failed estimation (the short-delay and band-limited
  estimators report ``status="failed"`` rows with NaN payloads);
- resolution uniqueness: ``UNIQUE`` — one frequency platform or lag window
  supports the estimate; the band-limited estimator clears it on
  ``ambiguous_multiband`` and the short-delay estimator when no consistent
  nested window exists;
- window selection: ``INTERIOR_WINDOW`` — the selected lag window or
  bandwidth is not pinned to a search boundary; the short-delay estimator
  clears it on ``minimum_window`` and the band-limited estimator when the
  fit touched the maximum lag.

The ``status_bits`` attribute maps flag names to bit values so
clean-process readers can decode codes without importing this module.

Public API
----------
CoherenceFrequencyStatus
    Bit flags layered into the estimate status code.
COHERENCE_FREQUENCY_STATUS_MASK
    Union of all defined status bits; stored codes must stay within it.
CoherenceFrequencyEstimate
    One coherence-frequency estimate row.
COHERENCE_FREQUENCY_PRODUCT
    Open product schema template of the coherence-frequency product.
validate_coherence_frequency_table
    NumPy-level validator of a materialized coherence-frequency table.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntFlag
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ProductSchema,
    SamplingBasisSchema,
    UncertaintySchema,
    VariableSchema,
)

from .quantities import SDEQuantity

__all__ = [
    "COHERENCE_FREQUENCY_PRODUCT",
    "COHERENCE_FREQUENCY_STATUS_MASK",
    "CoherenceFrequencyEstimate",
    "CoherenceFrequencyStatus",
    "validate_coherence_frequency_table",
]


class CoherenceFrequencyStatus(IntFlag):
    """Bit flags layered into the coherence-frequency status code.

    The flags separate the computation, resolution-uniqueness and
    window-selection axes of a coherence-frequency estimate (see the module
    docstring for the estimator statuses each flag grounds).

    - ``COMPUTED``: the row payload was actually computed; a cleared bit
      marks a failed estimation, never a measurement.
    - ``UNIQUE``: one frequency platform or lag window supports the
      estimate; a cleared bit marks a degenerate or ambiguous resolution.
    - ``INTERIOR_WINDOW``: the selected lag window or bandwidth is not
      pinned to a search boundary.
    """

    NONE = 0
    COMPUTED = 1
    UNIQUE = 2
    INTERIOR_WINDOW = 4


#: Union of all defined :class:`CoherenceFrequencyStatus` bits. Stored
#: ``status_code`` values must stay within this mask; an unknown bit means a
#: producer newer than this contract (or corruption) and fails loudly.
COHERENCE_FREQUENCY_STATUS_MASK: int = (
    CoherenceFrequencyStatus.COMPUTED
    | CoherenceFrequencyStatus.UNIQUE
    | CoherenceFrequencyStatus.INTERIOR_WINDOW
).value


def _require_status_bits(value: Any) -> Any:
    """Reject raw integers with bits outside the defined status mask."""
    if isinstance(value, int) and not isinstance(value, bool):
        raw = int(value)
        if raw < 0 or raw & ~COHERENCE_FREQUENCY_STATUS_MASK:
            raise ValueError(
                f"status {raw} sets bits outside the defined "
                "CoherenceFrequencyStatus mask"
            )
    return value


class CoherenceFrequencyEstimate(BaseModel):
    """One coherence-frequency estimate with explicit diagnostics."""

    model_config = ConfigDict(extra="forbid")

    frequency: float
    bandwidth: float | None = None
    lag: float | None = Field(
        default=None, description="Delay/lag the estimate was computed at."
    )
    estimator: str = Field(description="Estimator identifier, e.g. 'band_limited'.")
    conditional_std: float | None = None
    frequency_sem: float | None = Field(
        default=None,
        description="Standard error over independent trajectories; maps to "
        "the frequency_sem column of COHERENCE_FREQUENCY_PRODUCT.",
    )
    confidence_lower: float | None = None
    confidence_upper: float | None = None
    independent_count: int | None = None
    status: CoherenceFrequencyStatus = Field(
        default=CoherenceFrequencyStatus.COMPUTED,
        description="Layered CoherenceFrequencyStatus flags. Defaults to "
        "bare COMPUTED, with UNIQUE/INTERIOR_WINDOW asserted explicitly once "
        "established.",
    )
    diagnostics: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-serializable lag/bandwidth/model diagnostics.",
    )

    @field_validator("status", mode="before")
    @classmethod
    def _check_status_bits(cls, value: Any) -> Any:
        return _require_status_bits(value)


#: Open schema template of the coherence-frequency product: one estimate per
#: (scan, channel) cell, with the lag the estimate was computed at, the
#: frequency half-bandwidth it was computed over and the layered status code
#: as explicit variables (NaN marks a value the estimator does not produce).
#: The trajectory-sampling uncertainty of the frequency counts over
#: independent trajectories, never over the scan axis, and its payload lives
#: in the ``frequency_sem`` variable.
COHERENCE_FREQUENCY_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=[
        AxisSchema(name="scan", role=AxisRole.PARAMETER),
        AxisSchema(name="channel", role=AxisRole.COMPONENT),
    ],
    sampling_bases=[SamplingBasisSchema(name="trajectory")],
    variables=[
        VariableSchema(
            name="frequency",
            dtype="float64",
            value_domain="real",
            dims=("scan", "channel"),
            quantity=SDEQuantity.COHERENCE_FREQUENCY.value,
        ),
        VariableSchema(
            name="frequency_sem",
            dtype="float64",
            value_domain="real",
            dims=("scan", "channel"),
        ),
        VariableSchema(
            name="lag",
            dtype="float64",
            value_domain="real",
            dims=("scan", "channel"),
            units="time",
        ),
        VariableSchema(
            name="bandwidth",
            dtype="float64",
            value_domain="real",
            dims=("scan", "channel"),
        ),
        VariableSchema(
            name="status_code",
            dtype="int64",
            value_domain="real",
            dims=("scan", "channel"),
        ),
    ],
    uncertainties=[
        UncertaintySchema(
            target="frequency",
            kind="sem",
            sampling_basis="trajectory",
            covariance="real",
            scope="sampling",
            data_variable="frequency_sem",
        )
    ],
    attributes={
        "estimator": "",
        "status_bits": {
            "computed": CoherenceFrequencyStatus.COMPUTED.value,
            "unique": CoherenceFrequencyStatus.UNIQUE.value,
            "interior_window": CoherenceFrequencyStatus.INTERIOR_WINDOW.value,
        },
    },
)


def validate_coherence_frequency_table(
    columns: Mapping[str, Any],
    *,
    scan_count: int,
    channel_count: int,
) -> None:
    """Validate a materialized coherence-frequency table.

    The required columns ``frequency``/``frequency_sem``/``lag``/
    ``bandwidth``/``status_code`` must be present, and every column must
    hold exactly one entry per (scan, channel) cell. ``lag`` and
    ``bandwidth`` must be nonnegative where finite (NaN marks a value the
    estimator does not produce). ``status_code`` must have an integer dtype
    and stay within the :class:`CoherenceFrequencyStatus` mask.

    Raises
    ------
    ValueError
        On any contract violation.

    """
    if scan_count < 0 or channel_count < 0:
        raise ValueError("scan_count and channel_count must be nonnegative")
    for required in ("frequency", "frequency_sem", "lag", "bandwidth", "status_code"):
        if required not in columns:
            raise ValueError(
                f"coherence-frequency table misses required column {required!r}"
            )
    for name, values in columns.items():
        column = np.asarray(values)
        if column.shape != (scan_count, channel_count):
            raise ValueError(
                f"column {name!r} must hold exactly one entry per "
                f"(scan, channel) cell ({scan_count}, {channel_count})"
            )
    status = np.asarray(columns["status_code"])
    if not np.issubdtype(status.dtype, np.integer):
        raise ValueError("status_code must have an integer dtype")
    if status.size and (
        int(status.min()) < 0
        or bool(np.any(np.bitwise_and(status, ~COHERENCE_FREQUENCY_STATUS_MASK)))
    ):
        raise ValueError(
            "status_code values set bits outside the defined "
            "CoherenceFrequencyStatus mask"
        )
    for name in ("lag", "bandwidth"):
        column = np.asarray(columns[name], dtype=np.float64)
        if np.any(np.isfinite(column) & (column < 0)):
            raise ValueError(f"{name} must be nonnegative where finite")
