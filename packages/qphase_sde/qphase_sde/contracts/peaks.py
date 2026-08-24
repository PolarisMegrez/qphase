"""qphase_sde: Peak Candidate Contracts (2.0)
---------------------------------------------------------
Freezes the unified peak-candidate representation returned by every
``peak_finder`` implementation (prominence, Lorentz, rational, scale-space).
Candidates are rows of a typed table — ragged per-scan-point candidate lists
are stored as typed columns plus row offsets, never as object arrays.

Three products share the candidate space:

- ``PEAK_PRODUCT``: the candidate table itself, ragged along ``candidate``
  with ``candidate_offsets[scan_offset]`` (length ``scan + 1``).
- ``PEAK_FIT_PRODUCT``: model fit parameters as typed numeric rows keyed by
  ``candidate_row``; the parameter-code → name mapping lives in attributes,
  so there are no string columns and no object payloads. Lorentz, rational
  and non-parametric finders share one candidate table without being forced
  to fake a common parameter set.
- ``PEAK_PATH_PRODUCT``: scan-axis paths referencing candidate rows. Paths
  never modify the candidate body, and a path is never claimed to be a
  physical cross-scan branch identity.

Optional capabilities (width, prominence, ...) are typed columns paired with
explicit ``<name>_valid`` mask columns — missing capability must never be
faked with NaN in a schema-complete-looking column. Uncertainty scopes are
explicit: ``conditional`` (error at the detected location), ``sampling``
(across independent realizations) and ``path_model_selection`` (path/model
ambiguity) are distinct scopes carried by typed uncertainty entries, not by
arbitrary fit payloads.

Public API
----------
UncertaintyScope
    Explicit uncertainty scopes.
PeakCandidate
    One row of the candidate table.
PeakPathResult
    Scan-axis path built over candidate rows.
PEAK_PRODUCT, PEAK_FIT_PRODUCT, PEAK_PATH_PRODUCT
    Open product schema templates of the peak products.
OPTIONAL_CAPABILITY_FIELDS
    Optional typed candidate columns requiring ``<name>_valid`` masks.
validate_candidate_table, validate_path_table, validate_confidence_bounds
    NumPy-level validators of materialized peak tables.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
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
    "OPTIONAL_CAPABILITY_FIELDS",
    "PEAK_FIT_PRODUCT",
    "PEAK_PATH_PRODUCT",
    "PEAK_PRODUCT",
    "PeakCandidate",
    "PeakPathResult",
    "UncertaintyScope",
    "validate_candidate_table",
    "validate_confidence_bounds",
    "validate_path_table",
]


class UncertaintyScope(str, Enum):
    """Explicit scope of a reported uncertainty."""

    CONDITIONAL = "conditional"
    SAMPLING = "sampling"
    PATH_MODEL_SELECTION = "path_model_selection"


class PeakCandidate(BaseModel):
    """One peak-candidate row of the unified peak table.

    ``conditional_location_std``/``confidence_*`` carry the conditional scope;
    ``sampling_location_std`` carries the sampling scope. Model-specific fit
    parameters never enter this row — they live in the separate fit-parameter
    table keyed by candidate row.
    """

    model_config = ConfigDict(extra="forbid")

    location: float
    intensity: float
    conditional_location_std: float | None = None
    sampling_location_std: float | None = None
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

    @model_validator(mode="after")
    def _check_candidate(self) -> PeakCandidate:
        if (
            self.confidence_lower is not None
            and self.confidence_upper is not None
            and self.confidence_lower > self.confidence_upper
        ):
            raise ValueError("confidence lower bound exceeds upper bound")
        for name in (
            "conditional_location_std",
            "sampling_location_std",
            "width",
            "prominence",
            "support",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative")
        return self


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

    @model_validator(mode="after")
    def _check_path(self) -> PeakPathResult:
        if len(self.candidate_rows) != len(self.scan_positions):
            raise ValueError(
                "candidate_rows and scan_positions must have equal length"
            )
        if any(row < 0 for row in self.candidate_rows):
            raise ValueError("candidate_rows must be nonnegative")
        if any(position < 0 for position in self.scan_positions):
            raise ValueError("scan_positions must be nonnegative")
        return self


#: Optional typed candidate columns. Each present capability must be paired
#: with an explicit ``<name>_valid`` int mask column of the same length.
OPTIONAL_CAPABILITY_FIELDS: tuple[str, ...] = (
    "width",
    "prominence",
    "curvature",
    "support",
    "quality",
    "conditional_location_std",
    "sampling_location_std",
    "confidence_lower",
    "confidence_upper",
)


def _optional_variables() -> list[VariableSchema]:
    variables: list[VariableSchema] = []
    for name in OPTIONAL_CAPABILITY_FIELDS:
        variables.append(
            VariableSchema(
                name=name,
                dtype="float64",
                value_domain="real",
                dims=("candidate",),
            )
        )
        variables.append(
            VariableSchema(
                name=f"{name}_valid",
                dtype="int8",
                value_domain="real",
                dims=("candidate",),
            )
        )
    return variables


#: Open schema template of the peak candidate product: a typed ragged table.
#: ``scan_offset`` closes to ``scan + 1`` entries and ``candidate`` to the
#: terminal offset once candidates are counted. Concrete finders may drop
#: optional capability columns they cannot provide; required columns and the
#: offsets contract always hold.
PEAK_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=[
        AxisSchema(name="scan", role=AxisRole.PARAMETER),
        AxisSchema(name="scan_offset", role=AxisRole.INDEX),
        AxisSchema(name="candidate", role=AxisRole.INDEX),
    ],
    sampling_bases=[SamplingBasisSchema(name="trajectory")],
    variables=[
        VariableSchema(
            name="candidate_offsets",
            dtype="int64",
            value_domain="real",
            dims=("scan_offset",),
        ),
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
        VariableSchema(
            name="status_code",
            dtype="int64",
            value_domain="real",
            dims=("candidate",),
        ),
        *_optional_variables(),
    ],
    uncertainties=[
        UncertaintySchema(
            target="location",
            kind="sample_std",
            covariance="real",
            scope=UncertaintyScope.CONDITIONAL.value,
            data_variable="conditional_location_std",
        ),
        UncertaintySchema(
            target="location",
            kind="sample_std",
            covariance="real",
            scope=UncertaintyScope.SAMPLING.value,
            sampling_basis="trajectory",
            data_variable="sampling_location_std",
        ),
    ],
    attributes={
        "table": "peak_candidates",
        "ragged_axis": "candidate",
        "offsets_variable": "candidate_offsets",
        "scan_axis": "scan",
        "valid_mask_suffix": "_valid",
        "optional_capabilities": list(OPTIONAL_CAPABILITY_FIELDS),
    },
)

#: Open schema template of the peak fit-parameter product: typed numeric rows
#: keyed by ``candidate_row``. The parameter-code → name mapping is an
#: attribute (a list whose index is the code); there are no string columns
#: and no object payloads.
PEAK_FIT_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=[
        AxisSchema(name="fit_row", role=AxisRole.INDEX),
    ],
    variables=[
        VariableSchema(
            name="candidate_row",
            dtype="int64",
            value_domain="real",
            dims=("fit_row",),
        ),
        VariableSchema(
            name="parameter_code",
            dtype="int64",
            value_domain="real",
            dims=("fit_row",),
        ),
        VariableSchema(
            name="value",
            dtype="float64",
            value_domain="real",
            dims=("fit_row",),
            quantity=SDEQuantity.FIT_PARAMETERS.value,
        ),
    ],
    attributes={
        "table": "peak_fit_parameters",
        "foreign_key": "candidate_row",
        "parameter_names": [],
    },
)

#: Open schema template of the peak path product. ``path_offsets`` has one
#: entry per path plus the terminal offset; ``candidate_row`` and
#: ``scan_position`` run along ``path_member``.
PEAK_PATH_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=[
        AxisSchema(name="path", role=AxisRole.INDEX),
        AxisSchema(name="path_offset", role=AxisRole.INDEX),
        AxisSchema(name="path_member", role=AxisRole.INDEX),
    ],
    variables=[
        VariableSchema(
            name="path_offsets",
            dtype="int64",
            value_domain="real",
            dims=("path_offset",),
        ),
        VariableSchema(
            name="candidate_row",
            dtype="int64",
            value_domain="real",
            dims=("path_member",),
        ),
        VariableSchema(
            name="scan_position",
            dtype="int64",
            value_domain="real",
            dims=("path_member",),
        ),
        VariableSchema(
            name="path_status_code",
            dtype="int64",
            value_domain="real",
            dims=("path",),
        ),
        VariableSchema(
            name="path_quality",
            dtype="float64",
            value_domain="real",
            dims=("path",),
        ),
    ],
    attributes={
        "table": "peak_paths",
        "offsets_variable": "path_offsets",
        "foreign_key": "candidate_row",
        "uncertainty_scope": UncertaintyScope.PATH_MODEL_SELECTION.value,
    },
)


def _offsets_array(name: str, offsets: Any) -> np.ndarray:
    array = np.asarray(offsets)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype")
    if len(array) == 0 or array[0] != 0:
        raise ValueError(f"{name} must start at 0")
    if len(array) > 1 and np.any(np.diff(array) < 0):
        raise ValueError(f"{name} must be monotonically non-decreasing")
    return array


def validate_candidate_table(
    candidate_offsets: Any,
    scan_count: int,
    columns: Mapping[str, Any],
    optional_capabilities: Iterable[str] = OPTIONAL_CAPABILITY_FIELDS,
) -> None:
    """Validate the offsets and columns of a materialized candidate table.

    ``candidate_offsets`` must start at 0, be monotonically non-decreasing,
    hold exactly ``scan_count + 1`` entries and end at the candidate count.
    The required columns ``location``/``intensity``/``status_code`` must be
    present, and every column must hold exactly one entry per candidate. Each
    present optional capability must be paired with a ``<name>_valid`` mask
    column of the same length holding only 0/1 values.

    Raises
    ------
    ValueError
        On any contract violation.

    """
    offsets = _offsets_array("candidate_offsets", candidate_offsets)
    if len(offsets) != scan_count + 1:
        raise ValueError(
            f"candidate_offsets must hold scan_count + 1 "
            f"({scan_count + 1}) entries, got {len(offsets)}"
        )
    candidate_count = int(offsets[-1])

    for required in ("location", "intensity", "status_code"):
        if required not in columns:
            raise ValueError(
                f"candidate table misses required column {required!r}"
            )
    for name, values in columns.items():
        column = np.asarray(values)
        if column.ndim != 1 or len(column) != candidate_count:
            raise ValueError(
                f"column {name!r} must be one-dimensional with exactly one "
                f"entry per candidate ({candidate_count})"
            )
    for capability in optional_capabilities:
        if capability not in columns:
            continue
        mask_name = f"{capability}_valid"
        if mask_name not in columns:
            raise ValueError(
                f"optional capability {capability!r} requires a "
                f"{mask_name!r} mask column; NaN is not a valid mask"
            )
        mask = np.asarray(columns[mask_name])
        if not np.issubdtype(mask.dtype, np.integer) or not set(
            np.unique(mask)
        ) <= {0, 1}:
            raise ValueError(
                f"mask column {mask_name!r} must hold only 0/1 values"
            )


def validate_path_table(
    path_offsets: Any,
    candidate_row: Any,
    scan_position: Any,
    candidate_count: int,
) -> None:
    """Validate a materialized peak path table.

    ``path_offsets`` must be legal offsets whose terminal value equals the
    number of path members; ``candidate_row`` and ``scan_position`` must have
    equal length, and every candidate row must reference an existing
    candidate. Paths never modify the candidate body.

    Raises
    ------
    ValueError
        On any contract violation.

    """
    offsets = _offsets_array("path_offsets", path_offsets)
    rows = np.asarray(candidate_row)
    positions = np.asarray(scan_position)
    if rows.ndim != 1 or positions.ndim != 1:
        raise ValueError("candidate_row and scan_position must be 1-D")
    if not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("candidate_row must have an integer dtype")
    if len(rows) != len(positions):
        raise ValueError(
            "candidate_row and scan_position must have equal length"
        )
    if int(offsets[-1]) != len(rows):
        raise ValueError(
            f"the terminal path offset ({int(offsets[-1])}) must equal the "
            f"path member count ({len(rows)})"
        )
    if len(rows) and (int(rows.min()) < 0 or int(rows.max()) >= candidate_count):
        raise ValueError(
            f"candidate_row references rows outside [0, {candidate_count})"
        )


def validate_confidence_bounds(
    confidence_lower: Any,
    confidence_upper: Any,
    valid_mask: Any | None = None,
) -> None:
    """Validate that confidence lower bounds never exceed upper bounds.

    Bounds are compared elementwise; when ``valid_mask`` is given, only
    masked-in entries are checked.

    Raises
    ------
    ValueError
        On shape mismatch or an inverted interval.

    """
    lower = np.asarray(confidence_lower, dtype=np.float64)
    upper = np.asarray(confidence_upper, dtype=np.float64)
    if lower.shape != upper.shape:
        raise ValueError(
            "confidence_lower and confidence_upper must have the same shape"
        )
    if valid_mask is not None:
        mask = np.asarray(valid_mask).astype(bool)
        if mask.shape != lower.shape:
            raise ValueError("valid_mask must match the bounds shape")
        lower = lower[mask]
        upper = upper[mask]
    if np.any(lower > upper):
        raise ValueError("confidence lower bound exceeds upper bound")
