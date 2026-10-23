"""Peak candidate, fit-parameter and path declarations.

The product schemas below are the only place the peak product shapes are
defined; :mod:`validators` unpacks candidate, fit and path payloads against
them.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum, IntFlag
from typing import Any, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ProductSchema,
    SamplingBasisSchema,
    UncertaintySchema,
    VariableSchema,
)

from .._status import coerce_status_flags
from ..quantities import SDEQuantity


class UncertaintyScope(str, Enum):
    """Explicit scope of a reported uncertainty."""

    CONDITIONAL = "conditional"
    SAMPLING = "sampling"
    PATH_MODEL_SELECTION = "path_model_selection"

class PeakStatus(IntFlag):
    """Bit flags layered into the single candidate/path status code.

    The flags separate three of the four peak-status axes (computation,
    center identifiability, path ambiguity); the fourth axis — estimand
    capability — is expressed per candidate by the ``<name>_valid`` masks of
    the optional capability columns and is deliberately not a status bit.
    One enum serves both peak tables, but each table enforces its own mask
    (:data:`PEAK_CANDIDATE_STATUS_MASK` / :data:`PEAK_PATH_STATUS_MASK`).

    - ``COMPUTED``: the row payload was actually computed. A row without
      this bit is a placeholder or a failed computation and must not be
      consumed as a detection.
    - ``CENTER_IDENTIFIABLE``: the peak center is identifiable from the data
      (sufficient support, non-degenerate fit). Candidate rows only — a
      path has no center of its own.
    - ``PATH_UNIQUE``: no path ambiguity — the path's continuation is
      unique. Path rows only — path membership of a candidate is never
      claimed on the candidate table itself.
    """

    NONE = 0
    COMPUTED = 1
    CENTER_IDENTIFIABLE = 2
    PATH_UNIQUE = 4

PEAK_STATUS_MASK: int = (
    PeakStatus.COMPUTED | PeakStatus.CENTER_IDENTIFIABLE | PeakStatus.PATH_UNIQUE
).value

PEAK_CANDIDATE_STATUS_MASK: int = (
    PeakStatus.COMPUTED | PeakStatus.CENTER_IDENTIFIABLE
).value

PEAK_PATH_STATUS_MASK: int = (PeakStatus.COMPUTED | PeakStatus.PATH_UNIQUE).value

class PeakCandidate(BaseModel):
    """One peak-candidate row of the unified peak table.

    ``conditional_location_std``/``confidence_*`` carry the conditional scope;
    ``sampling_location_std`` carries the sampling scope. Model-specific fit
    parameters never enter this row — they live in the separate fit-parameter
    table keyed by candidate row. ``status`` layers the candidate-level
    :class:`PeakStatus` flags (computation and center identifiability only —
    ``PATH_UNIQUE`` belongs to path products); estimand capability is
    expressed by the ``<name>_valid`` table masks, not by this field.
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
    status: PeakStatus = Field(
        default=PeakStatus.COMPUTED,
        description="Layered candidate-level PeakStatus flags. Defaults to "
        "bare COMPUTED: the row was computed, while CENTER_IDENTIFIABLE is "
        "asserted explicitly once the center is established.",
    )

    @field_validator("status", mode="before")
    @classmethod
    def _check_status_bits(cls, value: Any) -> Any:
        return coerce_status_flags(
            value, enum=PeakStatus, mask=PEAK_CANDIDATE_STATUS_MASK, label="candidate"
        )

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
    """A scan-axis peak path referencing candidate rows of the peak table.

    ``status`` reuses :class:`PeakStatus` under the path mask
    (:data:`PEAK_PATH_STATUS_MASK`): ``PATH_UNIQUE`` carries the
    path-ambiguity axis, while ``CENTER_IDENTIFIABLE`` is candidate-level and
    rejected here. A path references candidate rows only — it is never a
    global physical cross-scan branch identity.
    """

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
    status: PeakStatus = Field(
        default=PeakStatus.COMPUTED,
        description="Layered path-level PeakStatus flags. Defaults to bare "
        "COMPUTED: the row was computed, while PATH_UNIQUE is asserted "
        "explicitly once continuation uniqueness is established.",
    )

    @field_validator("status", mode="before")
    @classmethod
    def _check_status_bits(cls, value: Any) -> Any:
        return coerce_status_flags(
            value, enum=PeakStatus, mask=PEAK_PATH_STATUS_MASK, label="path"
        )

    @model_validator(mode="after")
    def _check_path(self) -> PeakPathResult:
        if len(self.candidate_rows) != len(self.scan_positions):
            raise ValueError("candidate_rows and scan_positions must have equal length")
        if any(row < 0 for row in self.candidate_rows):
            raise ValueError("candidate_rows must be nonnegative")
        if any(position < 0 for position in self.scan_positions):
            raise ValueError("scan_positions must be nonnegative")
        return self

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

PEAK_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=[
        AxisSchema(name="scan", role=AxisRole.PARAMETER),
        AxisSchema(name="group", role=AxisRole.INDEX),
        AxisSchema(name="group_offset", role=AxisRole.INDEX),
        AxisSchema(name="candidate", role=AxisRole.INDEX),
    ],
    sampling_bases=[SamplingBasisSchema(name="trajectory")],
    variables=[
        VariableSchema(
            name="group_scan_index",
            dtype="int64",
            value_domain="real",
            dims=("group",),
        ),
        VariableSchema(
            name="group_channel_index",
            dtype="int64",
            value_domain="real",
            dims=("group",),
        ),
        VariableSchema(
            name="candidate_offsets",
            dtype="int64",
            value_domain="real",
            dims=("group_offset",),
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
            name="sample_index",
            dtype="int64",
            value_domain="real",
            dims=("candidate",),
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
        "group_axis": "group",
        "group_scan_variable": "group_scan_index",
        "group_channel_variable": "group_channel_index",
        "scan_axis": "scan",
        "channel_names": ["channel_0"],
        "mode_labels": ["mode_0"],
        "status_bits": {
            "computed": PeakStatus.COMPUTED.value,
            "center_identifiable": PeakStatus.CENTER_IDENTIFIABLE.value,
        },
        "valid_mask_suffix": "_valid",
        "optional_capabilities": list(OPTIONAL_CAPABILITY_FIELDS),
    },
)

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
        "status_bits": {
            "computed": PeakStatus.COMPUTED.value,
            "path_unique": PeakStatus.PATH_UNIQUE.value,
        },
    },
)

class CandidateTablePayload(TypedDict):
    """Exact keyword payload accepted by :func:`validate_candidate_table`.

    A unified peak finder returns exactly this mapping — unpacking it into
    the validator (``validate_candidate_table(**payload)``) must not raise.
    ``mode_labels``, ``channel_measurement_kinds`` and the channel
    representations (``channel_mode_weights``/``channel_mode_matrix``) are
    optional together: the basis indexes the mode axis, so it is
    meaningless without the labels, and the kind of each channel selects
    the representation it carries.
    """

    candidate_offsets: Any
    group_scan_index: Any
    group_channel_index: Any
    columns: Mapping[str, Any]
    scan_count: int
    channel_names: list[str]
    mode_labels: NotRequired[list[str]]
    channel_measurement_kinds: NotRequired[list[str]]
    channel_mode_weights: NotRequired[list[list[float]]]
    channel_mode_matrix: NotRequired[list[list[float]]]
