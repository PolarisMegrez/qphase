"""qphase_sde: Peak Candidate Contracts (2.0)
---------------------------------------------------------
Freezes the unified peak-candidate representation returned by every
``peak_finder`` implementation (prominence, Lorentz, rational, scale-space).
Candidates are rows of a typed table — ragged per-group candidate lists are
stored as typed columns plus row offsets, never as object arrays.

Grouped layout
--------------
Candidates group by scan point and channel/mode: one *group* per
(scan, channel) pair the finder ran on. ``PEAK_PRODUCT`` carries four axes —
``scan`` (parameter), ``group`` (index), ``group_offset`` (index) and
``candidate`` (index, ragged)::

    group_scan_index[group]          scan point of each group (indexes scan)
    group_channel_index[group]       channel of each group (indexes
                                     the channel_names attribute)
    candidate_offsets[group_offset]  prefix sums over groups: group g owns
                                     candidate rows [offsets[g], offsets[g+1]);
                                     holds exactly group_count + 1 entries
    location[candidate], intensity[candidate], status_code[candidate]
    <capability>[candidate] + <capability>_valid[candidate]

``group_offset`` is a dedicated index axis because a closed axis of size
``group_count`` cannot host the ``group_count + 1`` offsets entries — core
validates variable shapes against closed axis sizes, so the offsets axis
follows the same ``path``/``path_offset`` pattern as ``PEAK_PATH_PRODUCT``.

Channel labels never enter payload columns: the mandatory schema attribute
``channel_names`` (a list of strings — single-channel products list one
name) is the stable index-to-label mapping that ``group_channel_index``
references. When channel superposition weights are relevant, the optional
``channel_weights`` attribute holds one ``[re, im]`` pair per channel;
complex numbers travel as real pairs, so there are no object or string
arrays anywhere in the product.

Status codes
------------
The single int64 ``status_code`` column layers the orthogonal bit flags of
:class:`PeakStatus`, keeping the four peak-status axes distinct without
promoting domain fields into core job status:

- computation status: ``COMPUTED`` — the row payload was actually computed;
  a cleared bit marks a placeholder or failed computation, never a
  detection;
- center identifiability: ``CENTER_IDENTIFIABLE`` — the peak center is
  identifiable from the data (sufficient support, non-degenerate fit);
- path ambiguity: ``PATH_UNIQUE`` — for a path, its continuation is unique;
  for a candidate, its path membership is unambiguous (meaningful once path
  tracking consumed the table);
- estimand capability: deliberately *not* a flag — it is expressed per
  candidate and per capability by the ``<name>_valid`` masks of the optional
  columns, so a candidate can carry, say, a width but no curvature without
  faking capability with NaN sentinels.

``PEAK_PATH_PRODUCT.path_status_code`` reuses the same flag enum (there
``PATH_UNIQUE`` carries the path-ambiguity axis); the ``status_bits``
attribute of both products maps flag names to bit values so clean-process
readers can decode codes without importing this module.

Products
--------
Three products share the candidate space:

- ``PEAK_PRODUCT``: the candidate table itself, grouped as above.
- ``PEAK_FIT_PRODUCT``: model fit parameters as typed numeric rows keyed by
  ``candidate_row``; the parameter-code → name mapping lives in attributes,
  so there are no string columns and no object payloads. Lorentz, rational
  and non-parametric finders share one candidate table without being forced
  to fake a common parameter set.
- ``PEAK_PATH_PRODUCT``: scan-axis paths referencing candidate rows. Paths
  never modify the candidate body, and a path is never claimed to be a
  physical cross-scan branch identity.

Uncertainty scopes are explicit: ``conditional`` (error at the detected
location), ``sampling`` (across independent realizations) and
``path_model_selection`` (path/model ambiguity) are distinct scopes carried
by typed uncertainty entries, not by arbitrary fit payloads.

Public API
----------
UncertaintyScope
    Explicit uncertainty scopes.
PeakStatus
    Bit flags layered into the candidate/path status code.
PEAK_STATUS_MASK
    Union of all defined status bits; stored codes must stay within it.
PeakCandidate
    One row of the candidate table.
PeakPathResult
    Scan-axis path built over candidate rows.
PEAK_PRODUCT, PEAK_FIT_PRODUCT, PEAK_PATH_PRODUCT
    Open product schema templates of the peak products.
OPTIONAL_CAPABILITY_FIELDS
    Optional typed candidate columns requiring ``<name>_valid`` masks.
validate_candidate_table, validate_fit_table, validate_path_table,
validate_confidence_bounds
    NumPy-level validators of materialized peak tables.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum, IntFlag
from typing import Any

import numpy as np
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

from .quantities import SDEQuantity

__all__ = [
    "OPTIONAL_CAPABILITY_FIELDS",
    "PEAK_FIT_PRODUCT",
    "PEAK_PATH_PRODUCT",
    "PEAK_PRODUCT",
    "PEAK_STATUS_MASK",
    "PeakCandidate",
    "PeakPathResult",
    "PeakStatus",
    "UncertaintyScope",
    "validate_candidate_table",
    "validate_confidence_bounds",
    "validate_fit_table",
    "validate_path_table",
]


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

    - ``COMPUTED``: the row payload was actually computed. A row without
      this bit is a placeholder or a failed computation and must not be
      consumed as a detection.
    - ``CENTER_IDENTIFIABLE``: the peak center is identifiable from the data
      (sufficient support, non-degenerate fit).
    - ``PATH_UNIQUE``: no path ambiguity — for a path, its continuation is
      unique; for a candidate, its path membership is unambiguous.
    """

    NONE = 0
    COMPUTED = 1
    CENTER_IDENTIFIABLE = 2
    PATH_UNIQUE = 4


#: Union of all defined :class:`PeakStatus` bits. Stored ``status_code`` and
#: ``path_status_code`` values must stay within this mask; an unknown bit
#: means a producer newer than this contract (or corruption) and fails loudly.
PEAK_STATUS_MASK: int = (
    PeakStatus.COMPUTED | PeakStatus.CENTER_IDENTIFIABLE | PeakStatus.PATH_UNIQUE
).value

#: Default status of a cleanly computed, identifiable, unambiguous row — the
#: layered-flag equivalent of the historical ``"ok"`` literal.
_DEFAULT_STATUS = (
    PeakStatus.COMPUTED | PeakStatus.CENTER_IDENTIFIABLE | PeakStatus.PATH_UNIQUE
)


def _require_defined_status_bits(value: Any) -> Any:
    """Reject raw integers with bits outside the defined PeakStatus mask."""
    if isinstance(value, int) and not isinstance(value, bool):
        raw = int(value)
        if raw < 0 or raw & ~PEAK_STATUS_MASK:
            raise ValueError(
                f"status {raw} sets bits outside the defined PeakStatus mask"
            )
    return value


class PeakCandidate(BaseModel):
    """One peak-candidate row of the unified peak table.

    ``conditional_location_std``/``confidence_*`` carry the conditional scope;
    ``sampling_location_std`` carries the sampling scope. Model-specific fit
    parameters never enter this row — they live in the separate fit-parameter
    table keyed by candidate row. ``status`` layers the orthogonal
    :class:`PeakStatus` flags; estimand capability is expressed by the
    ``<name>_valid`` table masks, not by this field.
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
        default=_DEFAULT_STATUS,
        description="Layered PeakStatus flags of the candidate row.",
    )

    @field_validator("status", mode="before")
    @classmethod
    def _check_status_bits(cls, value: Any) -> Any:
        return _require_defined_status_bits(value)

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

    ``status`` reuses :class:`PeakStatus`; there ``PATH_UNIQUE`` carries the
    path-ambiguity axis. A path references candidate rows only — it is never
    a global physical cross-scan branch identity.
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
        default=_DEFAULT_STATUS,
        description="Layered PeakStatus flags of the path.",
    )

    @field_validator("status", mode="before")
    @classmethod
    def _check_status_bits(cls, value: Any) -> Any:
        return _require_defined_status_bits(value)

    @model_validator(mode="after")
    def _check_path(self) -> PeakPathResult:
        if len(self.candidate_rows) != len(self.scan_positions):
            raise ValueError("candidate_rows and scan_positions must have equal length")
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


#: Open schema template of the peak candidate product: a typed ragged table
#: grouped by scan point and channel. ``group_offset`` closes to
#: ``group_count + 1`` entries and ``candidate`` to the terminal offset once
#: candidates are counted. Concrete finders must replace the template
#: ``channel_names`` attribute with their stable channel labels and may drop
#: optional capability columns they cannot provide; the group index columns,
#: the required candidate columns and the offsets contract always hold.
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
        "status_bits": {
            "computed": PeakStatus.COMPUTED.value,
            "center_identifiable": PeakStatus.CENTER_IDENTIFIABLE.value,
            "path_unique": PeakStatus.PATH_UNIQUE.value,
        },
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
#: ``scan_position`` run along ``path_member``. ``path_status_code`` reuses
#: the :class:`PeakStatus` flags (``PATH_UNIQUE`` carries the ambiguity axis).
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
            "center_identifiable": PeakStatus.CENTER_IDENTIFIABLE.value,
            "path_unique": PeakStatus.PATH_UNIQUE.value,
        },
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


def _index_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype")
    return array


def _status_code_array(name: str, codes: Any) -> np.ndarray:
    array = _index_array(name, codes)
    if len(array) and (int(array.min()) < 0 or int(array.max()) > PEAK_STATUS_MASK):
        raise ValueError(
            f"{name} values must stay within the defined PeakStatus mask "
            f"(0..{PEAK_STATUS_MASK})"
        )
    return array


def _channel_names(names: Iterable[str]) -> tuple[str, ...]:
    if isinstance(names, str):
        raise ValueError("channel_names must be a list of strings, not one string")
    result = tuple(names)
    if not result:
        raise ValueError("channel_names must list at least one channel")
    if any(not isinstance(name, str) or not name for name in result):
        raise ValueError("channel_names must hold non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("channel_names must be unique")
    return result


def _check_channel_weights(weights: Any, channel_count: int) -> None:
    array = np.asarray(weights)
    if array.shape != (channel_count, 2):
        raise ValueError(
            f"channel_weights must hold one [re, im] pair per channel, "
            f"expected shape ({channel_count}, 2), got {array.shape}"
        )
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise ValueError("channel_weights must hold real [re, im] pairs")
    if not np.all(np.isfinite(array.astype(np.float64))):
        raise ValueError("channel_weights must be finite")


def validate_candidate_table(
    candidate_offsets: Any,
    group_scan_index: Any,
    group_channel_index: Any,
    columns: Mapping[str, Any],
    *,
    scan_count: int,
    channel_names: Iterable[str],
    channel_weights: Any | None = None,
    optional_capabilities: Iterable[str] = OPTIONAL_CAPABILITY_FIELDS,
) -> None:
    """Validate the group structure and columns of a candidate table.

    ``group_scan_index`` and ``group_channel_index`` hold one integer entry
    per (scan, channel) group; ``candidate_offsets`` must start at 0, be
    monotonically non-decreasing, hold exactly ``group_count + 1`` entries
    and end at the candidate count. Group indices must reference existing
    scan points (``[0, scan_count)``) and channels
    (``[0, len(channel_names))``); ``channel_names`` is the mandatory stable
    index-to-label mapping (single-channel tables list one name), and
    ``channel_weights``, when given, must hold one finite ``[re, im]`` pair
    per channel.

    The required columns ``location``/``intensity``/``status_code`` must be
    present, and every column must hold exactly one entry per candidate.
    ``status_code`` values must stay within the :class:`PeakStatus` mask.
    Each present optional capability must be paired with a ``<name>_valid``
    mask column of the same length holding only 0/1 values.

    Raises
    ------
    ValueError
        On any contract violation.

    """
    offsets = _offsets_array("candidate_offsets", candidate_offsets)
    scan_index = _index_array("group_scan_index", group_scan_index)
    channel_index = _index_array("group_channel_index", group_channel_index)
    names = _channel_names(channel_names)
    if scan_count < 0:
        raise ValueError("scan_count must be nonnegative")
    if len(scan_index) != len(channel_index):
        raise ValueError(
            "group_scan_index and group_channel_index must hold exactly one "
            "entry per group"
        )
    group_count = len(scan_index)
    if len(offsets) != group_count + 1:
        raise ValueError(
            f"candidate_offsets must hold group_count + 1 "
            f"({group_count + 1}) entries, got {len(offsets)}"
        )
    if group_count:
        if int(scan_index.min()) < 0 or int(scan_index.max()) >= scan_count:
            raise ValueError(
                f"group_scan_index references scan points outside [0, {scan_count})"
            )
        if int(channel_index.min()) < 0 or int(channel_index.max()) >= len(names):
            raise ValueError(
                f"group_channel_index references channels outside [0, {len(names)})"
            )
    if channel_weights is not None:
        _check_channel_weights(channel_weights, len(names))
    candidate_count = int(offsets[-1])

    for required in ("location", "intensity", "status_code"):
        if required not in columns:
            raise ValueError(f"candidate table misses required column {required!r}")
    for name, values in columns.items():
        column = np.asarray(values)
        if column.ndim != 1 or len(column) != candidate_count:
            raise ValueError(
                f"column {name!r} must be one-dimensional with exactly one "
                f"entry per candidate ({candidate_count})"
            )
    _status_code_array("status_code", columns["status_code"])
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
        if not np.issubdtype(mask.dtype, np.integer) or not set(np.unique(mask)) <= {
            0,
            1,
        }:
            raise ValueError(f"mask column {mask_name!r} must hold only 0/1 values")


def validate_fit_table(
    candidate_row: Any,
    parameter_code: Any,
    value: Any,
    *,
    candidate_count: int,
    parameter_names: Iterable[str] = (),
) -> None:
    """Validate a materialized peak fit-parameter table.

    The three columns must be one-dimensional with equal length; every
    ``candidate_row`` must reference an existing candidate, and every
    ``parameter_code`` must stay within ``parameter_names`` when the
    code-to-name mapping is given.

    Raises
    ------
    ValueError
        On any contract violation.

    """
    rows = _index_array("candidate_row", candidate_row)
    codes = _index_array("parameter_code", parameter_code)
    values = np.asarray(value)
    if values.ndim != 1:
        raise ValueError("value must be one-dimensional")
    if np.iscomplexobj(values) or not np.issubdtype(values.dtype, np.number):
        raise ValueError("value must be a real numeric column")
    if not (len(rows) == len(codes) == len(values)):
        raise ValueError(
            "candidate_row, parameter_code and value must have equal length"
        )
    if candidate_count < 0:
        raise ValueError("candidate_count must be nonnegative")
    if len(rows) and (int(rows.min()) < 0 or int(rows.max()) >= candidate_count):
        raise ValueError(
            f"candidate_row references rows outside [0, {candidate_count})"
        )
    names = tuple(parameter_names)
    if len(codes) and int(codes.min()) < 0:
        raise ValueError("parameter_code must be nonnegative")
    if names and len(codes) and int(codes.max()) >= len(names):
        raise ValueError(
            f"parameter_code references parameters outside [0, {len(names)})"
        )


def validate_path_table(
    path_offsets: Any,
    candidate_row: Any,
    scan_position: Any,
    candidate_count: int,
    *,
    path_status_code: Any,
) -> None:
    """Validate a materialized peak path table.

    ``path_offsets`` must be legal offsets whose terminal value equals the
    number of path members; ``candidate_row`` and ``scan_position`` must have
    equal length, and every candidate row must reference an existing
    candidate. ``path_status_code`` must hold one :class:`PeakStatus` code
    per path. Paths never modify the candidate body.

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
        raise ValueError("candidate_row and scan_position must have equal length")
    if int(offsets[-1]) != len(rows):
        raise ValueError(
            f"the terminal path offset ({int(offsets[-1])}) must equal the "
            f"path member count ({len(rows)})"
        )
    if len(rows) and (int(rows.min()) < 0 or int(rows.max()) >= candidate_count):
        raise ValueError(
            f"candidate_row references rows outside [0, {candidate_count})"
        )
    status = _status_code_array("path_status_code", path_status_code)
    if len(status) != len(offsets) - 1:
        raise ValueError(
            f"path_status_code must hold exactly one entry per path "
            f"({len(offsets) - 1}), got {len(status)}"
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
