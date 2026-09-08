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
references. The measurement basis is explicit: ``mode_labels`` lists the
modes the readouts act on and ``channel_measurement_kinds`` classifies
every channel as ``bare``, ``coherent``, ``incoherent`` or ``general``. A
channel measurement is a Hermitian matrix ``W`` acting on the mode vector
— the measured correlation is the quadratic form ``α†Wα`` — and exactly
one of two representations is populated per channel:

- ``channel_mode_weights`` holds the rank-1 readout vector ``l`` of
  ``W = l l†`` as ``[re, im]`` pairs flattened row-major over
  (channel × mode) — only ``bare``/``coherent`` channels carry it. A
  *bare* channel reads a single mode: exactly one unit weight with all
  others zero (``W = e_m e_m†``). A *coherent* channel carries any finite
  non-zero ``l`` of the coherent readout ``c = l†α`` (``W = l l†`` is
  rank-1).
- ``channel_mode_matrix`` holds the full Hermitian ``W`` as ``[re, im]``
  pairs flattened row-major over (channel × mode × mode) — required for
  ``incoherent``/``general`` channels. An *incoherent* channel carries a
  diagonal ``W`` with real non-negative entries: a trace is the incoherent
  sum ``W = I`` over the modes, which is generally NOT rank-1 and cannot
  be written as a coherent readout ``c = l†α``. A *general* channel
  carries any Hermitian ``W``.

Channel names and mode labels are the stable mapping and complex numbers
travel as real pairs, so there are no object or string arrays anywhere in
the product.

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
- path ambiguity: ``PATH_UNIQUE`` — a path's continuation is unique;
- estimand capability: deliberately *not* a flag — it is expressed per
  candidate and per capability by the ``<name>_valid`` masks of the optional
  columns, so a candidate can carry, say, a width but no curvature without
  faking capability with NaN sentinels.

Candidate status expresses only computation and center identifiability:
``PATH_UNIQUE`` belongs to path products, so candidate ``status_code``
values must stay within ``PEAK_CANDIDATE_STATUS_MASK`` and
``PeakCandidate.status`` defaults to bare ``COMPUTED`` (identifiability is
asserted explicitly once established). ``PEAK_PATH_PRODUCT`` rows layer the
computation and path-ambiguity axes: ``path_status_code`` must stay within
``PEAK_PATH_STATUS_MASK`` — ``CENTER_IDENTIFIABLE`` is meaningless on paths
because a path has no center of its own (its member candidate rows carry
that axis). The ``status_bits`` attribute of each product maps exactly the
flag names its table may use to bit values, so clean-process readers can
decode codes without importing this module; ``PEAK_STATUS_MASK`` remains the
union of all defined bits for unknown-producer detection.

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
    Union of all defined status bits; unknown bits fail loudly.
PEAK_CANDIDATE_STATUS_MASK, PEAK_PATH_STATUS_MASK
    Per-table status masks: candidate codes layer COMPUTED and
    CENTER_IDENTIFIABLE; path codes layer COMPUTED and PATH_UNIQUE.
CHANNEL_MEASUREMENT_KINDS
    Channel measurement kinds of the peak measurement basis.
PeakCandidate
    One row of the candidate table.
PeakPathResult
    Scan-axis path built over candidate rows.
CandidateTablePayload
    Exact keyword payload accepted by ``validate_candidate_table``.
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
from typing import Any, NotRequired, TypedDict

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

from ._status import coerce_status_flags
from .quantities import SDEQuantity

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


#: Union of all defined :class:`PeakStatus` bits. An unknown bit means a
#: producer newer than this contract (or corruption) and fails loudly.
PEAK_STATUS_MASK: int = (
    PeakStatus.COMPUTED | PeakStatus.CENTER_IDENTIFIABLE | PeakStatus.PATH_UNIQUE
).value

#: Bits a candidate row may layer: computation and center identifiability.
#: ``PATH_UNIQUE`` is the path-ambiguity axis and belongs to path products.
PEAK_CANDIDATE_STATUS_MASK: int = (
    PeakStatus.COMPUTED | PeakStatus.CENTER_IDENTIFIABLE
).value

#: Bits a path row may layer: computation and path ambiguity.
#: ``CENTER_IDENTIFIABLE`` is a candidate-level property — a path has no
#: center of its own — so path tables must not set it.
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
#: ``channel_names``/``mode_labels`` attributes with their stable labels and
#: may add the measurement basis (``channel_measurement_kinds`` plus
#: ``channel_mode_weights``/``channel_mode_matrix``); they may drop
#: optional capability columns they cannot provide (dropping an uncertainty
#: data variable requires dropping its uncertainty entry too — core enforces
#: the closure). The group index columns, the required candidate columns and
#: the offsets contract always hold.
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
        "mode_labels": ["mode_0"],
        "status_bits": {
            "computed": PeakStatus.COMPUTED.value,
            "center_identifiable": PeakStatus.CENTER_IDENTIFIABLE.value,
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
#: ``scan_position`` run along ``path_member``. ``path_status_code`` layers
#: the path-level :class:`PeakStatus` flags (``PATH_UNIQUE`` carries the
#: ambiguity axis; ``CENTER_IDENTIFIABLE`` is candidate-level and rejected).
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


def _status_code_array(name: str, codes: Any, *, mask: int, label: str) -> np.ndarray:
    array = _index_array(name, codes)
    # Bitwise, not a range check: the per-table masks are not necessarily
    # contiguous (the path mask 0b101 excludes CENTER_IDENTIFIABLE = 0b010).
    if len(array) and (
        int(array.min()) < 0 or bool(np.any(np.bitwise_and(array, ~mask)))
    ):
        raise ValueError(
            f"{name} values set bits outside the {label} PeakStatus mask (0b{mask:b})"
        )
    return array


def _stable_labels(field: str, values: Iterable[str], *, noun: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{field} must be a list of strings, not one string")
    result = tuple(values)
    if not result:
        raise ValueError(f"{field} must list at least one {noun}")
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{field} must hold non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must be unique")
    return result


#: Channel measurement kinds of the peak measurement basis. A channel
#: measurement is a Hermitian matrix ``W`` acting on the mode vector:
#: ``bare`` reads a single mode (``W = e_m e_m†``, rank-1), ``coherent``
#: is a coherent readout ``c = l†α`` (``W = l l†``, rank-1),
#: ``incoherent`` is an incoherent sum over the modes with diagonal real
#: non-negative ``W`` (a plain trace is ``W = I``, generally not rank-1),
#: and ``general`` is any Hermitian ``W``.
CHANNEL_MEASUREMENT_KINDS: tuple[str, ...] = (
    "bare",
    "coherent",
    "incoherent",
    "general",
)

_MEASUREMENT_KIND_SET = frozenset(CHANNEL_MEASUREMENT_KINDS)

#: Tolerances of the measurement-basis checks (unit weights, zero blocks,
#: Hermiticity and positive semidefiniteness): loose enough for float64
#: roundoff, tight enough to catch structural violations.
_BASIS_RTOL = 1e-6
_BASIS_ATOL = 1e-9


def _real_pair_array(
    name: str, values: Any, entry_count: int, entry_desc: str, context: str
) -> np.ndarray:
    """Validate a flattened real ``[re, im]`` pair array; return float64."""
    array = np.asarray(values)
    if array.shape != (entry_count, 2):
        raise ValueError(
            f"{name} must hold one [re, im] pair per {entry_desc} entry "
            f"flattened row-major, expected shape ({entry_count}, 2)"
            f"{context}, got {array.shape}"
        )
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must hold real [re, im] pairs")
    array = array.astype(np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _check_channel_mode_weights(
    weights: Any, channel_count: int, mode_count: int
) -> np.ndarray:
    return _real_pair_array(
        "channel_mode_weights",
        weights,
        channel_count * mode_count,
        "(channel, mode)",
        f" for {channel_count} channels x {mode_count} modes",
    )


def _check_channel_mode_matrix(
    matrix: Any, channel_count: int, mode_count: int
) -> np.ndarray:
    return _real_pair_array(
        "channel_mode_matrix",
        matrix,
        channel_count * mode_count * mode_count,
        "(channel, mode, mode)",
        f" for {channel_count} channels x {mode_count} modes",
    )


def _check_measurement_kinds(kinds: Any, channel_count: int) -> tuple[str, ...]:
    if isinstance(kinds, str):
        raise ValueError(
            "channel_measurement_kinds must be a list of strings, not one string"
        )
    result: tuple[str, ...] = tuple(kinds)
    if len(result) != channel_count:
        raise ValueError(
            "channel_measurement_kinds must hold exactly one entry per "
            f"channel ({channel_count}), got {len(result)}"
        )
    unknown = [kind for kind in result if kind not in _MEASUREMENT_KIND_SET]
    if unknown:
        raise ValueError(
            "channel_measurement_kinds entries must be one of "
            f"{sorted(_MEASUREMENT_KIND_SET)}, got {unknown}"
        )
    return result


def _check_measurement_basis(
    channel_names: tuple[str, ...],
    mode_count: int,
    kinds: tuple[str, ...],
    weights: np.ndarray | None,
    matrices: np.ndarray | None,
) -> None:
    """Enforce the per-channel representation/kind agreement.

    Exactly one representation is populated per channel: the rank-1
    readout vector ``l`` of ``W = l l†`` (``channel_mode_weights``) for
    ``bare``/``coherent`` channels, the Hermitian measurement matrix ``W``
    (``channel_mode_matrix``) for ``incoherent``/``general`` channels. A
    representation block of a channel of the other kind must be identically
    zero, so a channel can never populate both.
    """
    channel_count = len(channel_names)
    vectors = (
        None
        if weights is None
        else (weights[:, 0] + 1j * weights[:, 1]).reshape(channel_count, mode_count)
    )
    w_matrices = (
        None
        if matrices is None
        else (matrices[:, 0] + 1j * matrices[:, 1]).reshape(
            channel_count, mode_count, mode_count
        )
    )
    for index, (name, kind) in enumerate(zip(channel_names, kinds, strict=True)):
        vector = None if vectors is None else vectors[index]
        matrix = None if w_matrices is None else w_matrices[index]
        if kind in ("bare", "coherent"):
            if vector is None:
                raise ValueError(
                    f"{kind} channel {name!r} requires channel_mode_weights: "
                    "the rank-1 readout vector l of W = l l^dagger"
                )
            if matrix is not None and bool(np.any(matrix != 0.0)):
                raise ValueError(
                    f"channel {name!r} populates both channel_mode_weights and "
                    "channel_mode_matrix: exactly one representation per channel"
                )
            if kind == "bare":
                norms = np.abs(vector)
                unit = np.isclose(norms, 1.0, rtol=_BASIS_RTOL, atol=_BASIS_ATOL)
                zero = norms <= _BASIS_ATOL
                if int(np.count_nonzero(unit)) != 1 or not bool(np.all(unit | zero)):
                    raise ValueError(
                        f"bare channel {name!r} must carry exactly one unit "
                        "weight with all other weights zero (W = e_m e_m^dagger)"
                    )
            elif not bool(np.any(vector != 0.0)):
                raise ValueError(
                    f"coherent channel {name!r} must carry a non-zero readout "
                    "vector l (W = l l^dagger)"
                )
        else:
            if matrix is None:
                raise ValueError(
                    f"{kind} channel {name!r} requires channel_mode_matrix: "
                    "the Hermitian measurement matrix W acting on the mode "
                    "vector"
                )
            if vector is not None and bool(np.any(vector != 0.0)):
                raise ValueError(
                    f"channel {name!r} populates both channel_mode_weights and "
                    "channel_mode_matrix: exactly one representation per channel"
                )
            scale = float(np.max(np.abs(matrix)))
            if scale == 0.0:
                raise ValueError(
                    f"{kind} channel {name!r} must carry a non-zero "
                    "measurement matrix W"
                )
            hermiticity = float(np.max(np.abs(matrix - matrix.conj().T)))
            tolerance = _BASIS_RTOL * scale
            if hermiticity > tolerance:
                raise ValueError(
                    f"channel_mode_matrix of channel {name!r} must be "
                    "Hermitian (W = W^dagger)"
                )
            eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))
            eigen_tolerance = tolerance
            if float(np.min(eigenvalues)) < -eigen_tolerance:
                raise ValueError(
                    f"channel_mode_matrix of channel {name!r} must be "
                    "positive semidefinite (non-negative, W >= 0) for a "
                    "power measurement"
                )
            if kind == "incoherent":
                off_diagonal = matrix.copy()
                np.fill_diagonal(off_diagonal, 0.0)
                if bool(np.any(np.abs(off_diagonal) > tolerance)):
                    raise ValueError(
                        f"incoherent channel {name!r} must carry a diagonal "
                        "W: an incoherent sum over the modes (W = I for a "
                        "plain trace)"
                    )
                diagonal = np.diagonal(matrix)
                if bool(np.any(np.abs(diagonal.imag) > tolerance)) or bool(
                    np.any(diagonal.real < -tolerance)
                ):
                    raise ValueError(
                        f"incoherent channel {name!r} must carry real "
                        "non-negative diagonal entries"
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


def validate_candidate_table(
    candidate_offsets: Any,
    group_scan_index: Any,
    group_channel_index: Any,
    columns: Mapping[str, Any],
    *,
    scan_count: int,
    channel_names: Iterable[str],
    mode_labels: Iterable[str] | None = None,
    channel_measurement_kinds: Iterable[str] | None = None,
    channel_mode_weights: Any | None = None,
    channel_mode_matrix: Any | None = None,
    optional_capabilities: Iterable[str] = OPTIONAL_CAPABILITY_FIELDS,
) -> None:
    """Validate the group structure and columns of a candidate table.

    ``group_scan_index`` and ``group_channel_index`` hold one integer entry
    per (scan, channel) group; ``candidate_offsets`` must start at 0, be
    monotonically non-decreasing, hold exactly ``group_count + 1`` entries
    and end at the candidate count. Group indices must reference existing
    scan points (``[0, scan_count)``) and channels
    (``[0, len(channel_names))``), and each (scan, channel) pair owns at
    most one group — there is no implicit replicate dimension.

    ``channel_names`` is the mandatory stable index-to-label mapping
    (single-channel tables list one name). The measurement basis is
    optional as a whole, but when either representation is present,
    ``mode_labels`` and ``channel_measurement_kinds`` are required with it:
    a channel measurement is a Hermitian matrix ``W`` acting on the mode
    vector (the measured correlation is the quadratic form ``α†Wα``), and
    the kind of each channel — ``bare``, ``coherent``, ``incoherent`` or
    ``general`` — selects the representation it carries.

    Exactly one representation is populated per channel.
    ``channel_mode_weights`` holds the rank-1 readout vector ``l`` of
    ``W = l l†``: one finite ``[re, im]`` pair per (channel, mode) entry,
    flattened row-major, so exactly ``len(channel_names) * len(mode_labels)``
    pairs. It is allowed only for ``bare``/``coherent`` channels — a
    ``bare`` channel carries exactly one unit weight with all others zero
    (``W = e_m e_m†``), a ``coherent`` channel any finite non-zero ``l``
    of the coherent readout ``c = l†α``. ``channel_mode_matrix`` holds the
    full Hermitian ``W``: one finite ``[re, im]`` pair per
    (channel, mode, mode) entry, flattened row-major, so exactly
    ``len(channel_names) * len(mode_labels)**2`` pairs. It is required for
    ``incoherent``/``general`` channels — ``W`` must be Hermitian, and an
    ``incoherent`` channel additionally carries a diagonal ``W`` with real
    non-negative entries: a trace is the incoherent sum ``W = I`` over the
    modes, which is generally NOT rank-1 and cannot be written as a
    coherent readout. A channel populating both representations — or a
    representation contradicting its kind — is rejected.

    The required columns ``location``/``intensity``/``status_code`` must be
    present, and every column must hold exactly one entry per candidate.
    ``status_code`` values must stay within the candidate
    :class:`PeakStatus` mask (:data:`PEAK_CANDIDATE_STATUS_MASK`) —
    ``PATH_UNIQUE`` is a path-product bit and fails here. Each present
    optional capability must be paired with a ``<name>_valid`` mask column
    of the same length holding only 0/1 values.

    Raises
    ------
    ValueError
        On any contract violation.

    """
    offsets = _offsets_array("candidate_offsets", candidate_offsets)
    scan_index = _index_array("group_scan_index", group_scan_index)
    channel_index = _index_array("group_channel_index", group_channel_index)
    names = _stable_labels("channel_names", channel_names, noun="channel")
    modes = (
        None
        if mode_labels is None
        else _stable_labels("mode_labels", mode_labels, noun="mode")
    )
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
        pairs = set(zip(scan_index.tolist(), channel_index.tolist(), strict=True))
        if len(pairs) != group_count:
            raise ValueError(
                "duplicate (group_scan_index, group_channel_index) pairs: "
                "exactly one group per scan/channel pair"
            )
    if channel_mode_weights is None and channel_mode_matrix is None:
        if channel_measurement_kinds is not None:
            raise ValueError(
                "channel_measurement_kinds requires a measurement "
                "representation: channel_mode_weights and/or "
                "channel_mode_matrix"
            )
    else:
        if modes is None:
            raise ValueError(
                "the measurement basis (channel_mode_weights/"
                "channel_mode_matrix) requires mode_labels: the basis "
                "indexes the mode axis"
            )
        if channel_measurement_kinds is None:
            raise ValueError(
                "the measurement basis (channel_mode_weights/"
                "channel_mode_matrix) requires channel_measurement_kinds: "
                "the kind selects the representation each channel carries"
            )
        kinds = _check_measurement_kinds(channel_measurement_kinds, len(names))
        weights = (
            None
            if channel_mode_weights is None
            else _check_channel_mode_weights(
                channel_mode_weights, len(names), len(modes)
            )
        )
        matrices = (
            None
            if channel_mode_matrix is None
            else _check_channel_mode_matrix(channel_mode_matrix, len(names), len(modes))
        )
        _check_measurement_basis(names, len(modes), kinds, weights, matrices)
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
    _status_code_array(
        "status_code",
        columns["status_code"],
        mask=PEAK_CANDIDATE_STATUS_MASK,
        label="candidate",
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
    candidate. ``path_status_code`` must hold one path-level
    :class:`PeakStatus` code per path, within
    :data:`PEAK_PATH_STATUS_MASK` — ``CENTER_IDENTIFIABLE`` is
    candidate-level and fails here. Paths never modify the candidate body.

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
    status = _status_code_array(
        "path_status_code",
        path_status_code,
        mask=PEAK_PATH_STATUS_MASK,
        label="path",
    )
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
