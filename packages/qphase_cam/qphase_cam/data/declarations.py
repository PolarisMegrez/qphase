"""qphase_cam: product declarations and schema templates
---------------------------------------------------------
Two declaration layers, both static:

* the six :class:`qphase.data.ProductDeclaration` records the task profiles
  reference — a name, a data kind, a quantity and the field vocabulary of one
  product. Phase 10C1 (state, state-scan, candidates) and Phase 10C2a
  (continuation, annotations) replaced the earlier placeholder field names with
  the fields the implemented schemas actually carry, so a declaration can no
  longer describe a payload that does not exist.
* the :class:`qphase.data.ProductSchema` *templates* of Phases 10C1, 10C2a and
  10C2b: :data:`CAM_STATE_PRODUCT` (one parameter point),
  :data:`CAM_STATE_SCAN_PRODUCT` (a scan of parameter points),
  :data:`CAM_CANDIDATE_PRODUCT` (ragged bifurcation/search candidates),
  :data:`CAM_CONTINUATION_PRODUCT` (local continuation paths),
  :data:`CAM_ANNOTATION_PRODUCT` (postprocessor annotations) and
  :data:`CAM_EVIDENCE_PRODUCT` (search evidence per search case).

A template keeps every axis **open** (``AxisSchema.size is None``): it is a
plan, not a payload. The builders in :mod:`qphase_cam.data.state`,
:mod:`qphase_cam.data.candidates`, :mod:`qphase_cam.data.continuation`,
:mod:`qphase_cam.data.annotations` and :mod:`qphase_cam.data.evidence` close the
axes and materialize a dataset, so
a consumer can always tell "declared" from "written". The variable generators
live here and are shared with the builders, which is what keeps a frozen column
list from drifting away from what is actually materialized.

With Phase 10C2b the whole Phase 10C product vocabulary exists, so every
declaration here has a concrete schema behind it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ProductDeclaration,
    ProductSchema,
    VariableSchema,
)

__all__ = [
    "CAM_ANNOTATION_DECLARATION",
    "CAM_ANNOTATION_PRODUCT",
    "CAM_CANDIDATE_DECLARATION",
    "CAM_CANDIDATE_PRODUCT",
    "CANDIDATE_IDENTITY_SEMANTICS",
    "CAM_CONTINUATION_DECLARATION",
    "CAM_PREDICTION_DECLARATION",
    "CAM_PREDICTION_PRODUCT",
    "PREDICTION_STATUS_SEMANTICS",
    "CAM_CONTINUATION_PRODUCT",
    "CAM_EVIDENCE_DECLARATION",
    "CAM_EVIDENCE_PRODUCT",
    "CAM_STATE_DECLARATION",
    "CAM_STATE_PRODUCT",
    "CAM_STATE_SCAN_DECLARATION",
    "CAM_STATE_SCAN_PRODUCT",
    "EVIDENCE_COUNT_FIELDS",
    "EVIDENCE_STATUS_SEMANTICS",
    "GENERATION_TRIAL_SEMANTICS",
    "LEGACY_IDENTITY_SEMANTICS",
    "PATH_IDENTITY_SEMANTICS",
    "SOLUTION_AXIS_SEMANTICS",
    "SOLUTION_IDENTITY_SEMANTICS",
]


#: The candidate-start counters. These are the only counts that share one unit, so
#: these are the only ones that become columns of the evidence product.
EVIDENCE_COUNT_FIELDS: tuple[str, ...] = (
    "generated_candidate_count",
    "prefilter_pass_count",
    "prefilter_rejected_count",
    "refinement_start_count",
    "refinement_duplicate_count",
    "accepted_count",
    "rejected_count",
    "near_miss_saved",
    "near_miss_dropped",
)


# --- planning declarations -------------------------------------------------

#: Solve results. The ``parameters`` family is carried as ``parameter.<name>``
#: coordinate variables; ``success``/``valid_mask``/``solution_count`` describe
#: a fixed-capacity solution axis, which is why no ragged offset field appears.
CAM_STATE_DECLARATION = ProductDeclaration(
    name="state",
    kind=DataKind.STATISTICS,
    quantity="cam_steady_state",
    fields=[
        "parameters",
        "states",
        "residuals",
        "success",
        "valid_mask",
        "solution_count",
    ],
    table="row_table",
)

#: The same records across a scan axis, so one scan consumes one product.
CAM_STATE_SCAN_DECLARATION = ProductDeclaration(
    name="state_scan",
    kind=DataKind.STATISTICS,
    quantity="cam_steady_state_scan",
    fields=[
        "parameters",
        "states",
        "residuals",
        "success",
        "valid_mask",
        "solution_count",
    ],
    table="row_table",
)

#: Candidates proposed by a search task: what axis they were found on, their
#: identity, the states/controls that witness them, the classification and the
#: audit rows behind the decision.
CAM_CANDIDATE_DECLARATION = ProductDeclaration(
    name="candidates",
    kind=DataKind.STATISTICS,
    quantity="cam_bifurcation_candidate",
    fields=[
        "case_axes",
        "case_params",
        "candidate_offsets",
        "candidate_identity",
        "states",
        "state_vectors",
        "control_values",
        "control_names",
        "full_residual_norm",
        "search_residual_norm",
        "success",
        "status",
        "method",
        "verification_digits",
        "verification_status",
        "classification",
        "base_audit",
        "case_metadata",
    ],
    table="row_table",
)

#: Search evidence. It must be able to express a *negative* result — a search
#: that covered the design space and accepted no candidate — so it carries the
#: per-case identity and status, the search design and coverage, the
#: candidate-start counters, the per-path trial audit, the accepted candidate
#: references, the bounded near misses and the source artifacts it was derived
#: from. No state or signature column is mandatory.
CAM_EVIDENCE_DECLARATION = ProductDeclaration(
    name="evidence",
    kind=DataKind.STATISTICS,
    quantity="cam_search_evidence",
    fields=[
        "case_identity",
        "search_identity",
        "status",
        "physical_status",
        "search_design",
        "controls",
        "coverage",
        "path_audit",
        "execution_provenance",
        "notes",
        *EVIDENCE_COUNT_FIELDS,
        "prefilter_rejected",
        "rejected_by_reason",
        "accepted_candidates",
        "accepted_starts",
        "near_misses",
        "control_names",
        "source_artifacts",
    ],
    table="row_table",
)

#: One immutable prediction of a readout: what one approximation order implies for
#: the observable's intensity and response, with the identities, the validity
#: domain and the source artifacts it was derived from. It is a *product*, not an
#: annotation: a validation step may add parents and provenance to it, never edit
#: these fields.
CAM_PREDICTION_DECLARATION = ProductDeclaration(
    name="prediction",
    kind=DataKind.STATISTICS,
    quantity="cam_prediction",
    fields=[
        "prediction_id",
        "target_identity",
        "observable_name",
        "observable_kind",
        "charge_sector",
        "measurement_mapping",
        "approximation",
        "order",
        "status",
        "reason",
        "validity_domain",
        "expected_frequency_response_full",
        "expected_frequency_response_soft",
        "expected_frequency_response_regular",
        "allan_intensity_soft",
        "allan_intensity_full",
        "allan_intensity_reconstructed",
        "path_information_rate",
        "full_information_rate",
        "regular_response",
        "bordered_condition",
        "soft_rate",
        "sigma",
        "x_min",
        "x_max",
        "negative_physical_boundary",
        "positive_physical_boundary",
        "completeness",
        "source_artifacts",
        "method",
    ],
    table="row_table",
)

#: Continuation output: which path a row belongs to, its position along the
#: continuation coordinate, the states and the events met on the way.
CAM_CONTINUATION_DECLARATION = ProductDeclaration(
    name="continuation",
    kind=DataKind.STATISTICS,
    quantity="cam_continuation_branch",
    fields=[
        "parameters",
        "path_offsets",
        "path_identity",
        "continuation_coordinate",
        "states",
        "residuals",
        "success",
        "events",
    ],
    table="row_table",
)

#: Postprocessor annotations: what they are attached to, the annotation verdict,
#: the typed field columns and the method metadata that produced them.
CAM_ANNOTATION_DECLARATION = ProductDeclaration(
    name="annotations",
    kind=DataKind.STATISTICS,
    quantity="cam_annotation",
    fields=[
        "annotation_kind",
        "target_identity",
        "target_kind",
        "status",
        "fields",
        "method_metadata",
    ],
    table="row_table",
)


# --- schema templates ------------------------------------------------------

#: Why the solution axis is not a branch identity. Stated once, in the schema,
#: so a consumer reading only the artifact cannot misread the axis.
SOLUTION_AXIS_SEMANTICS = (
    "the solution axis is a fixed-capacity slot, not a branch identity: slots "
    "are ordered independently at every scan point and carry no cross-parameter "
    "continuity"
)

#: What a solution identity promises: it identifies one solved state at one
#: parameter point by content, never by the slot it happens to occupy.
SOLUTION_IDENTITY_SEMANTICS = (
    "identity is a content digest of the solved state and the parameter values of "
    "its scan point; an empty solution slot carries an empty string and the slot "
    "index is never part of the identity"
)

#: What a solver-owned candidate identity promises.
CANDIDATE_IDENTITY_SEMANTICS = (
    "identity is a solver-owned content digest of the candidate; it names the "
    "physical candidate, not a slot, and is therefore stable across the cases of "
    "one scan (the same candidate may appear once per case)"
)

#: What an imported ``legacy:`` identity does and does not promise (importer only).
LEGACY_IDENTITY_SEMANTICS = (
    "identity is the imported snapshot's content fingerprint plus the legacy "
    "case and candidate slots; it identifies this imported artifact, not a "
    "branch that survives a recomputation, another precision or another run"
)


def _state_axes() -> list[AxisSchema]:
    return [
        AxisSchema(name="scan", role=AxisRole.PARAMETER),
        AxisSchema(name="solution", role=AxisRole.INDEX),
        AxisSchema(name="state_row", role=AxisRole.INDEX),
        AxisSchema(name="state_col", role=AxisRole.INDEX),
    ]


def _state_variables() -> list[VariableSchema]:
    return [
        VariableSchema(
            name="states",
            dtype="complex128",
            value_domain="complex",
            dims=("scan", "solution", "state_row", "state_col"),
            quantity="cam_steady_state",
        ),
        VariableSchema(
            name="residuals",
            dtype="float64",
            value_domain="real",
            dims=("scan", "solution"),
            quantity="cam_steady_state_residual",
        ),
        VariableSchema(
            name="success",
            dtype="bool",
            value_domain="real",
            dims=("scan", "solution"),
        ),
        VariableSchema(
            name="valid_mask",
            dtype="bool",
            value_domain="real",
            dims=("scan", "solution"),
        ),
        VariableSchema(
            name="solution_count",
            dtype="int64",
            value_domain="real",
            dims=("scan",),
        ),
        VariableSchema(
            name="solution_identity",
            dtype="U1",
            value_domain="real",
            dims=("scan", "solution"),
            quantity="cam_solution_identity",
        ),
    ]


def _state_attributes(product: str, table: str) -> dict[str, Any]:
    return {
        "table": table,
        "product": product,
        "scan_axis": "scan",
        "solution_axis": "solution",
        "state_matrix_axes": ["state_row", "state_col"],
        "solution_semantics": SOLUTION_AXIS_SEMANTICS,
        "solution_identity_variable": "solution_identity",
        "solution_identity_semantics": SOLUTION_IDENTITY_SEMANTICS,
        "ragged": False,
    }


#: One parameter point. Axis sizes stay open here: the builder closes them.
CAM_STATE_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=_state_axes(),
    variables=_state_variables(),
    attributes=_state_attributes("state", "cam_state"),
)

#: The same product across a scan axis. The two differ only in their declared
#: product name and table, so a consumer never has to probe ``scan.size``.
CAM_STATE_SCAN_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=_state_axes(),
    variables=_state_variables(),
    attributes=_state_attributes("state_scan", "cam_state_scan"),
)


def _candidate_axes() -> list[AxisSchema]:
    return [
        AxisSchema(name="scan", role=AxisRole.PARAMETER),
        AxisSchema(name="group", role=AxisRole.INDEX),
        AxisSchema(name="group_offset", role=AxisRole.INDEX),
        AxisSchema(name="candidate", role=AxisRole.INDEX),
        AxisSchema(name="state_row", role=AxisRole.INDEX),
        AxisSchema(name="state_col", role=AxisRole.INDEX),
        AxisSchema(name="state_vector", role=AxisRole.INDEX),
        AxisSchema(name="control", role=AxisRole.INDEX),
    ]


def _candidate_variables(
    widths: Mapping[str, int] | None = None,
) -> list[VariableSchema]:
    """Return the candidate columns, with string widths from ``widths``.

    The template and every builder share this function so the frozen column
    list cannot drift from what a builder materializes; only the string width
    of a ``U`` column is data-dependent.
    """
    width_of = dict(widths or {})

    def text(name: str, dims: tuple[str, ...], quantity: str = "") -> VariableSchema:
        width = max(int(width_of.get(name, 1)), 1)
        return VariableSchema(
            name=name,
            dtype=f"U{width}",
            value_domain="real",
            dims=dims,
            quantity=quantity,
        )

    return [
        VariableSchema(
            name="group_scan_index",
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
        text("candidate_identity", ("candidate",), "cam_candidate_identity"),
        VariableSchema(
            name="states",
            dtype="complex128",
            value_domain="complex",
            dims=("candidate", "state_row", "state_col"),
            quantity="cam_candidate_state",
        ),
        VariableSchema(
            name="state_vectors",
            dtype="float64",
            value_domain="real",
            dims=("candidate", "state_vector"),
            quantity="cam_candidate_state_vector",
        ),
        VariableSchema(
            name="control_values",
            dtype="float64",
            value_domain="real",
            dims=("candidate", "control"),
            quantity="cam_candidate_control",
        ),
        text("control_names", ("control",)),
        VariableSchema(
            name="full_residual_norm",
            dtype="float64",
            value_domain="real",
            dims=("candidate",),
            quantity="cam_candidate_full_residual",
        ),
        VariableSchema(
            name="search_residual_norm",
            dtype="float64",
            value_domain="real",
            dims=("candidate",),
            quantity="cam_candidate_search_residual",
        ),
        VariableSchema(
            name="success",
            dtype="bool",
            value_domain="real",
            dims=("candidate",),
        ),
        text("status", ("candidate",)),
        text("method", ("candidate",)),
        VariableSchema(
            name="verification_digits",
            dtype="int64",
            value_domain="real",
            dims=("candidate",),
        ),
        text("verification_status", ("candidate",)),
        text("classification", ("candidate",)),
        text("base_audit", ("candidate",)),
        text("case_metadata", ("scan",)),
    ]


def _candidate_attributes() -> dict[str, Any]:
    return {
        "table": "cam_candidates",
        "product": "candidates",
        "scan_axis": "scan",
        "group_axis": "group",
        "group_offset_axis": "group_offset",
        "candidate_axis": "candidate",
        "ragged_axis": "candidate",
        "offsets_variable": "candidate_offsets",
        "identity_variable": "candidate_identity",
        "state_matrix_axes": ["state_row", "state_col"],
        "json_columns": ["classification", "base_audit", "case_metadata"],
        "identity_semantics": CANDIDATE_IDENTITY_SEMANTICS,
        "ragged": True,
    }


#: Ragged bifurcation/search candidates. Axis sizes stay open here.
CAM_CANDIDATE_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=_candidate_axes(),
    variables=_candidate_variables(),
    attributes=_candidate_attributes(),
)


#: Why a continuation path is a local, solver-owned object. Stated once, in the
#: schema, so a consumer reading only the artifact cannot treat two paths as one
#: global branch.
PATH_IDENTITY_SEMANTICS = (
    "a path identity is local to the solve that produced it and is supplied by "
    "the caller, never derived from a path slot; two paths are not claims about "
    "one global branch"
)


def _continuation_axes() -> list[AxisSchema]:
    return [
        AxisSchema(name="point", role=AxisRole.PARAMETER),
        AxisSchema(name="path", role=AxisRole.INDEX),
        AxisSchema(name="path_offset", role=AxisRole.INDEX),
        AxisSchema(name="state_row", role=AxisRole.INDEX),
        AxisSchema(name="state_col", role=AxisRole.INDEX),
    ]


def _continuation_variables() -> list[VariableSchema]:
    return [
        VariableSchema(
            name="path_identity",
            dtype="U1",
            value_domain="real",
            dims=("path",),
            quantity="cam_path_identity",
        ),
        VariableSchema(
            name="path_offsets",
            dtype="int64",
            value_domain="real",
            dims=("path_offset",),
        ),
        VariableSchema(
            name="continuation_coordinate",
            dtype="float64",
            value_domain="real",
            dims=("point",),
            quantity="cam_continuation_coordinate",
        ),
        VariableSchema(
            name="states",
            dtype="complex128",
            value_domain="complex",
            dims=("point", "state_row", "state_col"),
            quantity="cam_continuation_state",
        ),
        VariableSchema(
            name="residuals",
            dtype="float64",
            value_domain="real",
            dims=("point",),
            quantity="cam_continuation_residual",
        ),
        VariableSchema(
            name="success",
            dtype="bool",
            value_domain="real",
            dims=("point",),
        ),
        VariableSchema(
            name="events",
            dtype="U1",
            value_domain="real",
            dims=("point",),
        ),
    ]


def _continuation_attributes() -> dict[str, Any]:
    return {
        "table": "cam_continuation",
        "product": "continuation",
        "point_axis": "point",
        "path_axis": "path",
        "path_offset_axis": "path_offset",
        "ragged_axis": "point",
        "offsets_variable": "path_offsets",
        "identity_variable": "path_identity",
        "state_matrix_axes": ["state_row", "state_col"],
        "json_columns": ["events"],
        "identity_semantics": PATH_IDENTITY_SEMANTICS,
        "ragged": True,
    }


#: Local continuation paths. Axis sizes stay open here; the builder closes them.
CAM_CONTINUATION_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=_continuation_axes(),
    variables=_continuation_variables(),
    attributes=_continuation_attributes(),
)


def _annotation_axes() -> list[AxisSchema]:
    return [AxisSchema(name="annotation", role=AxisRole.INDEX)]


def _annotation_variables() -> list[VariableSchema]:
    return [
        VariableSchema(
            name="target_identity",
            dtype="U1",
            value_domain="real",
            dims=("annotation",),
            quantity="cam_annotation_target",
        ),
        VariableSchema(
            name="status",
            dtype="U1",
            value_domain="real",
            dims=("annotation",),
            quantity="cam_outcome_status",
        ),
        VariableSchema(
            name="case_identity",
            dtype="U1",
            value_domain="real",
            dims=("annotation",),
            quantity="cam_case_identity",
        ),
        VariableSchema(
            name="method_metadata",
            dtype="U1",
            value_domain="real",
            dims=("annotation",),
        ),
    ]


def _annotation_attributes() -> dict[str, Any]:
    return {
        "table": "cam_annotations",
        "product": "annotations",
        "annotation_axis": "annotation",
        "identity_variable": "target_identity",
        "case_identity_variable": "case_identity",
        # A candidate identity names the physical candidate, so the same target
        # may legitimately be annotated once per scan case. The case scope keeps
        # those rows distinct: uniqueness is (case_identity, target_identity) when
        # a case is given and target_identity alone otherwise.
        "identity_scope": (
            "annotations are unique per (case_identity, target_identity); a "
            "target without a case scope must be unique in the product"
        ),
        "status_variable": "status",
        "json_columns": ["method_metadata"],
        # Typed annotation payload columns are declared per product: a field is a
        # real variable with its own dims and dtype, never a JSON stand-in.
        "field_variables": [],
        "ragged": False,
    }


#: Postprocessor annotations. The ``annotation`` axis is open and the typed field
#: columns are added by the builder, one variable per field.
CAM_ANNOTATION_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=_annotation_axes(),
    variables=_annotation_variables(),
    attributes=_annotation_attributes(),
)


#: Why an evidence product may contain a case that accepted nothing. Stated once,
#: in the schema, so a consumer reading only the artifact cannot read a negative
#: result as a missing one.
EVIDENCE_STATUS_SEMANTICS = (
    "a case that searched its design space and accepted nothing is 'not_found', "
    "a case that never searched is 'not_searched', and an unverified physicality "
    "is 'unchecked': none of them is a failure verdict"
)

#: Why generation trials are not a column. They are per-path quantities.
GENERATION_TRIAL_SEMANTICS = (
    "generation_trial_count and the workload counters are per-path quantities and "
    "are carried verbatim per path in the path_audit JSON column; they are never "
    "summed into one unitless total"
)


def _evidence_axes() -> list[AxisSchema]:
    return [
        AxisSchema(name="case", role=AxisRole.INDEX),
        AxisSchema(name="case_offset", role=AxisRole.INDEX),
        AxisSchema(name="accepted", role=AxisRole.INDEX),
        AxisSchema(name="accepted_start", role=AxisRole.INDEX),
        AxisSchema(name="near_miss", role=AxisRole.INDEX),
        AxisSchema(name="control", role=AxisRole.INDEX),
    ]


def _evidence_variables() -> list[VariableSchema]:
    case_counts = [
        VariableSchema(
            name=name,
            dtype="int64",
            value_domain="real",
            dims=("case",),
            quantity="cam_candidate_start_count",
        )
        for name in EVIDENCE_COUNT_FIELDS
    ]
    return [
        VariableSchema(
            name="case_identity",
            dtype="U1",
            value_domain="real",
            dims=("case",),
            quantity="cam_case_identity",
        ),
        VariableSchema(
            name="search_identity",
            dtype="U1",
            value_domain="real",
            dims=("case",),
            quantity="cam_search_identity",
        ),
        VariableSchema(
            name="status",
            dtype="U1",
            value_domain="real",
            dims=("case",),
            quantity="cam_outcome_status",
        ),
        VariableSchema(
            name="physical_status",
            dtype="U1",
            value_domain="real",
            dims=("case",),
        ),
        VariableSchema(
            name="search_design",
            dtype="U1",
            value_domain="real",
            dims=("case",),
        ),
        # The control contract of one case. ``control_names`` is a typed axis, but
        # bounds and sampling are per-control values, so they travel as canonical
        # JSON per case: without this column a restored evidence product could not
        # rebuild the fragment it was derived from.
        VariableSchema(
            name="controls",
            dtype="U1",
            value_domain="real",
            dims=("case",),
        ),
        VariableSchema(
            name="notes",
            dtype="U1",
            value_domain="real",
            dims=("case",),
        ),
        VariableSchema(
            name="coverage",
            dtype="U1",
            value_domain="real",
            dims=("case",),
        ),
        VariableSchema(
            name="path_audit",
            dtype="U1",
            value_domain="real",
            dims=("case",),
        ),
        VariableSchema(
            name="execution_provenance",
            dtype="U1",
            value_domain="real",
            dims=("case",),
            quantity="cam_execution_provenance",
        ),
        VariableSchema(
            name="prefilter_rejected",
            dtype="U1",
            value_domain="real",
            dims=("case",),
        ),
        VariableSchema(
            name="rejected_by_reason",
            dtype="U1",
            value_domain="real",
            dims=("case",),
        ),
        *case_counts,
        VariableSchema(
            name="accepted_offsets",
            dtype="int64",
            value_domain="real",
            dims=("case_offset",),
        ),
        VariableSchema(
            name="accepted_start_offsets",
            dtype="int64",
            value_domain="real",
            dims=("case_offset",),
        ),
        VariableSchema(
            name="near_miss_offsets",
            dtype="int64",
            value_domain="real",
            dims=("case_offset",),
        ),
        VariableSchema(
            name="accepted_identity",
            dtype="U1",
            value_domain="real",
            dims=("accepted",),
            quantity="cam_candidate_identity",
        ),
        VariableSchema(
            name="accepted_start_identity",
            dtype="U1",
            value_domain="real",
            dims=("accepted_start",),
            quantity="cam_accepted_start_identity",
        ),
        VariableSchema(
            name="control_names",
            dtype="U1",
            value_domain="real",
            dims=("control",),
        ),
        VariableSchema(
            name="near_miss_path",
            dtype="U1",
            value_domain="real",
            dims=("near_miss",),
        ),
        VariableSchema(
            name="near_miss_reduction",
            dtype="U1",
            value_domain="real",
            dims=("near_miss",),
        ),
        VariableSchema(
            name="near_miss_seed_source",
            dtype="U1",
            value_domain="real",
            dims=("near_miss",),
        ),
        VariableSchema(
            name="near_miss_rejection_reasons",
            dtype="U1",
            value_domain="real",
            dims=("near_miss",),
        ),
        VariableSchema(
            name="near_miss_search_residual",
            dtype="float64",
            value_domain="real",
            dims=("near_miss",),
        ),
        VariableSchema(
            name="near_miss_full_residual",
            dtype="float64",
            value_domain="real",
            dims=("near_miss",),
        ),
        VariableSchema(
            name="near_miss_min_state_eigenvalue",
            dtype="float64",
            value_domain="real",
            dims=("near_miss",),
        ),
        VariableSchema(
            name="near_miss_control_values",
            dtype="float64",
            value_domain="real",
            dims=("near_miss", "control"),
        ),
    ]


def _evidence_attributes() -> dict[str, Any]:
    return {
        "table": "cam_search_evidence",
        "product": "evidence",
        "case_axis": "case",
        "case_offset_axis": "case_offset",
        "ragged_axes": ["accepted", "accepted_start", "near_miss"],
        "offsets_variables": [
            "accepted_offsets",
            "accepted_start_offsets",
            "near_miss_offsets",
        ],
        "accepted_identity_variable": "accepted_identity",
        "accepted_start_identity_variable": "accepted_start_identity",
        # An accepted start is a refined trial; a candidate is a deduplicated
        # row. They are two layers, and neither is a branch of its own.
        "identity_layers": (
            "accepted_identity references the final (deduplicated) candidate rows "
            "one to one; accepted_start_identity lists every accepted start and "
            "may repeat a candidate across several starts"
        ),
        "control_names_variable": "control_names",
        "controls_variable": "controls",
        "controls_semantics": (
            "controls carries the per-case control contract as canonical JSON "
            "{name: {bounds: [low, high], sampling: str}}; control_names is the "
            "typed axis, and bounds/sampling are per-control values that axis "
            "cannot express. A reader rebuilds one search fragment from these "
            "columns instead of guessing the window"
        ),
        "notes_variable": "notes",
        "notes_semantics": (
            "notes carries the fragment's own notes as canonical JSON per case; it "
            "is an audit annotation, never a count or a status"
        ),
        "count_fields": list(EVIDENCE_COUNT_FIELDS),
        "candidate_start_unit": (
            "every count column counts candidate starts; prefilter, refinement, "
            "duplicate and acceptance all split the same population"
        ),
        "json_columns": [
            "search_design",
            "controls",
            "notes",
            "coverage",
            "path_audit",
            "execution_provenance",
            "prefilter_rejected",
            "rejected_by_reason",
            "near_miss_rejection_reasons",
        ],
        # The environment that produced each case survives in the artifact, so a
        # restored evidence product can still say which solver, plugins, package
        # versions, audit revision, backend and dtype it came from.
        "execution_provenance_variable": "execution_provenance",
        "execution_provenance_fields": [
            "solver",
            "plugins",
            "distribution",
            "api_revision",
            "backend",
            "dtype",
        ],
        "status_semantics": EVIDENCE_STATUS_SEMANTICS,
        "generation_trial_semantics": GENERATION_TRIAL_SEMANTICS,
        "ragged": True,
    }


#: Search evidence: one row per search case plus the accepted identities and the
#: bounded near misses that belong to it. Axis sizes stay open here.
CAM_EVIDENCE_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=_evidence_axes(),
    variables=_evidence_variables(),
    attributes=_evidence_attributes(),
)


#: What a prediction status means.
PREDICTION_STATUS_SEMANTICS = (
    "one CAMOutcomeStatus per prediction; a failed approximation keeps its numbers "
    "as NaN and its reason, and never becomes a valid prediction"
)


def _prediction_axes() -> list[AxisSchema]:
    return [AxisSchema(name="prediction", role=AxisRole.INDEX)]


def _prediction_variables() -> list[VariableSchema]:
    text = [
        "prediction_id",
        "target_identity",
        "observable_name",
        "observable_kind",
        "charge_sector",
        "measurement_mapping",
        "approximation",
        "status",
        "reason",
        "completeness",
    ]
    numeric = [
        "order",
        "expected_frequency_response_full",
        "expected_frequency_response_soft",
        "expected_frequency_response_regular",
        "allan_intensity_soft",
        "allan_intensity_full",
        "allan_intensity_reconstructed",
        "path_information_rate",
        "full_information_rate",
        "regular_response",
        "bordered_condition",
        "soft_rate",
        "sigma",
        "x_min",
        "x_max",
    ]
    json_columns = ["validity_domain", "source_artifacts", "method"]
    variables = [
        VariableSchema(name=name, dtype="U1", value_domain="real", dims=("prediction",))
        for name in text
    ]
    variables.extend(
        VariableSchema(
            name=name,
            dtype="int64" if name == "order" else "float64",
            value_domain="real",
            dims=("prediction",),
        )
        for name in numeric
    )
    variables.extend(
        VariableSchema(name=name, dtype="U1", value_domain="real", dims=("prediction",))
        for name in json_columns
    )
    variables.extend(
        VariableSchema(
            name=name,
            dtype="int8",
            value_domain="real",
            dims=("prediction",),
            quantity="cam_prediction_tristate",
        )
        for name in ("negative_physical_boundary", "positive_physical_boundary")
    )
    return variables


def _prediction_attributes() -> dict[str, Any]:
    return {
        "table": "cam_prediction",
        "product": "prediction",
        "prediction_axis": "prediction",
        "approximations": ["full_ou", "one_loop", "nonlinear_manifold", "soft_ou"],
        "approximation_semantics": (
            "soft_ou is the critical-mode-only Gaussian estimate, full_ou the full "
            "linear response, and nonlinear_manifold the reconstruction along the "
            "truncated slow manifold; none of them is a rigorous bound on the full "
            "nonlinear error"
        ),
        "identity_semantics": (
            "prediction_id is a content digest of the observable identity, the "
            "target identity, the approximation, the order, the validity domain, "
            "the source artifact ids and the numeric payload; it never depends on a "
            "slot position, a path or a job name"
        ),
        "immutability_semantics": (
            "a prediction is published as an artifact; a later validation adds "
            "parents and provenance and never edits these columns"
        ),
        "source_artifact_variable": "source_artifacts",
        "json_columns": ["validity_domain", "source_artifacts", "method"],
        "tristate_semantics": (
            "-1 means the check never ran, 0 means it failed, 1 means it passed"
        ),
        "completeness_values": ["complete", "intermediate"],
        "status_semantics": PREDICTION_STATUS_SEMANTICS,
        "field_units": {
            "allan_intensity_*": "readout variance per unit bandwidth",
            "expected_frequency_response_*": "readout per unit parameter",
            "full_information_rate": "response^2 / reconstructed intensity",
        },
    }


#: One row per prediction; axis sizes stay open here.
CAM_PREDICTION_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=_prediction_axes(),
    variables=_prediction_variables(),
    attributes=_prediction_attributes(),
)
