"""qphase_cam: product declarations and schema templates
---------------------------------------------------------
Two declaration layers, both static:

* the six :class:`qphase.data.ProductDeclaration` records the task profiles
  reference — a name, a data kind, a quantity and the field vocabulary of one
  product. Phase 10C1 replaced the 10B placeholder field names with the fields
  the implemented schemas actually carry, so a declaration can no longer
  describe a payload that does not exist.
* the three :class:`qphase.data.ProductSchema` *templates* of Phase 10C1:
  :data:`CAM_STATE_PRODUCT` (one parameter point),
  :data:`CAM_STATE_SCAN_PRODUCT` (a scan of parameter points) and
  :data:`CAM_CANDIDATE_PRODUCT` (ragged bifurcation/search candidates).

A template keeps every axis **open** (``AxisSchema.size is None``): it is a
plan, not a payload. The builders in :mod:`qphase_cam.data.state` and
:mod:`qphase_cam.data.candidates` close the axes and materialize a dataset, so
a consumer can always tell "declared" from "written". The variable generators
live here and are shared with the builders, which is what keeps a frozen column
list from drifting away from what is actually materialized.

Phase 10C2 adds the continuation, annotation and evidence declarations here;
none of them is declared today, and none may reach the manifest before its
schema exists.
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
    "CAM_CANDIDATE_DECLARATION",
    "CAM_CANDIDATE_PRODUCT",
    "CAM_CONTINUATION_DECLARATION",
    "CAM_EVIDENCE_DECLARATION",
    "CAM_STATE_DECLARATION",
    "CAM_STATE_PRODUCT",
    "CAM_STATE_SCAN_DECLARATION",
    "CAM_STATE_SCAN_PRODUCT",
    "LEGACY_IDENTITY_SEMANTICS",
    "SOLUTION_AXIS_SEMANTICS",
]


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
#: search design, the coverage, the accounting of attempts and rejections, the
#: source artifacts it was derived from, the accepted candidates and the
#: rejection summary. No state or signature column is mandatory.
CAM_EVIDENCE_DECLARATION = ProductDeclaration(
    name="evidence",
    kind=DataKind.STATISTICS,
    quantity="cam_search_evidence",
    fields=[
        "search_design",
        "coverage",
        "accounting",
        "source_artifacts",
        "accepted_candidates",
        "rejection_summary",
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
        "path_identity",
        "continuation_coordinate",
        "states",
        "events",
    ],
    table="row_table",
)

#: Postprocessor annotations: what they are attached to, the annotation fields
#: and the method metadata that produced them.
CAM_ANNOTATION_DECLARATION = ProductDeclaration(
    name="annotations",
    kind=DataKind.STATISTICS,
    quantity="cam_annotation",
    fields=[
        "target_identity",
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

#: What an imported ``legacy:`` identity does and does not promise.
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
    ]


def _state_attributes(product: str, table: str) -> dict[str, Any]:
    return {
        "table": table,
        "product": product,
        "scan_axis": "scan",
        "solution_axis": "solution",
        "state_matrix_axes": ["state_row", "state_col"],
        "solution_semantics": SOLUTION_AXIS_SEMANTICS,
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
        "identity_semantics": LEGACY_IDENTITY_SEMANTICS,
        "ragged": True,
    }


#: Ragged bifurcation/search candidates. Axis sizes stay open here.
CAM_CANDIDATE_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=_candidate_axes(),
    variables=_candidate_variables(),
    attributes=_candidate_attributes(),
)
