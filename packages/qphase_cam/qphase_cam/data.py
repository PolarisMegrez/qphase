"""qphase_cam: Planning Data Products
---------------------------------------------------------
Planning-level declarations of the CAM product kinds. These are the
``qphase.data.ProductDeclaration`` records the task profiles reference: a name,
a data kind, a quantity and generic field names — no schema object, no
implementation class and no model-specific vocabulary.

They are *declarations only*: the concrete typed product schemas of Phase 10C
will be added next to them, and the resource manifest's ``data_products`` stays
empty until then so nothing claims a schema that does not exist yet.

Quantity names are namespaced with ``cam_`` so they cannot collide with another
resource package's quantities. The field lists describe what each product must
carry; they are the vocabulary Phase 10C implements, not an implementation.

Public API
----------
CAM_STATE_DECLARATION
    One solve task result: the solved states of one parameter grid.
CAM_STATE_SCAN_DECLARATION
    Steady states resolved across a scan axis (multistability).
CAM_CANDIDATE_DECLARATION
    Bifurcation/search candidates with their identity and classification.
CAM_EVIDENCE_DECLARATION
    Search-evidence: design, coverage and accounting, including negative
    coverage without any accepted candidate.
CAM_CONTINUATION_DECLARATION
    Continuation branches, their coordinate and events.
CAM_ANNOTATION_DECLARATION
    Postprocessor annotations attached to solved states.
"""

from __future__ import annotations

from qphase.data import DataKind, ProductDeclaration

__all__ = [
    "CAM_ANNOTATION_DECLARATION",
    "CAM_CANDIDATE_DECLARATION",
    "CAM_CONTINUATION_DECLARATION",
    "CAM_EVIDENCE_DECLARATION",
    "CAM_STATE_DECLARATION",
    "CAM_STATE_SCAN_DECLARATION",
]

#: Solve results: the parameter points, their states, the residuals that
#: accepted them, per-point convergence flags and the ragged state offsets.
CAM_STATE_DECLARATION = ProductDeclaration(
    name="state",
    kind=DataKind.STATISTICS,
    quantity="cam_steady_state",
    fields=[
        "parameters",
        "states",
        "residuals",
        "convergence",
        "solution_offsets",
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
        "convergence",
        "solution_offsets",
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
        "candidate_offsets",
        "candidate_identity",
        "states",
        "controls",
        "classification",
        "base_audit",
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
