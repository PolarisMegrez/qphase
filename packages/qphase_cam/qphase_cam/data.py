"""qphase_cam: Planning Data Products
---------------------------------------------------------
Planning-level declarations of the CAM product kinds. These are the
``qphase.data.ProductDeclaration`` records the task profiles reference: a name,
a data kind, a quantity and generic field names — no schema object, no
implementation class and no model-specific vocabulary.

They are *declarations only*: the concrete typed product schemas of Phase 10C
will be added next to them, and the resource manifest's ``data_products`` stays
empty until then so nothing claims a schema that does not exist yet.

Public API
----------
CAM_STATE_DECLARATION
    One solved steady state row.
CAM_STATE_SCAN_DECLARATION
    Steady states across a scan axis.
CAM_CANDIDATE_DECLARATION
    Bifurcation/search candidates.
CAM_EVIDENCE_DECLARATION
    Search-evidence rows for one candidate or scan point.
CAM_CONTINUATION_DECLARATION
    Continuation branches and their continuation axis.
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

#: One steady state: the state matrix plus the residual that accepted it.
CAM_STATE_DECLARATION = ProductDeclaration(
    name="state",
    kind=DataKind.STATISTICS,
    quantity="steady_state",
    fields=["state", "residual", "accepted"],
)

#: Steady states resolved across a scan axis (multistability).
CAM_STATE_SCAN_DECLARATION = ProductDeclaration(
    name="state_scan",
    kind=DataKind.STATISTICS,
    quantity="steady_state_scan",
    fields=["state", "residual", "accepted"],
)

#: Candidates proposed by a search task (bifurcation search included).
CAM_CANDIDATE_DECLARATION = ProductDeclaration(
    name="candidates",
    kind=DataKind.STATISTICS,
    quantity="bifurcation_candidate",
    fields=["parameter", "state", "residual", "signature"],
    table="row_table",
)

#: Evidence rows that justify a candidate or a scan point.
CAM_EVIDENCE_DECLARATION = ProductDeclaration(
    name="evidence",
    kind=DataKind.STATISTICS,
    quantity="search_evidence",
    fields=["parameter", "state", "residual", "signature", "source"],
    table="row_table",
)

#: Continuation output: branch points plus the continuation axis coordinates.
CAM_CONTINUATION_DECLARATION = ProductDeclaration(
    name="continuation",
    kind=DataKind.STATISTICS,
    quantity="continuation_branch",
    fields=["state", "parameter", "branch_position"],
)

#: Postprocessor annotations attached to already-solved states.
CAM_ANNOTATION_DECLARATION = ProductDeclaration(
    name="annotations",
    kind=DataKind.STATISTICS,
    quantity="annotation",
    fields=["label", "value"],
    table="row_table",
)
