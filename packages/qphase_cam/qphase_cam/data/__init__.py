"""qphase_cam: typed data products (2.0)
---------------------------------------------------------
The CAM product vocabulary, in the order a consumer meets it:

* :mod:`qphase_cam.data.declarations` — the six ``ProductDeclaration`` records
  the task profiles reference and the three ``ProductSchema`` *templates* of
  Phase 10C1 (open axes: a template is a plan, not a payload).
* :mod:`qphase_cam.data.state` — :func:`build_state_product`: one parameter
  point or a scan of them as fixed-capacity ``solution`` slots.
* :mod:`qphase_cam.data.candidates` — :func:`build_candidate_product`: ragged
  bifurcation/search candidates with explicit identity and JSON text columns.
* :mod:`qphase_cam.data.bundle` — :func:`cam_bundle`: the core generic bundle
  plus the ``product_roles`` descriptor.
* :mod:`qphase_cam.data.legacy` — the one-off import of the pre-2.0 result
  objects and their trusted NPZ payloads.
* :mod:`qphase_cam.data.compare` — the export comparisons that prove an import
  carries the same science.

This module is the public surface: names are re-exported, never implemented
here, so ``qphase_cam.data`` stays a stable import path while the implementation
can grow module by module (Phase 10C2 adds continuation, annotation and
evidence next to the existing ones).

Schema rules the subpackage enforces
------------------------------------
* Payloads are typed NumPy arrays: no object arrays, no pickles, no ``dict``
  payload columns. JSON reaches a product as canonical *text* columns.
* The ``solution`` axis is a fixed-capacity slot, never a branch identity;
  cross-parameter continuity is the continuation product's job (Phase 10C2).
* ``candidate_identity`` is an explicit, unique ``U`` column. An imported
  legacy candidate gets ``legacy:<content fingerprint>:<case>:<candidate>``,
  which identifies the imported snapshot — not a branch that survives a
  recomputation. Phase 10D adds the solver-owned identity.
* Ragged candidates use the core ragged contract (``scan``/``group``/
  ``group_offset``/``candidate`` plus ``candidate_offsets``), so empty cases
  and a scan without candidates are representable.
* Legacy ``postprocess``/``branches`` content and any value with no JSON form
  are reported by name in the bundle provenance; nothing is dropped silently.

Public API
----------
CAM_STATE_PRODUCT
    Schema template of one solved parameter point.
CAM_STATE_SCAN_PRODUCT
    Schema template of steady states resolved across a scan axis.
CAM_CANDIDATE_PRODUCT
    Schema template of ragged bifurcation/search candidates.
CAM_BUNDLE_TYPE_ID
    Bundle descriptor type id of a CAM product collection.
build_state_product
    Build the state (or state-scan) dataset from fixed-capacity arrays.
build_candidate_product
    Build the ragged candidate dataset from flattened case arrays.
cam_bundle
    Group CAM products into a :class:`~qphase.data.GenericDataBundle`.
import_cam_result
    Import a :class:`~qphase_cam.result.CAMResult`.
import_bifurcation_result
    Import a :class:`~qphase_cam.bifurcation_result.CAMBifurcationResult`.
import_bifurcation_scan
    Import a :class:`~qphase_cam.bifurcation_result.CAMBifurcationScanResult`.
import_legacy_artifact
    Import a trusted pre-2.0 NPZ payload, dispatching on its keys.
compare_state_export
    Per-field comparison of a state product against its legacy result.
compare_candidates_export
    Per-field comparison of a candidate product against its legacy result.
"""

from __future__ import annotations

from .bundle import (
    CAM_BUNDLE_ADAPTER_ID,
    CAM_BUNDLE_TYPE_ID,
    CAM_PRODUCT_ROLES,
    cam_bundle,
)
from .candidates import build_candidate_product
from .compare import compare_candidates_export, compare_state_export
from .declarations import (
    CAM_ANNOTATION_DECLARATION,
    CAM_CANDIDATE_DECLARATION,
    CAM_CANDIDATE_PRODUCT,
    CAM_CONTINUATION_DECLARATION,
    CAM_EVIDENCE_DECLARATION,
    CAM_STATE_DECLARATION,
    CAM_STATE_PRODUCT,
    CAM_STATE_SCAN_DECLARATION,
    CAM_STATE_SCAN_PRODUCT,
    LEGACY_IDENTITY_SEMANTICS,
    SOLUTION_AXIS_SEMANTICS,
)
from .legacy import (
    import_bifurcation_result,
    import_bifurcation_scan,
    import_cam_result,
    import_legacy_artifact,
)
from .state import build_state_product

__all__ = [
    "CAM_ANNOTATION_DECLARATION",
    "CAM_BUNDLE_ADAPTER_ID",
    "CAM_BUNDLE_TYPE_ID",
    "CAM_CANDIDATE_DECLARATION",
    "CAM_CANDIDATE_PRODUCT",
    "CAM_CONTINUATION_DECLARATION",
    "CAM_EVIDENCE_DECLARATION",
    "CAM_PRODUCT_ROLES",
    "CAM_STATE_DECLARATION",
    "CAM_STATE_PRODUCT",
    "CAM_STATE_SCAN_DECLARATION",
    "CAM_STATE_SCAN_PRODUCT",
    "LEGACY_IDENTITY_SEMANTICS",
    "SOLUTION_AXIS_SEMANTICS",
    "build_candidate_product",
    "build_state_product",
    "cam_bundle",
    "compare_candidates_export",
    "compare_state_export",
    "import_bifurcation_result",
    "import_bifurcation_scan",
    "import_cam_result",
    "import_legacy_artifact",
]
