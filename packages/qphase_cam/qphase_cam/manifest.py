"""qphase_cam: Resource Package Manifest and Task Profiles (2.0)
---------------------------------------------------------
Static declaration of the CAM resource package's managed assets and the six
planning task profiles. Two layers live here on purpose:

* :data:`MANIFEST` — the ``qphase.resource/1`` inventory: the unique
  ``engine.cam`` reference, plugin-class namespaces, backend capabilities,
  optional dependencies, the core compatibility range. It imports
  ``qphase.resources`` only; every engine/protocol target stays a dotted string
  so importing this module never pulls a solver, a postprocessor or qpsgen.
* :data:`CAM_TASK_PROFILES` — the six active task profiles (``solve``,
  ``scan``, ``search``, ``continue``, ``analyze``, ``summarize_search``), built
  from the core task-profile contract and the product declarations in
  :mod:`qphase_cam.data`.

Phase 10E attaches these profiles to ``qphase_cam.engine.Engine.manifest``
(``task_profiles`` + ``profile_selector``), so the compiler enforces exactly one
profile per job and every one of the six tasks has a runtime: four run a solver,
``analyze`` postprocesses an upstream bundle and ``summarize_search`` assembles
evidence from upstream artifacts.

Deliberate omissions:

* no second engine entry point; ``qphase_cam.evidence`` is a service contract
  (the search-audit fragment), not a plugin and not a placeholder;
* qpsgen is already a hard dependency in ``pyproject.toml`` (the QPSGen
  dynamics adapter imports it eagerly), so it is not declared as an *optional*
  dependency here; moving it into an extra is a packaging change for the phase
  that makes that adapter optional.

Version and core range are declared once each and must agree with
``pyproject.toml`` and ``qphase_cam.__version__``; the manifest tests compare
all four values so they cannot drift.

Public API
----------
MANIFEST
    The static resource package manifest (schema ``qphase.resource/1``).
CAM_TASK_PROFILES
    The six active task profiles, keyed by task id.
cam_task_profile
    Look up one active profile by id.
"""

from __future__ import annotations

from qphase.core.task_profile import (
    EngineTaskProfile,
    InputProductRequirement,
    OutputProductDeclaration,
    PluginRequirementSet,
)
from qphase.data import DataKind, ProductRequirement
from qphase.resources import (
    BackendCapabilities,
    CompatibilityRange,
    DataProductDeclaration,
    EngineDeclaration,
    OptionalDependencyDeclaration,
    PluginClassDeclaration,
    ResourcePackageManifest,
    ResourceProfile,
)

from .config import CAM_TASKS
from .data import (
    CAM_ANNOTATION_DECLARATION,
    CAM_CANDIDATE_DECLARATION,
    CAM_CONTINUATION_DECLARATION,
    CAM_EVIDENCE_DECLARATION,
    CAM_STATE_DECLARATION,
    CAM_STATE_SCAN_DECLARATION,
)

__all__ = ["CAM_TASK_PROFILES", "MANIFEST", "cam_task_profile"]

#: The CAM resource package manifest. Plugin-class namespaces mirror the
#: ``pyproject.toml`` entry-point namespaces, plus the overlay-only ``model``
#: namespace (concrete CAM models ship as project-local plugins). The
#: ``bifurcation_*`` namespaces are the search child-plugin classes.
MANIFEST = ResourcePackageManifest(
    resource_id="cam",
    package_version="2.0.0",
    engine=EngineDeclaration(
        entry_point="engine.cam",
        target="qphase_cam.engine:Engine",
    ),
    profiles=[ResourceProfile.BASE, ResourceProfile.COMPUTE],
    plugin_classes=[
        PluginClassDeclaration(
            namespace="model",
            protocol="qphase_cam.model:CAMModel",
            description="Coherent-amplitude matrix model protocol; concrete "
            "models are project overlays.",
        ),
        PluginClassDeclaration(
            namespace="cam_solver",
            protocol="qphase_cam.solver.base:CAMSolver",
            description="Solvers that produce fixed points or bifurcation "
            "candidates from a CAM model.",
        ),
        PluginClassDeclaration(
            namespace="cam_postprocessor",
            protocol="qphase_cam.postprocessor.base:CAMPostprocessor",
            description="Computations applied to already-solved CAM states "
            "(spectra, frequencies, physicality, validity).",
        ),
        PluginClassDeclaration(
            namespace="bifurcation_target",
            protocol="qphase_cam.solver.bifurcation_target:BifurcationTarget",
            description="What a bifurcation search is looking for.",
        ),
        PluginClassDeclaration(
            namespace="bifurcation_strategy",
            protocol="qphase_cam.solver.bifurcation_strategy:BifurcationStrategy",
            description="How a bifurcation search explores the scan.",
        ),
        PluginClassDeclaration(
            namespace="bifurcation_discovery",
            protocol="qphase_cam.solver.bifurcation_discovery:BifurcationDiscovery",
            description="Initial-guess discovery for a bifurcation search.",
        ),
        PluginClassDeclaration(
            namespace="bifurcation_classifier",
            protocol="qphase_cam.solver.bifurcation_classifier:BifurcationClassifier",
            description="Candidate classification (scaling signature).",
        ),
    ],
    # Phases 10C1, 10C2a and 10C2b implement the steady-state, candidate,
    # continuation, annotation and search-evidence schemas; every schema_ref
    # below resolves to a real ProductSchema.
    data_products=[
        DataProductDeclaration(
            name=CAM_STATE_DECLARATION.name,
            kind="statistics",
            schema_ref="qphase_cam.data:CAM_STATE_PRODUCT",
            description="One solved parameter point as fixed-capacity state "
            "slots with their residuals and solution count.",
        ),
        DataProductDeclaration(
            name=CAM_STATE_SCAN_DECLARATION.name,
            kind="statistics",
            schema_ref="qphase_cam.data:CAM_STATE_SCAN_PRODUCT",
            description="Steady states resolved across a flattened scan axis, "
            "with the original grid shape kept as an attribute.",
        ),
        DataProductDeclaration(
            name=CAM_CANDIDATE_DECLARATION.name,
            kind="statistics",
            schema_ref="qphase_cam.data:CAM_CANDIDATE_PRODUCT",
            description="Ragged bifurcation/search candidates with explicit "
            "identity, classification and audit JSON columns.",
        ),
        DataProductDeclaration(
            name=CAM_CONTINUATION_DECLARATION.name,
            kind="statistics",
            schema_ref="qphase_cam.data:CAM_CONTINUATION_PRODUCT",
            description="Local continuation paths with explicit path identity, "
            "a continuation coordinate and per-point events.",
        ),
        DataProductDeclaration(
            name=CAM_ANNOTATION_DECLARATION.name,
            kind="statistics",
            schema_ref="qphase_cam.data:CAM_ANNOTATION_PRODUCT",
            description="Postprocessor annotations: typed field columns plus one "
            "outcome status per target, so a negative result is a status.",
        ),
        DataProductDeclaration(
            name=CAM_EVIDENCE_DECLARATION.name,
            kind="statistics",
            schema_ref="qphase_cam.data:CAM_EVIDENCE_PRODUCT",
            description="Search evidence: candidate-start counters, coverage and "
            "bounded near misses per case, with explicit negative statuses.",
        ),
    ],
    materializers=[],
    backend_capabilities=BackendCapabilities(
        backends=["numpy", "cupy"],
        devices=["cpu", "cuda"],
        # CAM solves one parameter grid at a time: no streaming contract.
        streaming=False,
    ),
    optional_dependencies=[
        OptionalDependencyDeclaration(
            name="cupy-cuda12x",
            purpose="CUDA-resident linear algebra for batched CAM solves.",
            required_for=["backend"],
        ),
    ],
    compatibility=CompatibilityRange(qphase_core=">=2.0.0,<3.0"),
)


#: One steady state at one parameter point.
SOLVE_PROFILE = EngineTaskProfile(
    id="solve",
    requirements=PluginRequirementSet(
        required=["backend", "cam_solver", "model"],
        optional=["cam_postprocessor"],
    ),
    outputs=[
        OutputProductDeclaration(
            name=CAM_STATE_DECLARATION.name,
            declaration=CAM_STATE_DECLARATION,
        )
    ],
)

#: Steady states resolved across a scan axis.
SCAN_PROFILE = EngineTaskProfile(
    id="scan",
    requirements=PluginRequirementSet(
        required=["backend", "cam_solver", "model"],
        optional=["cam_postprocessor"],
    ),
    outputs=[
        OutputProductDeclaration(
            name=CAM_STATE_SCAN_DECLARATION.name,
            declaration=CAM_STATE_SCAN_DECLARATION,
        )
    ],
)

#: Bifurcation search: candidates plus the evidence that justifies them.
SEARCH_PROFILE = EngineTaskProfile(
    id="search",
    requirements=PluginRequirementSet(
        required=["backend", "cam_solver", "model"],
        optional=["cam_postprocessor"],
    ),
    outputs=[
        OutputProductDeclaration(
            name=CAM_CANDIDATE_DECLARATION.name,
            declaration=CAM_CANDIDATE_DECLARATION,
        ),
        OutputProductDeclaration(
            name=CAM_EVIDENCE_DECLARATION.name,
            declaration=CAM_EVIDENCE_DECLARATION,
        ),
    ],
)

#: Continuation along one parameter.
CONTINUE_PROFILE = EngineTaskProfile(
    id="continue",
    requirements=PluginRequirementSet(
        required=["backend", "cam_solver", "model"],
        optional=["cam_postprocessor"],
    ),
    outputs=[
        OutputProductDeclaration(
            name=CAM_CONTINUATION_DECLARATION.name,
            declaration=CAM_CONTINUATION_DECLARATION,
        )
    ],
)

#: Postprocess existing typed CAM products. No solver is involved; backend and
#: model are capabilities some postprocessors need (a model-aware one resolves
#: them through a profile resolver rather than by plugin name), so they stay
#: optional instead of forbidden.
ANALYZE_PROFILE = EngineTaskProfile(
    id="analyze",
    requirements=PluginRequirementSet(
        required=["cam_postprocessor"],
        optional=["backend", "model"],
        forbidden=["cam_solver"],
    ),
    inputs=[
        InputProductRequirement(
            name="input",
            requirement=ProductRequirement(
                name="input",
                kind=DataKind.STATISTICS,
            ),
        )
    ],
    outputs=[
        OutputProductDeclaration(
            name=CAM_ANNOTATION_DECLARATION.name,
            declaration=CAM_ANNOTATION_DECLARATION,
        )
    ],
)

#: Summarize search candidates/evidence. The evidence assembler does no
#: numerical work: every scientific postprocess that needs a backend or a model
#: belongs to ``analyze``, so only annotation providers stay optional here.
SUMMARIZE_SEARCH_PROFILE = EngineTaskProfile(
    id="summarize_search",
    requirements=PluginRequirementSet(
        optional=["cam_postprocessor"],
        forbidden=["backend", "cam_solver", "model"],
    ),
    inputs=[
        # A job input expresses exactly one external input slot, so the
        # candidate product is the required slot and the evidence rows are an
        # optional companion: `qphase.core.compiler` rejects a profile with two
        # required slots instead of pretending multi-slot input support.
        InputProductRequirement(
            name="input",
            requirement=ProductRequirement(
                name="input",
                kind=DataKind.STATISTICS,
                quantity=CAM_CANDIDATE_DECLARATION.quantity,
            ),
        ),
        InputProductRequirement(
            name="evidence",
            requirement=ProductRequirement(
                name="evidence",
                kind=DataKind.STATISTICS,
                quantity=CAM_EVIDENCE_DECLARATION.quantity,
                optional=True,
            ),
        ),
    ],
    outputs=[
        OutputProductDeclaration(
            name=CAM_EVIDENCE_DECLARATION.name,
            declaration=CAM_EVIDENCE_DECLARATION,
        )
    ],
)

#: The six planning task profiles, in :data:`qphase_cam.config.CAM_TASKS` order.
CAM_TASK_PROFILES: tuple[EngineTaskProfile, ...] = (
    SOLVE_PROFILE,
    SCAN_PROFILE,
    SEARCH_PROFILE,
    CONTINUE_PROFILE,
    ANALYZE_PROFILE,
    SUMMARIZE_SEARCH_PROFILE,
)


def cam_task_profile(task_id: str) -> EngineTaskProfile:
    """Return the planning task profile with the given id."""
    for profile in CAM_TASK_PROFILES:
        if profile.id == task_id:
            return profile
    raise KeyError(f"unknown CAM task profile {task_id!r}")


def _check_task_order() -> None:
    """Keep the profiles aligned with the frozen task list."""
    ids = tuple(profile.id for profile in CAM_TASK_PROFILES)
    if ids != CAM_TASKS:
        raise AssertionError(
            f"CAM task profiles {ids} must match CAM_TASKS {CAM_TASKS}"
        )


_check_task_order()
