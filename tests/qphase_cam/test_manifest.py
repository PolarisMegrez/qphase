"""Phase 10B: CAM resource manifest and planning task profiles.

These tests exercise the *planning* contract only. They never touch the active
``qphase_cam.engine.Engine.manifest``: the current engine keeps its existing
behaviour and Phase 10E performs the handover that makes the compiler enforce
the profiles below.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from qphase.core.compiler import WorkflowCompiler
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.project import ProjectContext
from qphase.core.protocols import EngineManifest
from qphase.core.registry import registry
from qphase.core.system_config import SystemConfig
from qphase.core.task_profile import (
    PluginRequirementSet,
    TaskProfileResolutionContext,
    resolve_plugin_requirements,
)
from qphase.resources import validate_manifest
from qphase_cam.config import CAM_TASKS, select_cam_task_profile
from qphase_cam.data import (
    CAM_ANNOTATION_DECLARATION,
    CAM_CANDIDATE_DECLARATION,
    CAM_CONTINUATION_DECLARATION,
    CAM_EVIDENCE_DECLARATION,
    CAM_STATE_DECLARATION,
    CAM_STATE_SCAN_DECLARATION,
    EVIDENCE_COUNT_FIELDS,
)
from qphase_cam.manifest import CAM_TASK_PROFILES, MANIFEST, cam_task_profile

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "packages" / "qphase_cam"
PACKAGE_SOURCE = PACKAGE_ROOT / "qphase_cam"


def _pyproject_entry_points() -> dict[str, str]:
    pyproject = tomllib.loads(
        (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    return dict(pyproject["project"]["entry-points"]["qphase"])


# --- resource manifest -----------------------------------------------------


def test_manifest_is_valid_and_points_at_the_installed_engine():
    assert validate_manifest(MANIFEST) == []
    assert MANIFEST.resource_id == "cam"
    entry_points = _pyproject_entry_points()
    assert entry_points[MANIFEST.engine.entry_point] == MANIFEST.engine.target
    assert MANIFEST.engine.entry_point == "engine.cam"


def test_version_and_core_range_agree_across_declarations():
    """pyproject, ``__version__`` and the manifest declare the same release."""
    import qphase_cam

    pyproject = tomllib.loads(
        (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    version = pyproject["project"]["version"]
    assert version == "2.0.0"
    assert qphase_cam.__version__ == version
    assert MANIFEST.package_version == version

    core_requirement = [
        dependency
        for dependency in pyproject["project"]["dependencies"]
        if dependency.startswith("qphase")
    ]
    assert core_requirement == ["qphase>=2.0.0,<3.0"], core_requirement
    assert MANIFEST.compatibility.qphase_core == ">=2.0.0,<3.0"
    # The installed metadata must agree too, or entry-point discovery is stale.
    from importlib.metadata import version as installed_version

    assert installed_version("qphase-cam") == version


def test_manifest_namespaces_match_the_declared_entry_points():
    """Every CAM-owned entry-point namespace is declared, plus overlay model."""
    entry_points = _pyproject_entry_points()
    namespaces = {name.split(".", 1)[0] for name in entry_points} - {
        "engine",
        "resource",
    }
    declared = set(MANIFEST.plugin_class_namespaces)
    assert namespaces <= declared
    # Models are project overlays: declared, never shipped as entry points.
    assert "model" in declared
    assert "model" not in namespaces
    assert declared - namespaces == {"model"}


def test_manifest_declares_only_the_implemented_data_products():
    """Every declared schema_ref must resolve to a real ProductSchema."""
    import importlib

    from qphase.data import ProductSchema

    declared = [product.name for product in MANIFEST.data_products]
    assert declared == [
        CAM_STATE_DECLARATION.name,
        CAM_STATE_SCAN_DECLARATION.name,
        CAM_CANDIDATE_DECLARATION.name,
        CAM_CONTINUATION_DECLARATION.name,
        CAM_ANNOTATION_DECLARATION.name,
        CAM_EVIDENCE_DECLARATION.name,
    ]
    for product in MANIFEST.data_products:
        module_name, _, attribute = product.schema_ref.partition(":")
        module = importlib.import_module(module_name)
        schema = getattr(module, attribute)
        assert isinstance(schema, ProductSchema), product.name
        assert schema.kind.value == product.kind
    # Phase 10C is complete: every declared product has a schema behind it.
    assert MANIFEST.materializers == []
    assert MANIFEST.backend_capabilities.streaming is False
    assert "numpy" in MANIFEST.backend_capabilities.backends


def test_manifest_imports_no_concrete_plugins():
    """Importing the manifest must not pull solvers, postprocessors or qpsgen."""
    tree = ast.parse((PACKAGE_SOURCE / "manifest.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    forbidden_prefixes = (
        "qphase_cam.solver",
        "qphase_cam.postprocessor",
        "qphase_cam.engine",
        "qphase_cam.core",
        "qpsgen",
    )
    offenders = sorted(name for name in imported if name.startswith(forbidden_prefixes))
    assert not offenders, offenders


def test_manifest_protocol_targets_resolve():
    import importlib

    for plugin_class in MANIFEST.plugin_classes:
        module_name, _, attr = plugin_class.protocol.partition(":")
        module = importlib.import_module(module_name)
        assert hasattr(module, attr), plugin_class.namespace


# --- planning task profiles ------------------------------------------------


def test_profiles_cover_the_frozen_task_list_once():
    ids = tuple(profile.id for profile in CAM_TASK_PROFILES)
    assert ids == CAM_TASKS
    assert len(set(ids)) == len(ids) == 6
    for task in CAM_TASKS:
        assert cam_task_profile(task).id == task
    with pytest.raises(KeyError, match="unknown CAM task profile"):
        cam_task_profile("purge")


@pytest.mark.parametrize("profile", CAM_TASK_PROFILES, ids=lambda p: p.id)
def test_profile_requirement_sets_are_disjoint_and_generic(profile):
    """Requirements stay disjoint and never name a concrete plugin."""
    requirements = profile.requirements
    assert isinstance(requirements, PluginRequirementSet)
    # ``backend`` is a core-owned namespace, not a CAM plugin class.
    cam_owned = set(MANIFEST.plugin_class_namespaces)
    for namespace in (*requirements.required, *requirements.optional):
        if namespace == "backend":
            continue
        assert namespace in cam_owned, namespace
    # Concrete plugin names would be lowercase identifiers of real classes,
    # which the profile layer must not name.
    for namespace in requirements.required:
        assert namespace != "cam_solver" or profile.id in {
            "solve",
            "scan",
            "search",
            "continue",
        }


def test_solver_tasks_require_a_solver_and_postprocessors_do_not():
    by_id = {profile.id: profile for profile in CAM_TASK_PROFILES}
    for task in ("solve", "scan", "search", "continue"):
        requirements = by_id[task].requirements
        assert {"backend", "model", "cam_solver"} <= set(requirements.required)
        assert "cam_postprocessor" in requirements.optional
    analyze = by_id["analyze"].requirements
    assert "cam_solver" in analyze.forbidden
    assert "cam_postprocessor" in analyze.required
    summarize = by_id["summarize_search"].requirements
    # The evidence assembler does no numerical work: a scientific postprocess
    # that needs a backend or a model belongs to ``analyze``.
    assert set(summarize.forbidden) == {"backend", "cam_solver", "model"}
    assert summarize.optional == ["cam_postprocessor"]
    assert not summarize.required


def test_typed_inputs_and_outputs_are_declared():
    by_id = {profile.id: profile for profile in CAM_TASK_PROFILES}
    assert [out.name for out in by_id["solve"].outputs] == ["state"]
    assert [out.name for out in by_id["scan"].outputs] == ["state_scan"]
    assert {out.name for out in by_id["search"].outputs} == {"candidates", "evidence"}
    assert [out.name for out in by_id["continue"].outputs] == ["continuation"]
    assert [out.name for out in by_id["analyze"].outputs] == ["annotations"]
    assert [slot.name for slot in by_id["analyze"].inputs] == ["input"]
    summarize_slots = {slot.name: slot for slot in by_id["summarize_search"].inputs}
    assert set(summarize_slots) == {"input", "evidence"}
    # Core accepts at most one *required* input slot per profile.
    required_slots = [
        slot
        for slot in by_id["summarize_search"].inputs
        if not slot.requirement.optional
    ]
    assert [slot.name for slot in required_slots] == ["input"]
    assert summarize_slots["evidence"].requirement.optional is True
    # The declarations behind those names are the planning products.
    assert by_id["solve"].outputs[0].declaration == CAM_STATE_DECLARATION
    assert by_id["scan"].outputs[0].declaration == CAM_STATE_SCAN_DECLARATION
    assert by_id["continue"].outputs[0].declaration == CAM_CONTINUATION_DECLARATION
    assert by_id["analyze"].outputs[0].declaration == CAM_ANNOTATION_DECLARATION
    assert by_id["search"].outputs[0].declaration == CAM_CANDIDATE_DECLARATION
    assert by_id["search"].outputs[1].declaration == CAM_EVIDENCE_DECLARATION


def test_declarations_carry_no_model_specific_vocabulary():
    allowed_quantities = {
        "cam_steady_state",
        "cam_steady_state_scan",
        "cam_bifurcation_candidate",
        "cam_search_evidence",
        "cam_continuation_branch",
        "cam_annotation",
    }
    for declaration in (
        CAM_STATE_DECLARATION,
        CAM_STATE_SCAN_DECLARATION,
        CAM_CANDIDATE_DECLARATION,
        CAM_EVIDENCE_DECLARATION,
        CAM_CONTINUATION_DECLARATION,
        CAM_ANNOTATION_DECLARATION,
    ):
        assert declaration.quantity in allowed_quantities
        for field in declaration.fields:
            for banned in ("kerr", "gamma", "mode", "omega", "eta2"):
                assert banned not in field, field


def test_declared_fields_are_the_frozen_planning_vocabulary():
    """The field lists are the vocabulary Phase 10C1 implements."""
    expected = {
        CAM_STATE_DECLARATION.name: [
            "parameters",
            "states",
            "residuals",
            "success",
            "valid_mask",
            "solution_count",
        ],
        CAM_STATE_SCAN_DECLARATION.name: [
            "parameters",
            "states",
            "residuals",
            "success",
            "valid_mask",
            "solution_count",
        ],
        CAM_CANDIDATE_DECLARATION.name: [
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
        CAM_EVIDENCE_DECLARATION.name: [
            "case_identity",
            "search_identity",
            "status",
            "physical_status",
            "search_design",
            "coverage",
            "path_audit",
            "execution_provenance",
            *EVIDENCE_COUNT_FIELDS,
            "prefilter_rejected",
            "rejected_by_reason",
            "accepted_candidates",
            "near_misses",
            "control_names",
            "source_artifacts",
        ],
        CAM_CONTINUATION_DECLARATION.name: [
            "parameters",
            "path_offsets",
            "path_identity",
            "continuation_coordinate",
            "states",
            "residuals",
            "success",
            "events",
        ],
        CAM_ANNOTATION_DECLARATION.name: [
            "annotation_kind",
            "target_identity",
            "target_kind",
            "status",
            "fields",
            "method_metadata",
        ],
    }
    declarations = {
        declaration.name: declaration
        for declaration in (
            CAM_STATE_DECLARATION,
            CAM_STATE_SCAN_DECLARATION,
            CAM_CANDIDATE_DECLARATION,
            CAM_EVIDENCE_DECLARATION,
            CAM_CONTINUATION_DECLARATION,
            CAM_ANNOTATION_DECLARATION,
        )
    }
    assert set(expected) == set(declarations)
    for name, fields in expected.items():
        assert declarations[name].fields == fields, name
    # Ragged products keep a table marker; dense state rows do not need one.
    assert CAM_CANDIDATE_DECLARATION.table == "row_table"
    assert CAM_EVIDENCE_DECLARATION.table == "row_table"
    assert CAM_STATE_SCAN_DECLARATION.table == "row_table"


def test_evidence_declaration_can_express_a_negative_result():
    """A search that accepts nothing still has full evidence: no state or
    signature column is mandatory.
    """
    fields = set(CAM_EVIDENCE_DECLARATION.fields)
    assert {
        "coverage",
        "status",
        "path_audit",
        "rejected_by_reason",
        "near_misses",
        "accepted_candidates",
    } <= fields
    assert set(EVIDENCE_COUNT_FIELDS) <= fields
    assert "states" not in fields
    assert "classification" not in fields


# --- selector --------------------------------------------------------------


def test_selector_returns_each_declared_task():
    for task in CAM_TASKS:
        config = {"engine": {"cam": {"task": task}}}
        assert select_cam_task_profile(config) == task


def test_selector_rejects_missing_or_malformed_task():
    with pytest.raises(ValueError, match="explicit task"):
        select_cam_task_profile({})
    with pytest.raises(ValueError, match="explicit task"):
        select_cam_task_profile({"engine": "cam"})
    with pytest.raises(ValueError, match="required"):
        select_cam_task_profile({"engine": {"cam": {}}})
    with pytest.raises(ValueError, match="must be a string"):
        select_cam_task_profile({"engine": {"cam": {"task": 3}}})
    with pytest.raises(ValueError, match="unknown CAM task"):
        select_cam_task_profile({"engine": {"cam": {"task": "purge"}}})


def test_selector_reads_only_the_job_explicit_view():
    """Plugins declared by the job never change the selected task."""
    config = {
        "engine": {"cam": {"task": "analyze"}},
        "cam_postprocessor": {"jacobian_spectrum": {}},
        "plugins": {"cam_solver": {"steady_state": {}}},
    }
    assert select_cam_task_profile(config) == "analyze"


# --- planning-level compiler integration ----------------------------------


def test_profiles_are_not_attached_to_the_active_engine_manifest():
    """Phase 10B must not activate the profiles on the running engine."""
    from qphase_cam.engine import Engine

    assert Engine.manifest.task_profiles is None
    assert Engine.manifest.profile_selector is None


class _PlanningEngineConfig(SimpleNamespace):
    """Placeholder engine config model for the planning fixture."""

    @classmethod
    def model_validate(cls, value: Any) -> _PlanningEngineConfig:
        return cls(**value) if isinstance(value, dict) else value


class _PlanningEngine:
    """Engine fixture carrying the CAM planning manifest, nothing else.

    Mirrors the ``profiled_engine`` pattern of ``tests/qphase/test_compiler.py``:
    the compiler reads a manifest, so the planning profiles can be validated
    without instantiating the real engine or activating them on it.
    """

    name: ClassVar[str] = "cam"
    description: ClassVar[str] = "CAM planning-profile fixture (Phase 10B)"
    config_schema: ClassVar[type] = _PlanningEngineConfig
    manifest: ClassVar[EngineManifest] = EngineManifest(
        task_profiles="qphase_cam.manifest:CAM_TASK_PROFILES",
        profile_selector="qphase_cam.config:select_cam_task_profile",
    )

    def __init__(self, config: Any = None, plugins: dict | None = None, **kwargs: Any):
        del kwargs
        self.config = config
        self.plugins = plugins or {}


@pytest.fixture
def planning_engine():
    """Register the planning engine fixture under the real ``engine.cam`` name.

    The selector reads ``engine.cam.task``, so the fixture stands in for the
    engine under its own name; the previous registration is restored on exit.
    """
    from tests.plugins.dummy_plugin import DummyPlugin

    engine_table = registry._tables.setdefault("engine", {})
    previous = engine_table.get("cam")
    registry.register("engine", "cam", _PlanningEngine, overwrite=True)
    for namespace in ("backend", "model", "cam_solver", "cam_postprocessor"):
        registry.register(
            namespace=namespace, name="dummy", builder=DummyPlugin, overwrite=True
        )
    try:
        yield
    finally:
        if previous is None:
            registry._tables.get("engine", {}).pop("cam", None)
        else:
            engine_table["cam"] = previous
        for namespace in ("backend", "model", "cam_solver", "cam_postprocessor"):
            registry._tables.get(namespace, {}).pop("dummy", None)


def _workflow(*jobs: JobConfig) -> WorkflowSpec:
    return WorkflowSpec(
        schema_="qphase.workflow/2",
        id="cam-planning-test",
        title="CAM planning",
        jobs=list(jobs),
    )


def _job(task: str, **sections: Any) -> JobConfig:
    return JobConfig(name=task, engine={"cam": {"task": task}}, **sections)


def test_compiler_plans_all_six_tasks_without_instantiating_the_engine(
    tmp_path, planning_engine, monkeypatch
):
    project = ProjectContext.create(tmp_path / "project")
    compiler = WorkflowCompiler(project, SystemConfig())

    def _explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the compiler must plan without instantiating the engine")

    monkeypatch.setattr(_PlanningEngine, "__init__", _explode)

    solver_job = {
        "backend": {"dummy": {}},
        "model": {"dummy": {}},
        "cam_solver": {"dummy": {}},
    }
    jobs = [
        _job("solve", **solver_job),
        _job("scan", **solver_job),
        _job("search", **solver_job),
        _job("continue", **solver_job),
        _job(
            "analyze",
            input={"from": str(tmp_path), "mode": "dataset"},
            cam_postprocessor={"dummy": {}},
        ),
        _job(
            "summarize_search",
            input={"from": str(tmp_path), "mode": "dataset"},
            cam_postprocessor={"dummy": {}},
        ),
    ]
    compiled = compiler.compile(_workflow(*jobs))

    for job in jobs:
        planned = compiled.job(job.name)
        profile = planned.task_profile
        assert profile is not None
        assert profile.profile_id == job.name
        assert isinstance(profile.requirements, PluginRequirementSet)
        expected = cam_task_profile(job.name).requirements
        assert profile.requirements == expected


def test_compiler_rejects_a_solver_task_without_a_solver(tmp_path, planning_engine):
    project = ProjectContext.create(tmp_path / "project")
    compiler = WorkflowCompiler(project, SystemConfig())

    with pytest.raises(Exception, match="missing required plugins"):
        compiler.compile(
            _workflow(_job("solve", backend={"dummy": {}}, model={"dummy": {}}))
        )


def test_resolver_returns_the_profile_defaults_without_a_resolver_call():
    for profile in CAM_TASK_PROFILES:
        assert profile.resolver is None
        resolved = resolve_plugin_requirements(profile, TaskProfileResolutionContext())
        assert resolved == profile.requirements
