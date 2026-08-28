"""Contract tests for tag syntax, tag policy and annotation documents."""

import pytest
from qphase.core.annotations import (
    ArtifactAnnotationDocument,
    OccurrenceAnnotations,
    SessionAnnotationDocument,
    TagAssignment,
)
from qphase.core.errors import QPhaseConfigError
from qphase.core.project import ProjectContext
from qphase.core.tags import (
    TagNamespacePolicy,
    TagPolicy,
    canonicalize_tag_list,
    canonicalize_tag_syntax,
    load_tag_policy,
    parse_tag,
)


def test_parse_tag_splits_namespace_and_path():
    assert parse_tag("task:bifurcation/search") == ("task", "bifurcation/search")
    assert parse_tag("Task:Bifurcation") == ("task", "bifurcation")


def test_canonicalize_tag_syntax_normalizes_case_and_whitespace():
    assert canonicalize_tag_syntax("  Task:Bifurcation/Search ") == (
        "task:bifurcation/search"
    )


@pytest.mark.parametrize(
    "value",
    [
        "no-namespace",
        "task:",
        "task:a//b",
        "task:/a",
        "task:a b",
        "1task:x",
        "task:UPPER ok",
    ],
)
def test_parse_tag_rejects_malformed_values(value):
    with pytest.raises(QPhaseConfigError):
        parse_tag(value)


def test_canonicalize_tag_list_rejects_duplicates():
    with pytest.raises(QPhaseConfigError, match="duplicate"):
        canonicalize_tag_list(["task:a", "Task:A"])


def _policy() -> TagPolicy:
    return TagPolicy.model_validate(
        {
            "schema": "qphase.tag-policy/1",
            "namespaces": {
                "task": {
                    "hierarchical": True,
                    "cardinality": "many",
                    "values": ["bifurcation", "diagnostics"],
                    "aliases": {"bif": "bifurcation"},
                },
                "stage": {
                    "hierarchical": False,
                    "cardinality": "one",
                    "values": ["q1", "q2"],
                    "objects": ["session", "workflow"],
                },
                "note": {"open": True},
            },
            "reserved_namespaces": ["system"],
        }
    )


def test_policy_applies_aliases_and_hierarchical_descendants():
    canonical = _policy().apply(["task:bif/search", "task:diagnostics"], "session")
    assert canonical == ["task:bifurcation/search", "task:diagnostics"]


def test_policy_rejects_unknown_namespace_and_value():
    policy = _policy()
    with pytest.raises(QPhaseConfigError, match="unknown tag namespace"):
        policy.apply(["unknown:x"], "session")
    with pytest.raises(QPhaseConfigError, match="not an allowed value"):
        policy.apply(["task:unknown"], "session")


def test_policy_enforces_cardinality_and_object_scope():
    policy = _policy()
    with pytest.raises(QPhaseConfigError, match="allows one tag"):
        policy.apply(["stage:q1", "stage:q2"], "session")
    with pytest.raises(QPhaseConfigError, match="does not apply"):
        policy.apply(["stage:q1"], "artifact")
    with pytest.raises(QPhaseConfigError, match="not hierarchical"):
        policy.apply(["stage:q1/x"], "session")


def test_policy_rejects_reserved_namespace():
    with pytest.raises(QPhaseConfigError, match="reserved"):
        _policy().apply(["system:internal"], "session")


def test_open_namespace_accepts_freeform_values():
    assert _policy().apply(["note:anything/at-all"], "artifact") == [
        "note:anything/at-all"
    ]


def test_policy_revision_tracks_content():
    assert _policy().revision == _policy().revision
    changed = _policy().model_copy(update={"reserved_namespaces": ("sys",)})
    assert changed.revision != _policy().revision


def test_inheritance_and_object_scope_queries():
    policy = _policy()
    assert policy.allows_inheritance("task:bifurcation") is True
    assert policy.allows_inheritance("unregistered:x") is True
    assert policy.tag_applies_to("stage:q1", "workflow") is True
    assert policy.tag_applies_to("stage:q1", "artifact") is False
    assert policy.tag_applies_to("task:bifurcation", "artifact") is True


def test_load_tag_policy_roundtrip_and_absence(tmp_path):
    project = ProjectContext.create(tmp_path / "proj", name="t")
    assert load_tag_policy(project) is None
    policy_path = project.defaults_path.parent / "tags.yaml"
    policy_path.write_text(
        "schema: qphase.tag-policy/1\n"
        "namespaces:\n"
        "  task:\n"
        "    values: [bifurcation]\n",
        encoding="utf-8",
    )
    policy = load_tag_policy(project)
    assert policy is not None
    assert "task" in policy.namespaces


def test_load_tag_policy_rejects_malformed_document(tmp_path):
    project = ProjectContext.create(tmp_path / "proj", name="t")
    policy_path = project.defaults_path.parent / "tags.yaml"
    policy_path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(QPhaseConfigError):
        load_tag_policy(project)


def test_tag_assignment_is_canonicalized_and_identified():
    assignment = TagAssignment(tag="Task:Bifurcation")
    assert assignment.tag == "task:bifurcation"
    assert assignment.id
    other = TagAssignment(tag="task:bifurcation")
    assert other.id != assignment.id


def test_annotation_documents_defaults_and_schema():
    session_doc = SessionAnnotationDocument(project_id="p", session_id="s")
    assert session_doc.revision == 0
    assert session_doc.model_dump(mode="json", by_alias=True)["schema"] == (
        "qphase.session-annotations/1"
    )
    artifact_doc = ArtifactAnnotationDocument(project_id="p", artifact_id="a")
    assert artifact_doc.model_dump(mode="json", by_alias=True)["schema"] == (
        "qphase.artifact-annotations/1"
    )
    session_doc.occurrences["art-1"] = OccurrenceAnnotations(note="kept")
    assert session_doc.occurrences["art-1"].note == "kept"


def test_annotation_documents_forbid_extra_fields():
    with pytest.raises(Exception, match="extra"):
        SessionAnnotationDocument.model_validate(
            {
                "schema": "qphase.session-annotations/1",
                "project_id": "p",
                "session_id": "s",
                "unexpected": 1,
            }
        )



def test_freeze_namespace_rule_snapshots_minimal_governance():
    from qphase.core.tags import freeze_namespace_rule

    policy = _policy()
    rule = freeze_namespace_rule(policy, "stage:q1")
    assert rule is not None
    assert rule.inherit is True
    assert rule.cardinality == "one"
    assert rule.objects == ("session", "workflow")
    # Undeclared namespaces and absent policies freeze no rule.
    assert freeze_namespace_rule(policy, "system:internal") is None
    assert freeze_namespace_rule(None, "task:bifurcation") is None


def test_compute_effective_tags_prefers_frozen_rule_over_current_policy():
    from qphase.core.catalog import compute_effective_tags
    from qphase.core.tags import FrozenNamespaceRule

    frozen = FrozenNamespaceRule(inherit=False, cardinality="many", objects=())
    levels = [
        ("workflow_declared", [("task:bifurcation", "a1", "rev", frozen)], False),
        ("session_annotation", [("task:diagnostics", "a2", "rev", None)], True),
    ]
    # The current policy allows inheritance; the frozen rule still wins.
    tags = compute_effective_tags(levels, _policy(), "session")
    assert [tag.tag for tag in tags] == ["task:diagnostics"]

    # Frozen object applicability likewise overrides the current policy.
    scoped = FrozenNamespaceRule(
        inherit=True, cardinality="many", objects=("workflow",)
    )
    tags = compute_effective_tags(
        [
            ("workflow_declared", [("task:bifurcation", "a1", "rev", scoped)], False),
            ("session_annotation", [], True),
        ],
        _policy(),
        "session",
    )
    assert tags == []

    # Legacy assignments (no frozen rule) fall back to the current policy.
    tags = compute_effective_tags(
        [
            ("workflow_declared", [("task:bifurcation", "a1", "rev", None)], False),
            ("session_annotation", [], True),
        ],
        _policy(),
        "session",
    )
    assert [tag.tag for tag in tags] == ["task:bifurcation"]
    assert tags[0].inherited


def test_frozen_cardinality_overrides_current_policy_shadowing():
    from qphase.core.catalog import compute_effective_tags
    from qphase.core.tags import FrozenNamespaceRule

    # The current policy makes "stage" cardinality-one, but both assignments
    # froze cardinality "many": nothing is shadowed.
    frozen_many = FrozenNamespaceRule(inherit=True, cardinality="many", objects=())
    tags = compute_effective_tags(
        [
            ("workflow_declared", [("stage:q1", "a1", "rev", frozen_many)], False),
            ("session_annotation", [("stage:q2", "a2", "rev", frozen_many)], True),
        ],
        _policy(),
        "session",
    )
    assert [tag.tag for tag in tags] == ["stage:q1", "stage:q2"]
    assert not any(tag.shadowed for tag in tags)

    # Legacy assignments resolve cardinality from the current policy instead.
    tags = compute_effective_tags(
        [
            ("workflow_declared", [("stage:q1", "a1", "rev", None)], False),
            ("session_annotation", [("stage:q2", "a2", "rev", None)], True),
        ],
        _policy(),
        "session",
    )
    assert {tag.tag: tag.shadowed for tag in tags} == {
        "stage:q1": True,
        "stage:q2": False,
    }


def test_tag_assignment_freezes_minimal_namespace_rule():
    assignment = TagAssignment(
        tag="task:bifurcation",
        policy_revision="rev",
        inherit=False,
        cardinality="many",
        objects=("session",),
    )
    document = SessionAnnotationDocument(
        project_id="p", session_id="s", assignments=[assignment]
    )
    payload = document.model_dump(mode="json", by_alias=True)
    restored = SessionAnnotationDocument.model_validate(payload)
    (restored_assignment,) = restored.assignments
    assert restored_assignment.inherit is False
    assert restored_assignment.cardinality == "many"
    assert restored_assignment.objects == ("session",)

    # Legacy assignments without rule fields still load; their resolution
    # falls back to the current policy.
    legacy = TagAssignment(tag="task:bifurcation")
    assert legacy.inherit is None
    assert legacy.cardinality is None
    assert legacy.objects is None
