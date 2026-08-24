"""Contract tests for engine task profiles and profile resolution."""

import pytest
from pydantic import ValidationError
from qphase.core.task_profile import (
    EngineTaskProfile,
    PluginRequirementSet,
    TaskProfileResolutionContext,
    _default_loader,
    resolve_plugin_requirements,
)
from qphase.data import AxisRole, AxisSchema, DataKind, ProductSchema, VariableSchema


def test_requirement_sets_must_be_pairwise_disjoint():
    """A namespace may not appear in two requirement sets at once."""
    with pytest.raises(ValidationError, match="disjoint"):
        PluginRequirementSet(required=["model"], optional=["model"])
    with pytest.raises(ValidationError, match="disjoint"):
        PluginRequirementSet(required=["model"], forbidden=["model"])
    with pytest.raises(ValidationError, match="disjoint"):
        PluginRequirementSet(optional=["observer"], forbidden=["observer"])


def test_requirement_set_namespace_validation():
    """Namespaces are lowercase registry-style identifiers, unique per set."""
    for bad in ("Backend", "my-plugin", "1x", "a b", ""):
        with pytest.raises(ValidationError, match="namespace"):
            PluginRequirementSet(required=[bad])
    with pytest.raises(ValidationError, match="duplicate"):
        PluginRequirementSet(optional=["model", "model"])


def test_requirement_set_storage_is_order_independent():
    """Sets are stored sorted so fingerprints never depend on YAML order."""
    left = PluginRequirementSet(
        required=["model", "backend"], optional=["observer"]
    )
    right = PluginRequirementSet(
        required=["backend", "model"], optional=["observer"]
    )
    assert left == right
    assert left.required == ["backend", "model"]
    assert left.model_dump(mode="json") == right.model_dump(mode="json")


def test_resolution_context_is_extra_forbid_and_carries_schemas():
    """The resolver context rejects extras and carries input schemas only."""
    with pytest.raises(ValidationError):
        TaskProfileResolutionContext(normalized_job_config={}, bogus=1)
    with pytest.raises(ValidationError, match="JSON-serializable"):
        TaskProfileResolutionContext(
            normalized_job_config={"runtime_handle": object()}
        )

    schema = ProductSchema(
        kind=DataKind.SPECTRAL,
        axes=[
            AxisSchema(name="frequency", role=AxisRole.COORDINATE, size=4),
        ],
        variables=[
            VariableSchema(
                name="power", dtype="float64", value_domain="real",
                dims=("frequency",),
            )
        ],
        attributes={
            "frequency_units": "Hz",
            "orientation": "phase_decreasing",
            "sidedness": "one_sided",
            "normalization": "density",
            "window": "hann",
            "estimator": "periodogram",
        },
    )
    context = TaskProfileResolutionContext(
        normalized_job_config={"task": "analyze"},
        named_input_product_schemas={"spectrum": schema},
    )
    assert context.named_input_product_schemas["spectrum"] == schema


def test_resolve_without_resolver_returns_defaults():
    """Profiles without a resolver resolve to their default requirements."""
    profile = EngineTaskProfile(
        id="simulate",
        requirements=PluginRequirementSet(required=["model", "backend"]),
    )
    resolved = resolve_plugin_requirements(
        profile, TaskProfileResolutionContext()
    )
    assert resolved == profile.requirements


def test_resolver_replaces_defaults_completely():
    """Resolver output replaces the defaults wholesale — no implicit merge."""
    profile = EngineTaskProfile(
        id="analyze",
        requirements=PluginRequirementSet(
            required=["analyser"], forbidden=["model"]
        ),
        resolver="fake.module:resolve",
    )
    seen = {}

    def resolver(context):
        seen["context"] = context
        # A model-aware analyser legitimately turns ``model`` into required.
        return PluginRequirementSet(required=["analyser", "model"])

    context = TaskProfileResolutionContext(normalized_job_config={"task": "x"})
    resolved = resolve_plugin_requirements(
        profile, context, loader=lambda _path: resolver
    )
    assert resolved.required == ["analyser", "model"]
    assert resolved.forbidden == []
    assert seen["context"] is context


def test_resolver_must_return_requirement_set():
    """Resolvers returning anything else are rejected with a TypeError."""
    profile = EngineTaskProfile(id="x", resolver="fake.module:resolve")
    context = TaskProfileResolutionContext()
    with pytest.raises(TypeError, match="PluginRequirementSet"):
        resolve_plugin_requirements(
            profile, context, loader=lambda _path: lambda _ctx: {"required": []}
        )
    with pytest.raises(TypeError, match="PluginRequirementSet"):
        resolve_plugin_requirements(
            profile, context, loader=lambda _path: lambda _ctx: None
        )


def test_resolver_output_is_revalidated():
    """Resolver outputs face the same invariants as static profiles."""
    profile = EngineTaskProfile(id="x", resolver="fake.module:resolve")

    def resolver(_context):
        requirements = PluginRequirementSet(required=["model"])
        # Bypass constructor validation to simulate a tampered resolver.
        object.__setattr__(requirements, "forbidden", ["model"])
        return requirements

    with pytest.raises(ValidationError, match="disjoint"):
        resolve_plugin_requirements(
            profile, TaskProfileResolutionContext(), loader=lambda _path: resolver
        )


def test_default_loader_imports_dotted_paths():
    """The default loader resolves 'module:attr' references."""
    target = _default_loader("qphase.core.task_profile:validate_requirement_set")
    assert callable(target)


def test_resolver_dotted_path_validation():
    """Resolver references must be 'module:attr' dotted paths."""
    with pytest.raises(ValidationError, match="dotted path"):
        EngineTaskProfile(id="bad", resolver="not a path!")
    ok = EngineTaskProfile(
        id="ok", resolver="qphase_sde.config:resolve_analyze_profile"
    )
    assert ok.resolver == "qphase_sde.config:resolve_analyze_profile"
