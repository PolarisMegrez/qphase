"""qphase: Engine Task Profile Contracts
---------------------------------------------------------
Freezes the dynamic task-profile contract: engines describe which plugin
classes and typed input products a job requires, and which products it
declares as output. A profile *resolver* refines the profile's default
requirements based on a restricted context (normalized job config + input
product **schemas** only — never handles, payloads, artifact loaders, the
scheduler or the execution context) and returns a **complete**
``PluginRequirementSet`` that replaces the defaults, so a model-aware analyser
may legitimately turn ``model`` from forbidden into required without any
implicit merge order.

``forbidden`` is an explicit configuration error, not a silent "ignore this
plugin". Resolver outputs are re-validated with the same invariants.

Public API
----------
PluginRequirementSet
    Required/optional/forbidden plugin-class namespaces.
InputProductRequirement
    Named, typed input product selector.
OutputProductDeclaration
    Declared output product.
TaskProfileResolutionContext
    Restricted resolver input.
EngineTaskProfile
    Complete task profile with an optional resolver reference.
resolve_plugin_requirements
    Resolve a profile against a context, re-validating resolver output.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..data.graph import ProductDeclaration, ProductRequirement
from ..data.schema import ProductSchema

__all__ = [
    "EngineTaskProfile",
    "InputProductRequirement",
    "OutputProductDeclaration",
    "PluginRequirementSet",
    "TaskProfileResolutionContext",
    "resolve_plugin_requirements",
]

_DOTTED_PATH_PATTERN = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")
_NAMESPACE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class PluginRequirementSet(BaseModel):
    """Required/optional/forbidden plugin-class namespaces of a task.

    The three sets are pairwise disjoint, contain unique registry-style
    namespaces, and are stored in sorted order so fingerprints never depend on
    YAML/list ordering.
    """

    model_config = ConfigDict(extra="forbid")

    required: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)

    @field_validator("required", "optional", "forbidden")
    @classmethod
    def _check_namespaces(cls, value: list[str]) -> list[str]:
        for namespace in value:
            if not _NAMESPACE_PATTERN.match(namespace):
                raise ValueError(
                    f"invalid plugin-class namespace: {namespace!r}"
                )
        if len(set(value)) != len(value):
            raise ValueError("duplicate plugin-class namespace in one set")
        return sorted(value)

    @field_validator("forbidden")
    @classmethod
    def _forbidden_is_explicit(cls, value: list[str]) -> list[str]:
        # ``forbidden`` is an explicit configuration error surface, not a
        # silent skip: listing a namespace here makes its presence an error.
        return value

    @model_validator(mode="after")
    def _check_disjoint(self) -> PluginRequirementSet:
        sets = {
            "required": set(self.required),
            "optional": set(self.optional),
            "forbidden": set(self.forbidden),
        }
        for left_name, right_name in (
            ("required", "optional"),
            ("required", "forbidden"),
            ("optional", "forbidden"),
        ):
            overlap = sets[left_name] & sets[right_name]
            if overlap:
                raise ValueError(
                    f"plugin requirement sets must be disjoint; {left_name} "
                    f"and {right_name} share {sorted(overlap)}"
                )
        return self


def validate_requirement_set(requirements: PluginRequirementSet) -> None:
    """Re-validate a requirement set (used on resolver outputs)."""
    PluginRequirementSet.model_validate(
        requirements.model_dump(mode="json")
    )


class TaskProfileResolutionContext(BaseModel):
    """Restricted input available to profile resolvers.

    Resolvers receive the normalized job config and the schemas of named
    input products — never data handles, product payloads, artifact loaders,
    the scheduler or the execution context.
    """

    model_config = ConfigDict(extra="forbid")

    normalized_job_config: dict[str, Any] = Field(default_factory=dict)
    named_input_product_schemas: dict[str, ProductSchema] = Field(
        default_factory=dict
    )


class InputProductRequirement(BaseModel):
    """One named, typed input product requirement of a task.

    The ``name`` is the input slot used by named multi-input job selectors;
    ``requirement`` carries the type-level constraints.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    requirement: ProductRequirement


class OutputProductDeclaration(BaseModel):
    """One declared output product of a task."""

    model_config = ConfigDict(extra="forbid")

    name: str
    declaration: ProductDeclaration


class EngineTaskProfile(BaseModel):
    """Conditional task profile of an engine.

    ``requirements`` states the default plugin-class requirements;
    ``resolver`` optionally points to a callable
    ``(TaskProfileResolutionContext) -> PluginRequirementSet`` whose result
    *replaces* the defaults. Resolvers stay dotted-path strings so profiles
    remain JSON-serializable.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    requirements: PluginRequirementSet = Field(
        default_factory=PluginRequirementSet
    )
    inputs: list[InputProductRequirement] = Field(default_factory=list)
    outputs: list[OutputProductDeclaration] = Field(default_factory=list)
    resolver: str | None = Field(
        default=None,
        description="Dotted path to the profile resolver callable, if any.",
    )

    @field_validator("resolver")
    @classmethod
    def _check_resolver(cls, value: str | None) -> str | None:
        if value is not None and not _DOTTED_PATH_PATTERN.match(value):
            raise ValueError(f"invalid resolver dotted path: {value!r}")
        return value


def _default_loader(dotted_path: str) -> Any:
    """Import a 'module:attr' dotted path."""
    from importlib import import_module

    module_name, _, attr = dotted_path.partition(":")
    target: Any = import_module(module_name)
    for part in attr.split("."):
        target = getattr(target, part)
    return target


def resolve_plugin_requirements(
    profile: EngineTaskProfile,
    context: TaskProfileResolutionContext,
    loader: Callable[[str], Any] = _default_loader,
) -> PluginRequirementSet:
    """Resolve a task profile to a complete plugin requirement set.

    Without a resolver, the profile's default requirements are returned. With
    a resolver, the resolver receives only the restricted context and must
    return a complete ``PluginRequirementSet`` which replaces the defaults;
    the result is re-validated with the same invariants.
    """
    if profile.resolver is None:
        return profile.requirements
    resolver = loader(profile.resolver)
    resolved = resolver(context)
    if not isinstance(resolved, PluginRequirementSet):
        raise TypeError(
            f"task profile resolver {profile.resolver!r} must return a "
            f"PluginRequirementSet, got {type(resolved).__name__}"
        )
    validate_requirement_set(resolved)
    return resolved
