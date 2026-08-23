"""qphase: Engine Task Profile Contracts
---------------------------------------------------------
Freezes the dynamic task-profile contract: engines describe which plugin
classes and typed input products a job requires, and which products it
declares as output. A profile *resolver* refines static requirements based on
the job configuration and input product **schemas** only — it must never read
the underlying large arrays.

This replaces the implicit convention where an analyze-mode job pretends to
own a model or integrator.

Public API
----------
PluginRequirementSet
    Required/optional/forbidden plugin-class namespaces.
InputProductRequirement
    Named, typed input product selector.
OutputProductDeclaration
    Declared output product.
EngineTaskProfile
    Complete task profile with an optional resolver reference.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..data.graph import ProductDeclaration, ProductRequirement

__all__ = [
    "EngineTaskProfile",
    "InputProductRequirement",
    "OutputProductDeclaration",
    "PluginRequirementSet",
]

_DOTTED_PATH_PATTERN = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")


class PluginRequirementSet(BaseModel):
    """Required, optional and forbidden plugin-class namespaces of a task."""

    model_config = ConfigDict(extra="forbid")

    required: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)


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

    ``requirements`` states the static plugin-class requirements; ``resolver``
    optionally points to a callable ``(config, input_schemas) ->
    PluginRequirementSet`` that refines them. Resolvers stay dotted-path
    strings so profiles remain JSON-serializable.
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
