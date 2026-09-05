"""Fixture engine declaring conditional task profiles for compiler tests."""

from typing import Any, ClassVar

from pydantic import BaseModel
from qphase.core.protocols import EngineManifest
from qphase.core.task_profile import (
    EngineTaskProfile,
    PluginRequirementSet,
    TaskProfileResolutionContext,
)


class ProfiledEngineConfig(BaseModel):
    mode: str = "simulate"


def select_profile(normalized_job_config: dict[str, Any]) -> str:
    """Select 'resolved' when the engine config asks for it, else 'base'."""
    engines = normalized_job_config.get("engine")
    if isinstance(engines, dict):
        config = engines.get("profiled")
        if isinstance(config, dict) and config.get("mode") == "resolved":
            return "resolved"
    return "base"


def resolve_base_profile(
    context: TaskProfileResolutionContext,
) -> PluginRequirementSet:
    """Replace the base profile defaults with a complete requirement set."""
    del context
    return PluginRequirementSet(required=["backend", "model"], forbidden=["observer"])


BASE_PROFILE = EngineTaskProfile(
    id="base",
    requirements=PluginRequirementSet(required=["backend"]),
    resolver="tests.plugins.profiled_plugins:resolve_base_profile",
)

RESOLVED_PROFILE = EngineTaskProfile(
    id="resolved",
    requirements=PluginRequirementSet(required=["backend"], optional=["observer"]),
)

#: Profile collection referenced by the fixture engine manifest.
PROFILES = (BASE_PROFILE, RESOLVED_PROFILE)


class ProfiledEngine:
    """Engine fixture wiring task profiles through its manifest."""

    name: ClassVar[str] = "profiled"
    description: ClassVar[str] = "compiler task-profile test engine"
    config_schema: ClassVar[type[ProfiledEngineConfig]] = ProfiledEngineConfig
    manifest: ClassVar[EngineManifest] = EngineManifest(
        task_profiles="tests.plugins.profiled_plugins:PROFILES",
        profile_selector="tests.plugins.profiled_plugins:select_profile",
    )

    def __init__(self, config: Any = None, plugins: dict | None = None, **kwargs: Any):
        del kwargs
        self.config = config
        self.plugins = plugins or {}
