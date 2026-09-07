"""Fixture engine declaring conditional task profiles for compiler tests."""

from typing import Any, ClassVar

from pydantic import BaseModel
from qphase.core.protocols import EngineManifest
from qphase.core.task_profile import (
    EngineTaskProfile,
    InputProductRequirement,
    PluginRequirementSet,
    TaskProfileResolutionContext,
)
from qphase.data import ProductRequirement


class ProfiledEngineConfig(BaseModel):
    mode: str = "simulate"


def select_profile(job_explicit_config: dict[str, Any]) -> str:
    """Select 'resolved' when the engine config asks for it, else 'base'."""
    engines = job_explicit_config.get("engine")
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


#: Resolver contexts captured by ``resolve_recording_profile``; tests clear
#: this list before compiling.
RECORDED_CONTEXTS: list[TaskProfileResolutionContext] = []


def resolve_recording_profile(
    context: TaskProfileResolutionContext,
) -> PluginRequirementSet:
    """Record the resolver context and return a fixed requirement set."""
    RECORDED_CONTEXTS.append(context)
    return PluginRequirementSet(required=["backend"])


RECORDING_PROFILE = EngineTaskProfile(
    id="base",
    requirements=PluginRequirementSet(required=["backend"]),
    resolver="tests.plugins.profiled_plugins:resolve_recording_profile",
)

#: Single-profile collection whose resolver records its resolution context.
RECORDING_PROFILES = (RECORDING_PROFILE,)


class RecordingProfileEngine:
    """Engine fixture whose profile resolver records the context it receives."""

    name: ClassVar[str] = "profiled_recording"
    description: ClassVar[str] = "compiler task-profile context recording engine"
    config_schema: ClassVar[type[ProfiledEngineConfig]] = ProfiledEngineConfig
    manifest: ClassVar[EngineManifest] = EngineManifest(
        task_profiles="tests.plugins.profiled_plugins:RECORDING_PROFILES",
        profile_selector="tests.plugins.profiled_plugins:select_profile",
    )

    def __init__(self, config: Any = None, plugins: dict | None = None, **kwargs: Any):
        del kwargs
        self.config = config
        self.plugins = plugins or {}


DUPLICATE_PROFILE = EngineTaskProfile(
    id="base",
    requirements=PluginRequirementSet(required=["model"]),
)

#: Collection with a duplicate profile id (``base`` appears twice).
DUPLICATE_PROFILES = (BASE_PROFILE, DUPLICATE_PROFILE)


class DuplicateProfileEngine:
    """Engine fixture declaring two profiles with the same id."""

    name: ClassVar[str] = "profiled_duplicate"
    description: ClassVar[str] = "compiler duplicate task-profile id engine"
    config_schema: ClassVar[type[ProfiledEngineConfig]] = ProfiledEngineConfig
    manifest: ClassVar[EngineManifest] = EngineManifest(
        task_profiles="tests.plugins.profiled_plugins:DUPLICATE_PROFILES",
        profile_selector="tests.plugins.profiled_plugins:select_profile",
    )

    def __init__(self, config: Any = None, plugins: dict | None = None, **kwargs: Any):
        del kwargs
        self.config = config
        self.plugins = plugins or {}


def select_multi_input_profile(job_explicit_config: dict[str, Any]) -> str:
    """Select the multi-input profile unconditionally."""
    del job_explicit_config
    return "multi"


MULTI_INPUT_PROFILE = EngineTaskProfile(
    id="multi",
    requirements=PluginRequirementSet(required=["backend"]),
    inputs=[
        InputProductRequirement(
            name="left",
            requirement=ProductRequirement(name="left"),
        ),
        InputProductRequirement(
            name="right",
            requirement=ProductRequirement(name="right"),
        ),
    ],
)

#: Profile declaring more required input slots than a job input can express.
MULTI_INPUT_PROFILES = (MULTI_INPUT_PROFILE,)


class MultiInputProfileEngine:
    """Engine fixture whose only profile requires two input slots."""

    name: ClassVar[str] = "profiled_multi_input"
    description: ClassVar[str] = "compiler multi-input task-profile engine"
    config_schema: ClassVar[type[ProfiledEngineConfig]] = ProfiledEngineConfig
    manifest: ClassVar[EngineManifest] = EngineManifest(
        task_profiles="tests.plugins.profiled_plugins:MULTI_INPUT_PROFILES",
        profile_selector="tests.plugins.profiled_plugins:select_multi_input_profile",
    )

    def __init__(self, config: Any = None, plugins: dict | None = None, **kwargs: Any):
        del kwargs
        self.config = config
        self.plugins = plugins or {}
