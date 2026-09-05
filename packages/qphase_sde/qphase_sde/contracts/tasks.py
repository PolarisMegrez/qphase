"""qphase_sde: SDE Task Profile Contracts (2.0)
---------------------------------------------------------
Freezes the three SDE engine task profiles: ``simulate``, ``analyze`` and
``simulate_analyze``. Profiles declare which plugin classes a task requires,
permits or forbids — ``forbidden`` is an explicit configuration error, never a
silent skip, so a plain ``analyze`` task must not instantiate placeholder
model/integrator plugins. A model-aware analyser may still legitimately
require ``model`` through a profile *resolver* returning a complete
requirement set (see ``qphase.core.task_profile``).

The workflow compiler enforces these profiles for every SDE job: the
``EngineManifest`` points at :data:`SDE_TASK_PROFILES` and
:func:`select_sde_task_profile`, replacing the legacy
``required_plugins``/``input_plugins`` conditional scheduling for this engine.

Public API
----------
SDE_TASK_PROFILES
    The frozen SDE task profiles.
sde_task_profile
    Look up one frozen profile by id.
select_sde_task_profile
    Map a normalized job config to a task profile id.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qphase.core.task_profile import (
    EngineTaskProfile,
    InputProductRequirement,
    OutputProductDeclaration,
    PluginRequirementSet,
)
from qphase.data import (
    DataKind,
    ProductDeclaration,
    ProductRequirement,
)

__all__ = [
    "SDE_TASK_PROFILES",
    "select_sde_task_profile",
    "sde_task_profile",
]

#: Simulation: integrate trajectories; observers and inline analysers are
#: optional add-ons.
SIMULATE_PROFILE = EngineTaskProfile(
    id="simulate",
    requirements=PluginRequirementSet(
        required=["backend", "model", "integrator"],
        optional=["observer", "analyser"],
    ),
    outputs=[
        OutputProductDeclaration(
            name="trajectories",
            declaration=ProductDeclaration(
                name="trajectories", kind=DataKind.TIME_SERIES
            ),
        )
    ],
)

#: Analysis of existing typed products: model/integrator/observer are
#: forbidden by default and backend support is optional (analysers fall back
#: to a host NumPy backend). A model-aware analyser must opt in explicitly via
#: a resolver that returns a complete replacement requirement set.
ANALYZE_PROFILE = EngineTaskProfile(
    id="analyze",
    requirements=PluginRequirementSet(
        required=["analyser"],
        optional=["backend"],
        forbidden=["model", "integrator", "observer"],
    ),
    inputs=[
        InputProductRequirement(
            name="input",
            requirement=ProductRequirement(name="input"),
        )
    ],
)

#: Combined simulation followed by analysis of the fresh trajectories.
SIMULATE_ANALYZE_PROFILE = EngineTaskProfile(
    id="simulate_analyze",
    requirements=PluginRequirementSet(
        required=["backend", "model", "integrator", "analyser"],
        optional=["observer"],
    ),
    outputs=[
        OutputProductDeclaration(
            name="trajectories",
            declaration=ProductDeclaration(
                name="trajectories", kind=DataKind.TIME_SERIES
            ),
        )
    ],
)

#: The frozen SDE task profiles, keyed by task id.
SDE_TASK_PROFILES: tuple[EngineTaskProfile, ...] = (
    SIMULATE_PROFILE,
    ANALYZE_PROFILE,
    SIMULATE_ANALYZE_PROFILE,
)


def sde_task_profile(task_id: str) -> EngineTaskProfile:
    """Return the frozen SDE task profile with the given id."""
    for profile in SDE_TASK_PROFILES:
        if profile.id == task_id:
            return profile
    raise KeyError(f"unknown SDE task profile {task_id!r}")


def select_sde_task_profile(normalized_job_config: dict[str, Any]) -> str:
    """Select the SDE task profile id for one normalized job config.

    The user-facing config keeps only ``mode: simulate|analyze``:

    - ``mode: analyze`` maps to ``analyze``;
    - ``mode: simulate`` (the default) with at least one configured analyser
      maps to ``simulate_analyze``;
    - ``mode: simulate`` without analysers maps to ``simulate``.

    ``normalized_job_config`` is the compiler-merged job configuration: the
    engine section lives under ``engine.sde`` and configured analysers under
    the ``analyser`` plugin-class namespace (top-level or ``plugins`` block).
    """
    mode = "simulate"
    engines = normalized_job_config.get("engine")
    if isinstance(engines, Mapping):
        engine_config = engines.get("sde")
        if isinstance(engine_config, Mapping):
            mode = str(engine_config.get("mode", "simulate"))
    if mode == "analyze":
        return "analyze"
    if _configured_analysers(normalized_job_config):
        return "simulate_analyze"
    return "simulate"


def _configured_analysers(normalized_job_config: Mapping[str, Any]) -> bool:
    """Return whether the job configures at least one analyser plugin."""
    analysers = normalized_job_config.get("analyser")
    if isinstance(analysers, Mapping) and analysers:
        return True
    plugins = normalized_job_config.get("plugins")
    if isinstance(plugins, Mapping):
        analysers = plugins.get("analyser")
        if isinstance(analysers, Mapping) and analysers:
            return True
    return False
