"""qphase_sde: SDE Task Profile Contracts (2.0)
---------------------------------------------------------
Freezes the three SDE engine task profiles: ``simulate``, ``analyze`` and
``simulate_analyze``. Profiles declare which plugin classes a task requires,
permits or forbids — ``forbidden`` is an explicit configuration error, never a
silent skip, so a plain ``analyze`` task must not instantiate placeholder
model/integrator plugins. A model-aware analyser may still legitimately
require ``model`` through a profile *resolver* returning a complete
requirement set (see ``qphase.core.task_profile``).

Phase 0 freezes these contracts and their negative tests only. Replacing the
legacy ``EngineManifest.input_plugins`` scheduler wiring is Phase 2 work:
**an approved design here is not yet enforced runtime behavior**.

Public API
----------
SDE_TASK_PROFILES
    The frozen SDE task profiles.
sde_task_profile
    Look up one frozen profile by id.
"""

from __future__ import annotations

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
#: forbidden by default. A model-aware analyser must opt in explicitly via a
#: resolver that returns a complete replacement requirement set.
ANALYZE_PROFILE = EngineTaskProfile(
    id="analyze",
    requirements=PluginRequirementSet(
        required=["backend", "analyser"],
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
