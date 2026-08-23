"""qphase: Resource Package Profiles
---------------------------------------------------------
Defines the composable resource profiles (``base``, ``compute``, ``simulation``)
and the standard root-level assets each profile requires. Profiles keep the
resource-package skeleton minimal: a package only ships the modules and
directories that carry real semantics for it (for example, a visualization
package declares ``base`` alone and is not forced to create a meaningless
``model.py``).

Public API
----------
ResourceProfile
    Enum of composable resource profiles.
profile_required_modules
    Return the root-level module files required by a set of profiles.
profile_required_directories
    Return the root-level directories required by a set of profiles.
"""

from enum import Enum

__all__ = [
    "ResourceProfile",
    "profile_required_directories",
    "profile_required_modules",
]


class ResourceProfile(str, Enum):
    """Composable capability profiles of a resource package.

    Attributes
    ----------
    BASE : str
        Mandatory profile: manifest, engine, config, state, result, errors and
        the public ``contracts/`` directory.
    COMPUTE : str
        Adds ``planning.py`` (resolved plugins + input products compiled into an
        execution plan) and ``runtime/`` (package-private execution helpers).
    SIMULATION : str
        Adds ``model.py`` as the stable entry point for model protocols.

    """

    BASE = "base"
    COMPUTE = "compute"
    SIMULATION = "simulation"


_BASE_MODULES: tuple[str, ...] = (
    "manifest.py",
    "engine.py",
    "config.py",
    "state.py",
    "result.py",
    "errors.py",
)
_BASE_DIRECTORIES: tuple[str, ...] = ("contracts",)
_COMPUTE_MODULES: tuple[str, ...] = ("planning.py",)
_COMPUTE_DIRECTORIES: tuple[str, ...] = ("runtime",)
_SIMULATION_MODULES: tuple[str, ...] = ("model.py",)
_SIMULATION_DIRECTORIES: tuple[str, ...] = ()

# Standard optional asset directories. They are only allowed at the package root
# when explicitly declared in the resource manifest.
STANDARD_OPTIONAL_DIRECTORIES: tuple[str, ...] = (
    "math",
    "serialization",
    "_native",
)


def _collect(
    profiles: set[ResourceProfile],
    base: tuple[str, ...],
    compute: tuple[str, ...],
    simulation: tuple[str, ...],
) -> tuple[str, ...]:
    items: list[str] = list(base)
    if ResourceProfile.COMPUTE in profiles:
        items.extend(compute)
    if ResourceProfile.SIMULATION in profiles:
        items.extend(simulation)
    return tuple(items)


def profile_required_modules(profiles: set[ResourceProfile]) -> tuple[str, ...]:
    """Return root-level module files required by the given profiles."""
    return _collect(
        profiles, _BASE_MODULES, _COMPUTE_MODULES, _SIMULATION_MODULES
    )


def profile_required_directories(profiles: set[ResourceProfile]) -> tuple[str, ...]:
    """Return root-level directories required by the given profiles."""
    return _collect(
        profiles, _BASE_DIRECTORIES, _COMPUTE_DIRECTORIES, _SIMULATION_DIRECTORIES
    )
