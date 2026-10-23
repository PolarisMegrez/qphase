"""Core-owned engine fixture, so core tests never depend on a resource package.

The compiler, the job-config parser and the scheduler treat an engine as a
plugin providing a ``config_schema`` plus a manifest; none of them needs the
real SDE engine. Tests that only exercise those core behaviours register this
fixture instead, which keeps ``tests/qphase`` free of ``qphase_sde`` imports.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from qphase.core.protocols import EngineManifest


class DummyEngineConfig(BaseModel):
    """Permissive engine configuration: the tests only vary a few fields."""

    model_config = ConfigDict(extra="allow")

    t0: float = 0.0
    t1: float = 1.0
    dt: float = 0.01
    n_traj: int = 2
    seed: int | None = None
    mode: str = "simulate"


class DummyEngine:
    """Minimal engine plugin: one config schema, one manifest, no run loop."""

    name: ClassVar[str] = "dummy"
    description: ClassVar[str] = "core test engine"
    config_schema: ClassVar[type[DummyEngineConfig]] = DummyEngineConfig
    manifest: ClassVar[EngineManifest] = EngineManifest()

    def __init__(self, config: Any = None, plugins: dict | None = None, **kwargs: Any):
        del kwargs
        self.config = config
        self.plugins = plugins or {}


class DummyEngineRequiringModel(DummyEngine):
    """Engine fixture whose manifest requires a ``model`` plugin.

    Gives the scheduler's missing-required-plugin validation something to
    enforce without borrowing a resource package's manifest.
    """

    name: ClassVar[str] = "dummy_strict"
    manifest: ClassVar[EngineManifest] = EngineManifest(required_plugins={"model"})
