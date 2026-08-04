"""Candidate-discovery subplugins for bifurcation searches."""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import ConfigDict, Field
from qphase.core.protocols import PluginConfigBase


@runtime_checkable
class BifurcationDiscovery(Protocol):
    name: ClassVar[str]

    def seed_values(self, data: Any | None = None) -> list[Any]: ...


class SeedDiscoveryConfig(PluginConfigBase):
    model_config = ConfigDict(extra="forbid")
    samples_per_control: int = Field(7, ge=2, le=101)
    max_starts: int = Field(4096, ge=1)


class SeedDiscovery:
    name: ClassVar[str] = "seeds"
    description: ClassVar[str] = "Domain sampling and upstream result seeds"
    config_schema: ClassVar[type[SeedDiscoveryConfig]] = SeedDiscoveryConfig

    def __init__(self, config: SeedDiscoveryConfig) -> None:
        self.config = config

    def seed_values(self, data: Any | None = None) -> list[Any]:
        return [] if data is None else [data]


class ContinuationDiscoveryConfig(SeedDiscoveryConfig):
    initial_step: float = Field(0.002, gt=0.0)
    max_steps: int = Field(2000, ge=1)


class ContinuationDiscovery(SeedDiscovery):
    name: ClassVar[str] = "continuation"
    description: ClassVar[str] = "Continuation-informed candidate discovery"
    config_schema: ClassVar[type[ContinuationDiscoveryConfig]] = (
        ContinuationDiscoveryConfig
    )
