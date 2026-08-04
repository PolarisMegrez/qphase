"""Bifurcation target subplugins."""

from __future__ import annotations

from typing import ClassVar, Literal, Protocol, runtime_checkable

from pydantic import ConfigDict
from qphase.core.protocols import PluginConfigBase


@runtime_checkable
class BifurcationTarget(Protocol):
    """Condition family solved by a bifurcation solver."""

    name: ClassVar[str]
    order: int


class EquilibriumMultiplicityConfig(PluginConfigBase):
    model_config = ConfigDict(extra="forbid")
    order: Literal[2, 3, 4]


class EquilibriumMultiplicity:
    """Double, triple, or quadruple equilibrium root target."""

    name: ClassVar[str] = "equilibrium_multiplicity"
    description: ClassVar[str] = "Equilibrium root multiplicity"
    config_schema: ClassVar[type[EquilibriumMultiplicityConfig]] = (
        EquilibriumMultiplicityConfig
    )

    def __init__(self, config: EquilibriumMultiplicityConfig) -> None:
        self.config = config

    @property
    def order(self) -> int:
        return int(self.config.order)
