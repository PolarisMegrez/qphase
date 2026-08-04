"""Bifurcation search strategy subplugins."""

from __future__ import annotations

from typing import ClassVar, Literal, Protocol, runtime_checkable

from pydantic import ConfigDict, Field, model_validator
from qphase.core.protocols import PluginConfigBase


@runtime_checkable
class BifurcationStrategy(Protocol):
    name: ClassVar[str]
    mode: Literal["auto", "reduced", "full"]


class ReductionStrategyConfig(PluginConfigBase):
    model_config = ConfigDict(extra="forbid")
    retained_dimension: Literal[1] = 1
    order_parameter: str | None = None
    max_candidates: int = Field(8, ge=1)
    condition_limit: float = Field(1e10, gt=1.0)
    singular_tolerance: float = Field(1e-10, gt=0.0)
    order_parameter_bounds: tuple[float, float] | None = None

    @model_validator(mode="after")
    def validate_order_parameter_bounds(self) -> ReductionStrategyConfig:
        if (
            self.order_parameter_bounds is not None
            and self.order_parameter_bounds[1] <= self.order_parameter_bounds[0]
        ):
            raise ValueError("order_parameter_bounds must be increasing")
        return self


class AutoStrategy:
    name: ClassVar[str] = "auto"
    description: ClassVar[str] = "Reduced search with full-system fallback"
    config_schema: ClassVar[type[ReductionStrategyConfig]] = ReductionStrategyConfig
    mode: Literal["auto"] = "auto"

    def __init__(self, config: ReductionStrategyConfig) -> None:
        self.config = config


class ReducedStrategy(AutoStrategy):
    name: ClassVar[str] = "reduced"
    description: ClassVar[str] = "Require a regular scalar reduction"
    mode: Literal["reduced"] = "reduced"


class FullStrategyConfig(PluginConfigBase):
    model_config = ConfigDict(extra="forbid")
    singular_tolerance: float = Field(1e-10, gt=0.0)


class FullStrategy:
    name: ClassVar[str] = "full"
    description: ClassVar[str] = "Full bordered-system search"
    config_schema: ClassVar[type[FullStrategyConfig]] = FullStrategyConfig
    mode: Literal["full"] = "full"

    def __init__(self, config: FullStrategyConfig) -> None:
        self.config = config
