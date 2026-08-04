"""High-order CAM equilibrium bifurcation solver plugin."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from qphase.core.protocols import PluginManifest, SubpluginSlot

from qphase_cam.errors import BifurcationCapabilityError
from qphase_cam.state import CAMBifurcationOutput

from .base import CAMSolver, CAMSolverConfig
from .bifurcation_discovery import BifurcationDiscovery
from .bifurcation_strategy import BifurcationStrategy
from .bifurcation_target import BifurcationTarget


class ControlRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min: float
    max: float
    scale: float | None = Field(None, gt=0.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> ControlRange:
        if self.max <= self.min:
            raise ValueError("control max must be greater than min")
        return self


class RefinementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tolerance: float = Field(1e-11, gt=0.0)
    max_iterations: int = Field(100, ge=1)


class VerificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    initial_digits: int = Field(50, ge=20)
    max_digits: int = Field(200, ge=20)

    @model_validator(mode="after")
    def validate_digits(self) -> VerificationConfig:
        if self.max_digits < self.initial_digits:
            raise ValueError("max_digits must be at least initial_digits")
        return self


class BifurcationSolverConfig(CAMSolverConfig):
    controls: dict[str, ControlRange]
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)


class BifurcationSolver(CAMSolver):
    """Search model parameters for high-order CAM equilibrium roots."""

    name: ClassVar[str] = "bifurcation"
    description: ClassVar[str] = "High-order CAM equilibrium bifurcation search"
    config_schema: ClassVar[type[BifurcationSolverConfig]] = BifurcationSolverConfig
    output_kind: ClassVar[str] = "bifurcation_candidates"
    manifest: ClassVar[PluginManifest] = PluginManifest(
        subplugins={
            "target": SubpluginSlot(
                namespace="bifurcation_target",
                protocol=(
                    "qphase_cam.solver.bifurcation_target:BifurcationTarget"
                ),
                allowed=frozenset({"equilibrium_multiplicity"}),
            ),
            "strategy": SubpluginSlot(
                namespace="bifurcation_strategy",
                default="auto",
                protocol=(
                    "qphase_cam.solver.bifurcation_strategy:BifurcationStrategy"
                ),
                allowed=frozenset({"auto", "reduced", "full"}),
            ),
            "discovery": SubpluginSlot(
                namespace="bifurcation_discovery",
                default="seeds",
                protocol=(
                    "qphase_cam.solver.bifurcation_discovery:BifurcationDiscovery"
                ),
                allowed=frozenset({"seeds", "continuation"}),
            ),
        }
    )

    def __init__(
        self,
        config: BifurcationSolverConfig | None = None,
        *,
        subplugins: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        if subplugins is None:
            raise TypeError("bifurcation solver requires resolved subplugins")
        self.target: BifurcationTarget = subplugins["target"]
        self.strategy: BifurcationStrategy = subplugins["strategy"]
        self.discovery: BifurcationDiscovery = subplugins["discovery"]
        if len(self.config.controls) != self.target.order - 1:
            raise ValueError(
                "equilibrium multiplicity order m requires exactly m-1 controls"
            )

    def solve(
        self,
        model: Any,
        backend: Any,
        *,
        data: Any | None = None,
        context: Any | None = None,
    ) -> CAMBifurcationOutput:
        del model, backend, data, context
        raise BifurcationCapabilityError(
            "bifurcation solver configuration is valid, but its numerical "
            "strategy has not been initialized"
        )
