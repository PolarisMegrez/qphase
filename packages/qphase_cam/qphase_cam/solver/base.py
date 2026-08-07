"""Public base class for CAM solver plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Literal, TypeVar

from pydantic import ConfigDict
from qphase.core.protocols import PluginConfigBase

from qphase_cam.state import CAMSolverResult


class CAMSolverConfig(PluginConfigBase):
    """Strict base schema for CAM solver configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


SolverConfigT = TypeVar("SolverConfigT", bound=CAMSolverConfig)


class CAMSolver(ABC, Generic[SolverConfigT]):
    """Base class for solver plugins selected by the CAM engine."""

    name: ClassVar[str]
    description: ClassVar[str]
    config_schema: ClassVar[type[SolverConfigT]]
    supports_batch: ClassVar[bool] = False
    output_kind: ClassVar[Literal["fixed_points", "bifurcation_candidates"]] = (
        "fixed_points"
    )
    config: SolverConfigT

    def __init__(self, config: SolverConfigT | None = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise TypeError("provide either config or keyword options, not both")
        source = kwargs if config is None else config.model_dump()
        self.config = self.config_schema.model_validate(source)

    @abstractmethod
    def solve(self, model: Any, backend: Any) -> CAMSolverResult:
        """Solve the configured CAM task."""
