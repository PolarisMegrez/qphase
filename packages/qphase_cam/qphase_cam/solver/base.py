"""Public base class for CAM solver plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import ConfigDict
from qphase.core.protocols import PluginConfigBase

from qphase_cam.state import CAMSolverOutput


class CAMSolverConfig(PluginConfigBase):
    """Strict base schema for CAM solver configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class CAMSolver(ABC):
    """Base class for solver plugins selected by the CAM engine."""

    name: ClassVar[str]
    description: ClassVar[str]
    config_schema: ClassVar[type[CAMSolverConfig]]
    supports_batch: ClassVar[bool] = False

    def __init__(
        self, config: CAMSolverConfig | None = None, **kwargs: Any
    ) -> None:
        if config is not None and kwargs:
            raise TypeError("provide either config or keyword options, not both")
        source = kwargs if config is None else config.model_dump()
        self.config = self.config_schema.model_validate(source)

    @abstractmethod
    def solve(self, model: Any, backend: Any) -> CAMSolverOutput:
        """Solve the configured CAM task."""
