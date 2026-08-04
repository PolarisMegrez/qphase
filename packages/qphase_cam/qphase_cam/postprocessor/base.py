"""Public base class for CAM postprocessor plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import ConfigDict
from qphase.core.protocols import PluginConfigBase


class CAMPostprocessorConfig(PluginConfigBase):
    """Strict postprocessor configuration base."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class CAMPostprocessor(ABC):
    """Base class for computations applied after CAM solving."""

    name: ClassVar[str]
    description: ClassVar[str]
    config_schema: ClassVar[type[CAMPostprocessorConfig]]
    accepted_result_kinds: ClassVar[frozenset[str]] = frozenset({"fixed_points"})

    def __init__(
        self, config: CAMPostprocessorConfig | None = None, **kwargs: Any
    ) -> None:
        if config is not None and kwargs:
            raise TypeError("provide either config or keyword options, not both")
        source = kwargs if config is None else config.model_dump()
        self.config = self.config_schema.model_validate(source)
        self.result_metadata: dict[str, Any] = {}

    @abstractmethod
    def process(self, result: Any, model: Any, backend: Any) -> dict[str, Any]:
        """Return named arrays to merge into ``CAMResult.postprocess``."""
