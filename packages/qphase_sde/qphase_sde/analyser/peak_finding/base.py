from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field
from qphase.core.protocols import PluginConfigBase


class PeakInfo(BaseModel):
    """Container for peak finding results."""

    indices: list[int]
    frequencies: list[float]
    values: list[float]
    properties: dict[str, Any] = Field(default_factory=dict)


class BasePeakFinderConfig(PluginConfigBase):
    """Base configuration for peak finders."""

    method: str


class PeakFinder(ABC):
    """Abstract base class for peak finding algorithms."""

    name: ClassVar[str]
    description: ClassVar[str]
    config_schema: ClassVar[type[BasePeakFinderConfig]] = BasePeakFinderConfig

    def __init__(self, config: BasePeakFinderConfig | None = None, **kwargs: Any):
        if config is None:
            config = self.config_schema(**kwargs)
        self.config = config

    @abstractmethod
    def find_peaks(self, freqs: np.ndarray, psd: np.ndarray) -> PeakInfo:
        """Find peaks in power spectral density data.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency axis.
        psd : np.ndarray
            Power spectral density values.

        Returns
        -------
        PeakInfo
            Detected peaks and their properties.

        """
        pass
