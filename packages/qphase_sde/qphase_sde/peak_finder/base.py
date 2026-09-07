"""qphase_sde: Peak Finder Plugin Class
---------------------------------------------------------
Two contracts coexist during Wave 5:

- ``PeakFinder`` is the public legacy plugin class: ``find_peaks`` returns
  the legacy :class:`PeakInfo`, which the production PSD analyser consumes
  until the Wave 5B migration.
- :class:`TypedPeakFinder` is the frozen Wave 5B target contract: a runtime
  structural protocol whose finders return the grouped candidate-table
  payload of ``qphase_sde.contracts.peaks`` — exactly the mapping
  :func:`~qphase_sde.contracts.peaks.validate_candidate_table` accepts.

No concrete finder claims :class:`TypedPeakFinder` conformance yet.

Public API
----------
``PeakInfo`` : Legacy peak finding result container.
``BasePeakFinderConfig`` : Base configuration for peak finders.
``PeakFinder`` : Abstract legacy peak-finder plugin class.
``TypedPeakFinder`` : Frozen Wave 5B candidate-table protocol.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from qphase.core.protocols import PluginConfigBase

from ..contracts.peaks import CandidateTablePayload


class PeakInfo(BaseModel):
    """Container for peak finding results."""

    indices: list[int]
    frequencies: list[float]
    values: list[float]
    properties: dict[str, Any] = Field(default_factory=dict)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Serialize peak info, converting numpy/complex values to plain Python."""
        data = super().model_dump(**kwargs)
        data["properties"] = _serialize_peak_value(self.properties)
        return data


def _serialize_peak_value(value: Any) -> Any:
    """Recursively convert numpy arrays and complex numbers to JSON-safe types."""
    if isinstance(value, np.ndarray):
        return _serialize_peak_value(value.tolist())
    if isinstance(value, np.complexfloating | complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {k: _serialize_peak_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_serialize_peak_value(v) for v in value]
    if isinstance(value, np.floating | np.integer):
        return value.item()
    return value


class BasePeakFinderConfig(PluginConfigBase):
    """Base configuration for peak finders.

    Strict like the other plugin config bases in this repo (e.g. the CAM
    solver/postprocessor bases): unknown fields are rejected instead of
    being silently absorbed by the permissive ``PluginConfigBase`` default.
    """

    model_config = ConfigDict(extra="forbid")

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


@runtime_checkable
class TypedPeakFinder(Protocol):
    """Frozen Wave 5B target contract: unified peak-candidate finders.

    Implementations return the grouped candidate-table payload of
    ``qphase_sde.contracts.peaks`` — a :class:`CandidateTablePayload` mapping
    that must unpack into ``validate_candidate_table`` without raising.
    Until the Wave 5B migration, production consumers (the PSD analyser)
    keep using the legacy :class:`PeakFinder` contract; no concrete finder
    claims this protocol yet.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    config_schema: ClassVar[type[BasePeakFinderConfig]]

    def find_candidate_table(
        self,
        frequencies: np.ndarray,
        power: np.ndarray,
        *,
        scan_index: int,
        scan_count: int,
        channel_names: Sequence[str],
        mode_labels: Sequence[str] | None = None,
        channel_mode_weights: Sequence[Sequence[float]] | None = None,
    ) -> CandidateTablePayload:
        """Find peaks in one scan point's spectra and return the payload.

        Parameters
        ----------
        frequencies : np.ndarray
            Shared frequency axis.
        power : np.ndarray
            Power spectral density values, one row per channel.
        scan_index : int
            Scan point these spectra belong to (``group_scan_index``).
        scan_count : int
            Total number of scan points of the run.
        channel_names : Sequence[str]
            Stable channel labels of ``power``, echoed into the payload.
        mode_labels : Sequence[str] | None
            Modes the readout channels act on, when a measurement basis is
            declared.
        channel_mode_weights : Sequence[Sequence[float]] | None
            ``[re, im]`` readout weights flattened row-major over
            (channel x mode), echoed into the payload.

        Returns
        -------
        CandidateTablePayload
            The grouped candidate-table payload;
            ``qphase_sde.contracts.peaks.validate_candidate_table(**payload)``
            must not raise.

        """
        ...
