"""qphase_sde: Coherence-Frequency Plugin Class (2.0)
---------------------------------------------------------
Plugin-class protocol for coherence-frequency estimators. An estimator
consumes the typed inputs it declares (time series, spectrum, peak candidates
and/or peak paths) and produces one coherence-frequency estimate row of the
``COHERENCE_FREQUENCY_PRODUCT`` table from ``qphase_sde.contracts.coherence``,
with lag, bandwidth and model diagnostics as explicit fields.

Concrete implementations arrive in Wave 5B:
``coherence_frequency.short_delay`` (current ``analyser.coherence_carrier``),
``coherence_frequency.band_limited`` (current
``analyser.band_limited_carrier``) and ``coherence_frequency.finite_delay``
(current ``analyser.finite_delay_carrier``). This module declares the
plugin-class surface only and ships no concrete estimator.

Public API
----------
``BaseCoherenceFrequencyConfig`` : Base configuration for coherence-frequency
    estimators.
``CoherenceFrequencyProtocol`` : Abstract coherence-frequency plugin class.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from qphase.core.protocols import PluginConfigBase

from ..contracts.coherence import CoherenceFrequencyEstimate

__all__ = [
    "BaseCoherenceFrequencyConfig",
    "CoherenceFrequencyProtocol",
]


class BaseCoherenceFrequencyConfig(PluginConfigBase):
    """Base configuration for coherence-frequency estimators."""


class CoherenceFrequencyProtocol(ABC):
    """Abstract base class for coherence-frequency estimators."""

    name: ClassVar[str]
    description: ClassVar[str]
    config_schema: ClassVar[type[BaseCoherenceFrequencyConfig]] = (
        BaseCoherenceFrequencyConfig
    )
    #: Typed inputs the implementation consumes, named with the plugin-graph
    #: vocabulary: "time_series", "spectrum", "candidates", "paths".
    required_inputs: ClassVar[tuple[str, ...]]

    def __init__(
        self, config: BaseCoherenceFrequencyConfig | None = None, **kwargs: Any
    ):
        if config is None:
            config = self.config_schema(**kwargs)
        self.config = config

    @abstractmethod
    def estimate(self, inputs: Mapping[str, Any]) -> CoherenceFrequencyEstimate:
        """Estimate the coherence frequency from the declared typed inputs.

        Parameters
        ----------
        inputs : Mapping[str, Any]
            Materialized typed inputs keyed by the entries of
            ``required_inputs`` (time series, spectrum, peak candidates
            and/or peak paths, per implementation).

        Returns
        -------
        CoherenceFrequencyEstimate
            One estimate row of the coherence-frequency product, with
            explicit lag/bandwidth diagnostics and uncertainty scopes.

        """
        pass
