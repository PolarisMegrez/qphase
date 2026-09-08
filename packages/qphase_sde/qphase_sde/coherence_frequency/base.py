"""qphase_sde: Coherence-Frequency Plugin Class (2.0)
---------------------------------------------------------
Plugin-class protocol for coherence-frequency estimators. An estimator
consumes the typed inputs it declares (time series, spectrum, peak candidates
and/or peak paths) and produces one coherence-frequency estimate row of the
``COHERENCE_FREQUENCY_PRODUCT`` table from ``qphase_sde.contracts.coherence``,
with lag, bandwidth and model diagnostics as explicit fields.

Concrete child plugins live in ``short_delay``, ``band_limited`` and
``finite_delay``; this module only declares their shared plugin-class surface.

Public API
----------
``BaseCoherenceFrequencyConfig`` : Base configuration for coherence-frequency
    estimators.
``CoherenceFrequencyProtocol`` : Abstract coherence-frequency plugin class.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import ConfigDict
from qphase.core.protocols import PluginConfigBase

from ..contracts.coherence import CoherenceFrequencyEstimate

__all__ = [
    "BaseCoherenceFrequencyConfig",
    "CoherenceFrequencyProtocol",
]


class BaseCoherenceFrequencyConfig(PluginConfigBase):
    """Base configuration for coherence-frequency estimators.

    Strict like the other plugin config bases in this repo (e.g. the CAM
    solver/postprocessor bases): unknown fields are rejected instead of
    being silently absorbed by the permissive ``PluginConfigBase`` default.
    """

    model_config = ConfigDict(extra="forbid")


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
