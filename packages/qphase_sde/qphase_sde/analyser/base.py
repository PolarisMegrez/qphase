"""qphase_sde: Analyzer Base Class
------------------------------

Base class for all analyzers in the qphase_sde package.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol, runtime_checkable

from qphase.backend.base import BackendBase
from qphase.core.protocols import PluginBase, PluginConfigBase, ResultProtocol


@runtime_checkable
class AnalyzerProtocol(Protocol):
    """Protocol for analyzers."""

    def analyze(self, data: Any, backend: BackendBase) -> ResultProtocol: ...


class Analyzer(PluginBase, ABC):
    """Base class for analyzers.

    All analyzers must inherit from this class and implement the
    analyze method.
    """

    config_schema: ClassVar[type[PluginConfigBase]]

    def __init__(self, config: PluginConfigBase | None = None, **kwargs):
        """Initialize the analyzer.

        Parameters
        ----------
        config : PluginConfigBase, optional
            Configuration object. If None, created from kwargs.
        **kwargs : Any
            Configuration parameters if config is not provided.

        """
        if config is None:
            if hasattr(self, "config_schema"):
                self.config = self.config_schema(**kwargs)
            else:
                # Fallback for analyzers without specific config
                # This should ideally not happen if protocols are strictly followed
                pass
        else:
            self.config = config

    @abstractmethod
    def analyze(self, data: Any, backend: BackendBase) -> ResultProtocol:
        """Perform analysis on the data.

        Parameters
        ----------
        data : Any
            Input data for analysis.
        backend : BackendBase
            Backend to use for computation.

        Returns
        -------
        ResultProtocol
            Analysis results.

        """
        pass
