"""qphase_sde: Analyzer Base Class
------------------------------

Base class for all analyzers in the qphase_sde package.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from qphase.core.protocols import PluginBase, PluginConfigBase


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
    def analyze(self, data: Any, **kwargs) -> Any:
        """Perform analysis on the data.

        Parameters
        ----------
        data : Any
            Input data for analysis.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        Any
            Analysis results.

        """
        pass
