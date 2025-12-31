"""qphase_viz: Plotter Protocol
---------------------------------------------------------
Defines the interface for all plotters.

Public API
----------
``PlotterProtocol``
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from qphase.backend.base import ArrayBase
from qphase.core.protocols import PluginBase


@runtime_checkable
class PlotterProtocol(PluginBase, Protocol):
    """Protocol for visualization plotters."""

    def plot(self, data: ArrayBase, output_dir: Path, format: str) -> list[Path]:
        """Render plots based on data and internal configuration.

        Parameters
        ----------
        data : ArrayBase
            The simulation data (TrajectorySet or similar).
        output_dir : Path
            Directory to save the output file.
        format : str
            Output file format (e.g., 'png', 'pdf').

        Returns
        -------
        list[Path]
            Paths to the generated files.

        """
        ...
