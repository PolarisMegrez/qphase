"""qphase_viz: Plotter Protocol
---------------------------

Defines the interface for all plotters.
"""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from qphase.backend.base import ArrayBase


@runtime_checkable
class PlotterProtocol(Protocol):
    """Protocol for visualization plotters."""

    def plot(
        self, data: ArrayBase, config: dict[str, Any], output_dir: Path, format: str
    ) -> Path:
        """Render a plot based on data and configuration.

        Parameters
        ----------
        data : ArrayBase
            The simulation data (TrajectorySet or similar).
        config : dict
            The validated configuration dictionary for this specific plot.
        output_dir : Path
            Directory to save the output file.
        format : str
            Output file format (e.g., 'png', 'pdf').

        Returns
        -------
        Path
            Path to the generated file.

        """
        ...
