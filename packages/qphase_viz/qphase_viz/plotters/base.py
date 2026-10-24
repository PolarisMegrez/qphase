"""qphase_viz: Plotter Protocol
---------------------------------------------------------
Defines the interface for all plotters.

Public API
----------
``PlotterProtocol``
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from qphase.core.dataset import DatasetResultProtocol
from qphase.core.protocols import PluginBase

from ..styles import RenderOptions


@runtime_checkable
class PlotterProtocol(PluginBase, Protocol):
    """Protocol for visualization plotters."""

    def plot(
        self,
        data: DatasetResultProtocol,
        output_dir: Path,
        format: str,
        options: RenderOptions | None = None,
    ) -> list[Path]:
        """Render plots based on data and internal configuration.

        Parameters
        ----------
        data : DatasetResultProtocol
            A typed bundle exposing named Dataset products.
        output_dir : Path
            Directory to save the output file.
        format : str
            Output file format (e.g., 'png', 'pdf').
        options : RenderOptions | None
            Engine-resolved style. ``None`` renders with the ``publication``
            defaults, which is what a direct ``plotter.plot`` call gets.

        Returns
        -------
        list[Path]
            Paths to the generated files.

        """
        ...
