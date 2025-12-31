"""qphase_viz: Visualization Engine
---------------------------------------------------------
Implements the EngineBase protocol for visualization tasks.

Public API
----------
``VizEngine``, ``VizResult``
"""

from pathlib import Path
from typing import Any, ClassVar

import matplotlib.pyplot as plt
from qphase.core.errors import QPhaseRuntimeError
from qphase.core.protocols import EngineBase, ResultProtocol

from .config import VizEngineConfig
from .plotters.base import PlotterProtocol


def _set_default_rcparams() -> None:
    """Apply project-wide default Matplotlib font/mathtext settings."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "sans"
    plt.rcParams["mathtext.it"] = "sans:italic"
    plt.rcParams["mathtext.bf"] = "sans:bold"
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["ytick.labelsize"] = 15


class VizResult(ResultProtocol):
    """Result container for visualization engine.

    Parameters
    ----------
    generated_files : list[Path]
        List of paths to generated plot files.

    """

    def __init__(self, generated_files: list[Path]):
        self._data = generated_files
        self._metadata = {"count": len(generated_files)}

    @property
    def data(self) -> list[Path]:
        """Get the list of generated files."""
        return self._data

    @property
    def metadata(self) -> dict[str, Any]:
        """Get the result metadata."""
        return self._metadata

    def save(self, path: str | Path) -> None:
        """Save the result (No-op).

        Visualization results are already saved files.
        This method might save a manifest or index in the future.
        """
        # Visualization results are already saved files.
        # This method might save a manifest or index.
        pass


class VizEngine(EngineBase):
    """Visualization Engine.

    Orchestrates the rendering of plots based on configuration specs.
    """

    name: ClassVar[str] = "viz"
    description: ClassVar[str] = "Visualization Engine"
    config_schema: ClassVar[type[VizEngineConfig]] = VizEngineConfig

    def __init__(
        self,
        config: VizEngineConfig | None = None,
        plugins: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.config = config or VizEngineConfig()
        self.plugins = plugins or {}

    def run(
        self,
        data: Any | None = None,
        *,
        progress_cb: Any | None = None,
    ) -> VizResult:
        """Execute visualization tasks.

        Parameters
        ----------
        data : Any
            Input data, expected to be an ArrayBase (e.g., TrajectorySet).
        progress_cb : Any | None
            Optional callback for progress updates.

        """
        if data is None:
            raise QPhaseRuntimeError("VizEngine requires input data.")

        # Apply global styles
        _set_default_rcparams()
        if self.config.style_overrides:
            plt.rcParams.update(self.config.style_overrides)

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []

        # Filter for visualizer plugins
        visualizers = [
            p for p in self.plugins.values() if isinstance(p, PlotterProtocol)
        ]

        total_plugins = len(visualizers)

        for i, plotter in enumerate(visualizers):
            try:
                # Execute plot
                # The plotter is already configured via its own config
                out_paths = plotter.plot(data, output_dir, self.config.format)
                generated_files.extend(out_paths)
            except Exception as e:
                raise QPhaseRuntimeError(
                    f"Plotting failed for '{plotter.name}': {e}"
                ) from e

            # Report progress
            if progress_cb:
                percent = (i + 1) / total_plugins
                progress_cb(percent, None, f"Ran {plotter.name}", "rendering")

        return VizResult(generated_files)
