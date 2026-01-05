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
    analysis_results : dict[str, Any] | None
        Dictionary of analysis results (e.g. PSD data).

    """

    def __init__(
        self,
        generated_files: list[Path],
        analysis_results: dict[str, Any] | None = None,
    ):
        self._data = generated_files
        self._analysis = analysis_results or {}
        self._metadata = {
            "count": len(generated_files),
            "has_analysis": bool(self._analysis),
        }

    @property
    def data(self) -> list[Path]:
        """Get the list of generated files."""
        return self._data

    @property
    def analysis(self) -> dict[str, Any]:
        """Get the analysis results."""
        return self._analysis

    @property
    def metadata(self) -> dict[str, Any]:
        """Get the result metadata."""
        return self._metadata

    @property
    def label(self) -> Any:
        """Get the label from metadata."""
        return self._metadata.get("label")

    def save(self, path: str | Path) -> None:
        """Save the result.

        Saves analysis results if present.
        """
        if not self._analysis:
            return

        path = Path(path)
        # If path is a directory, save analysis there
        # If it's a file, maybe save a summary or zip?
        # For now, assume directory or create one
        if path.suffix:
            out_dir = path.parent
        else:
            out_dir = path
            out_dir.mkdir(parents=True, exist_ok=True)

        for name, res in self._analysis.items():
            # res should be AnalysisResult or similar
            if hasattr(res, "save"):
                res.save(out_dir / f"{name}_analysis.npz")


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
