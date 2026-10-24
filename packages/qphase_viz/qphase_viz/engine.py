"""qphase_viz: Visualization Engine
---------------------------------------------------------
Implements the EngineBase protocol for visualization tasks.

Public API
----------
``VizEngine``, ``VizResult``
"""

from pathlib import Path
from typing import Any, ClassVar

from qphase.core.dataset import DatasetResultProtocol
from qphase.core.errors import QPhaseRuntimeError
from qphase.core.protocols import EngineBase, EngineManifest, ResultProtocol

from .config import VizEngineConfig
from .plotters.base import PlotterProtocol
from .styles import RenderOptions


class VizResult(ResultProtocol):
    """Result container for visualization engine.

    A visualization job produces files, so the result is the list of paths the
    plotters wrote. There is no in-memory analysis payload and no second
    serialization: the engine already persisted every figure under the
    configured ``output_dir``.

    Parameters
    ----------
    generated_files : list[Path]
        Paths to the plot files this run produced.

    """

    def __init__(self, generated_files: list[Path]):
        self._data = generated_files
        self._metadata = {"count": len(generated_files)}

    @property
    def data(self) -> list[Path]:
        """Paths of the files written by this run."""
        return self._data

    @property
    def metadata(self) -> dict[str, Any]:
        """Result metadata: the number of generated files."""
        return self._metadata

    @property
    def label(self) -> Any:
        """Optional label from the metadata."""
        return self._metadata.get("label")

    def save(self, path: str | Path) -> None:
        """Do nothing: the figures are already on disk.

        ``ResultProtocol`` requires ``save``, but writing anything here would
        duplicate the engine's output model (a run persists its figures under
        ``VizEngineConfig.output_dir``) or invent a second, untyped payload.
        The parameter is ignored on purpose.
        """
        del path


class VizEngine(EngineBase):
    """Visualization Engine.

    Resolves the figure style once and hands every configured plotter the
    input bundle; the plotters own what each figure draws.
    """

    name: ClassVar[str] = "viz"
    description: ClassVar[str] = "Visualization Engine"
    config_schema: ClassVar[type[VizEngineConfig]] = VizEngineConfig
    manifest: ClassVar[EngineManifest] = EngineManifest(required_plugins={"visualizer"})

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
        context: Any | None = None,
    ) -> VizResult:
        """Execute visualization tasks.

        Parameters
        ----------
        data : DatasetResultProtocol
            Typed input bundle containing products selected by each plotter.
        context : Any | None
            Structured execution services supplied by the scheduler.

        """
        if data is None:
            raise QPhaseRuntimeError("VizEngine requires input data.")
        if not isinstance(data, DatasetResultProtocol):
            raise QPhaseRuntimeError(
                "VizEngine requires a typed DatasetResultProtocol input."
            )

        # Resolve the engine-wide style once; each plotter layers its own
        # spec settings on top and renders inside its own rc context.
        options = RenderOptions(
            style=self.config.style,
            editable=self.config.editable,
            overrides=self.config.style_overrides,
        )

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []
        reporter = context.progress if context is not None else None

        # Filter for visualizer plugins
        visualizers = [
            p for p in self.plugins.values() if isinstance(p, PlotterProtocol)
        ]

        total_plugins = len(visualizers)

        for i, plotter in enumerate(visualizers):
            try:
                # Execute plot
                # The plotter is already configured via its own config
                out_paths = plotter.plot(
                    data, output_dir, self.config.format, options
                )
                generated_files.extend(out_paths)
            except Exception as e:
                raise QPhaseRuntimeError(
                    f"Plotting failed for '{plotter.name}': {e}"
                ) from e

            # Report progress
            if reporter is not None:
                reporter.update(
                    completed=i + 1,
                    total=total_plugins,
                    unit="plot",
                    stage="rendering",
                    message=f"Ran {plotter.name}",
                )

        return VizResult(generated_files)
