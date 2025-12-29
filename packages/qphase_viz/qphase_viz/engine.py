"""qphase_viz: Visualization Engine
---------------------------------

Implements the EngineBase protocol for visualization tasks.
"""

from pathlib import Path
from typing import Any, ClassVar

import matplotlib.pyplot as plt
from qphase.core.errors import QPhaseConfigError, QPhaseRuntimeError
from qphase.core.protocols import EngineBase, ResultProtocol

from .config import (
    BasePlotterConfig,
    PhasePlaneConfig,
    PowerSpectrumConfig,
    TimeSeriesConfig,
    VizEngineConfig,
)
from .plotters.base import PlotterProtocol
from .plotters.evolution import TimeSeriesPlotter
from .plotters.phase import PhasePlanePlotter
from .plotters.spectrum import PowerSpectrumPlotter


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
    """Result container for visualization engine."""

    def __init__(self, generated_files: list[Path]):
        self._data = generated_files
        self._metadata = {"count": len(generated_files)}

    @property
    def data(self) -> list[Path]:
        return self._data

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def save(self, path: str | Path) -> None:
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

        # Ensure data is ArrayBase-compatible or a dict (for pre-computed analysis)
        # if not hasattr(data, "data") or not hasattr(data, "to_numpy"):
        #      # Try to wrap or cast? For now, assume it's compatible.
        #      pass

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []
        total_specs = len(self.config.specs)

        for i, spec in enumerate(self.config.specs):
            kind = spec.get("kind")
            if not kind:
                continue

            # Dispatch to plotter
            plotter: PlotterProtocol | None = None
            validated_config: dict[str, Any] = {}

            try:
                if kind == "time_series":
                    cfg: BasePlotterConfig = TimeSeriesConfig.model_validate(spec)
                    validated_config = cfg.model_dump()
                    plotter = TimeSeriesPlotter()
                elif kind == "phase_plane":
                    cfg = PhasePlaneConfig.model_validate(spec)
                    validated_config = cfg.model_dump()
                    plotter = PhasePlanePlotter()
                elif kind == "power_spectrum":
                    cfg = PowerSpectrumConfig.model_validate(spec)
                    validated_config = cfg.model_dump()
                    plotter = PowerSpectrumPlotter()
                else:
                    # Unknown plotter kind
                    continue
            except Exception as e:
                raise QPhaseConfigError(f"Invalid spec for '{kind}': {e}") from e

            if plotter:
                # Execute plot
                try:
                    out_path = plotter.plot(
                        data, validated_config, output_dir, self.config.format
                    )
                    generated_files.append(out_path)
                except Exception as e:
                    raise QPhaseRuntimeError(
                        f"Plotting failed for '{kind}': {e}"
                    ) from e

            # Report progress
            if progress_cb:
                percent = (i + 1) / total_specs
                progress_cb(percent, None, f"Rendered {kind}", "rendering")

        return VizResult(generated_files)
