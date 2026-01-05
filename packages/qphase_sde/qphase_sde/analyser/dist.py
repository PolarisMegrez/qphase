"""qphase_sde: Distribution Analyzer
---------------------------------------------------------
Compute phase space distribution (histogram) from multi-trajectory time series.

Public API
----------
``DistAnalyzer`` : Distribution analyzer.
``DistAnalyzerConfig`` : Configuration for Distribution analyzer.
"""

from typing import Any, ClassVar, cast

import numpy as _np
from pydantic import Field, model_validator
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy
from qphase.core.protocols import PluginConfigBase

from .base import Analyzer
from .result import AnalysisResult

__all__ = [
    "DistAnalyzer",
    "DistAnalyzerConfig",
]


class DistAnalyzerConfig(PluginConfigBase):
    """Configuration for Distribution Analyzer."""

    modes: list[int] = Field(..., description="Mode indices for analysis")
    bins: int = Field(50, description="Number of bins for histogram")
    range: list[list[float]] | None = Field(
        None, description="Range for each mode [[min, max], ...]"
    )
    density: bool = Field(True, description="Normalize histogram to form a PDF")

    @model_validator(mode="after")
    def validate_modes(self) -> "DistAnalyzerConfig":
        if not self.modes:
            raise ValueError("modes must be non-empty")
        return self


class DistAnalyzer(Analyzer):
    """Analyzer for Phase Space Distribution."""

    name: ClassVar[str] = "dist"
    description: ClassVar[str] = "Phase Space Distribution analyzer"
    config_schema: ClassVar[type[DistAnalyzerConfig]] = DistAnalyzerConfig  # type: ignore[assignment]

    def __init__(self, config: DistAnalyzerConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)  # type: ignore[arg-type]

    def analyze(self, data: Any, backend: BackendBase) -> AnalysisResult:
        """Compute distribution for multiple modes.

        Parameters
        ----------
        data : Any
            Complex-like time series array of shape ``(n_traj, n_time, n_modes)``
            or TrajectorySet.
        backend : BackendBase
            Backend to use for computation.

        Returns
        -------
        AnalysisResult
            Result containing distribution data.

        """
        config = cast(DistAnalyzerConfig, self.config)
        modes = config.modes
        bins = config.bins
        density = config.density
        range_list = config.range

        # Extract data array
        if hasattr(data, "data"):
            data_arr = data.data
        else:
            data_arr = data

        # Convert to numpy for histogram (CPU-bound, numpy is good enough)
        # If data is on GPU, we might want to use backend specific histogram
        # For now, let's pull to CPU to be safe and consistent
        data_np = convert_to_numpy(data_arr)

        results = {}

        for i, m in enumerate(modes):
            # Flatten trajectories and time
            # data_np shape: (n_traj, n_time, n_modes)
            samples = data_np[:, :, m].flatten()

            # Handle complex data: separate real and imag or magnitude?
            # Usually phase space is Re vs Im.
            # But here we are doing 1D distribution of the complex value?
            # Or 2D distribution of Re/Im?
            # Let's assume 2D distribution of Re/Im for complex modes

            if _np.iscomplexobj(samples):
                # 2D Histogram
                x = samples.real
                y = samples.imag

                curr_range = None
                if range_list and i < len(range_list):
                    # Expecting [[xmin, xmax], [ymin, ymax]] for this mode?
                    # Or just [min, max] for both?
                    # Let's simplify: range_list[i] is [min, max] for both axes
                    r = range_list[i]
                    curr_range = [r, r]

                H, xedges, yedges = _np.histogram2d(
                    x, y, bins=bins, range=curr_range, density=density
                )
                results[m] = {
                    "hist": H,
                    "xedges": xedges,
                    "yedges": yedges,
                    "type": "2d_complex",
                }
            else:
                # 1D Histogram
                curr_range = None
                if range_list and i < len(range_list):
                    curr_range = range_list[i]

                H, edges = _np.histogram(
                    samples, bins=bins, range=curr_range, density=density
                )
                results[m] = {"hist": H, "edges": edges, "type": "1d_real"}

        result_dict = {
            "distributions": results,
            "modes": modes,
            "bins": bins,
            "density": density,
        }

        return AnalysisResult(data_dict=result_dict, meta=result_dict)
