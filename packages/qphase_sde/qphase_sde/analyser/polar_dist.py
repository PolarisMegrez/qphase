"""qphase_sde: Polar Distribution Analyzer
---------------------------------------------------------
Compute longitudinal (magnitude) distribution from multi-trajectory time series.
Includes peak finding and width estimation for the distribution.

Public API
----------
``PolarDistAnalyzer`` : Polar Distribution analyzer.
``PolarDistAnalyzerConfig`` : Configuration for Polar Dist analyzer.
"""

from typing import Any, ClassVar, Literal, cast

import numpy as np
from pydantic import Field, model_validator
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy
from qphase.core.protocols import PluginConfigBase

from .base import Analyzer
from .result import AnalysisResult

__all__ = [
    "PolarDistAnalyzer",
    "PolarDistAnalyzerConfig",
]


class PolarDistAnalyzerConfig(PluginConfigBase):
    """Configuration for Polar Distribution Analyzer.

    Analyzes the distribution of the magnitude (r = |z|) of the modes.
    """

    modes: list[int] = Field(..., description="Mode indices for analysis")
    bins: (
        int
        | Literal["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]
    ) = Field("auto", description="Number of bins or method string for histogram")
    range: list[tuple[float, float]] | None = Field(
        None,
        description="Range for each mode. If None, auto-calculated from data.",
    )
    density: bool = Field(True, description="Normalize histogram to form a PDF")

    @model_validator(mode="after")
    def validate_modes(self) -> "PolarDistAnalyzerConfig":
        if not self.modes:
            raise ValueError("modes must be non-empty")
        return self


class PolarDistAnalyzer(Analyzer):
    """Analyzer for Polar (Longitudinal/Magnitude) Distribution."""

    name: ClassVar[str] = "polar_dist"
    description: ClassVar[str] = "Magnitude Distribution Analyzer with Peak Finding"
    config_schema: ClassVar[type[PolarDistAnalyzerConfig]] = PolarDistAnalyzerConfig

    def __init__(self, config: PolarDistAnalyzerConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

    def analyze(self, data: Any, backend: BackendBase) -> AnalysisResult:
        """Compute magnitude distribution for multiple modes.

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
        config = cast(PolarDistAnalyzerConfig, self.config)
        modes = config.modes
        bins = config.bins
        density = config.density
        range_list = config.range

        # Extract data array
        if hasattr(data, "data"):
            data_arr = data.data
        else:
            data_arr = data

        # Convert to numpy for histogram and stats
        data_np = convert_to_numpy(data_arr)

        results = {}

        for i, m in enumerate(modes):
            # Extract mode data
            # data_np shape: (n_traj, n_time, n_modes)
            # Flatten to 1D array of samples
            raw_samples = data_np[:, :, m].flatten()

            # Calculate Magnitude (r)
            samples = np.abs(raw_samples)

            # Determine range
            curr_range = None
            if range_list and i < len(range_list):
                curr_range = range_list[i]

            # Calculate Histogram
            # bins can be int or string (e.g., 'auto')
            H, edges = np.histogram(
                samples, bins=bins, range=curr_range, density=density
            )

            # Calculate Bin Centers for peak finding
            bin_centers = (edges[:-1] + edges[1:]) / 2

            # Calculate Mean of the variable
            mean_val = np.mean(samples)
            std_val = np.std(samples)

            mode_result = {
                "hist": H,
                "edges": edges,
                "bin_centers": bin_centers,
                "mean": mean_val,
                "std": std_val,
            }

            results[m] = mode_result

        result_dict = {
            "distributions": results,
            "modes": modes,
            "bins_config": bins,
            "density": density,
        }

        return AnalysisResult(data_dict=result_dict, meta=result_dict)
