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

from ..utils import resolve_mode_columns
from .base import Analyzer
from .result import AnalysisResult

__all__ = [
    "DistAnalyzer",
    "DistAnalyzerConfig",
]


def _is_complex(arr: Any) -> bool:
    """Return True if ``arr`` has a complex dtype, backend-agnostic."""
    dtype = getattr(arr, "dtype", None)
    if dtype is None:
        return False
    return str(getattr(dtype, "kind", dtype)) in {"c", "complex"}


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
    config_schema: ClassVar[type[DistAnalyzerConfig]] = DistAnalyzerConfig

    def __init__(self, config: DistAnalyzerConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

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
        mode_columns = resolve_mode_columns(data, modes)
        bins = config.bins
        density = config.density
        range_list = config.range

        # Extract data array. Some array-like objects (e.g. TrajectorySet) wrap
        # the actual array in a ``.data`` attribute. NumPy/CuPy arrays also
        # expose a ``.data`` attribute, but it is a memoryview/MemoryPointer,
        # not an array. If ``data`` itself is already array-like (has ndim),
        # use it directly; otherwise unwrap ``data.data``.
        if hasattr(data, "ndim") and hasattr(data, "shape"):
            data_arr = data
        elif (
            hasattr(data, "data")
            and hasattr(data.data, "ndim")
            and hasattr(data.data, "shape")
        ):
            data_arr = data.data
        else:
            data_arr = data

        # Use backend-native histogram when available to avoid pulling the full
        # trajectory to CPU on GPU backends. Fall back to numpy for backends that
        # do not implement histogram/histogram2d.
        use_backend_hist = hasattr(backend, "histogram") and hasattr(
            backend, "histogram2d"
        )

        if not use_backend_hist:
            data_np = convert_to_numpy(data_arr)

        results = {}

        for i, (m, column) in enumerate(zip(modes, mode_columns, strict=True)):
            # Flatten trajectories and time
            # data shape: (n_traj, n_time, n_modes)
            if use_backend_hist:
                samples = data_arr[:, :, column].reshape(-1)
            else:
                samples = data_np[:, :, column].flatten()

            is_complex = (
                _is_complex(samples) if use_backend_hist else _np.iscomplexobj(samples)
            )
            if is_complex:
                # 2D Histogram
                if use_backend_hist:
                    x = backend.real(samples)
                    y = backend.imag(samples)
                else:
                    x = samples.real
                    y = samples.imag

                hist_range_2d: list[tuple[float, float]] | None = None
                if range_list and i < len(range_list):
                    r = cast(tuple[float, float], tuple(range_list[i]))
                    hist_range_2d = [r, r]

                if use_backend_hist:
                    H, xedges, yedges = backend.histogram2d(
                        x, y, bins=bins, range=hist_range_2d, density=density
                    )
                    H = convert_to_numpy(H)
                    xedges = convert_to_numpy(xedges)
                    yedges = convert_to_numpy(yedges)
                else:
                    H, xedges, yedges = _np.histogram2d(
                        x, y, bins=bins, range=hist_range_2d, density=density
                    )
                results[m] = {
                    "hist": H,
                    "xedges": xedges,
                    "yedges": yedges,
                    "type": "2d_complex",
                }
            else:
                # 1D Histogram
                hist_range_1d: tuple[float, float] | None = None
                if range_list and i < len(range_list):
                    hist_range_1d = cast(tuple[float, float], tuple(range_list[i]))

                if use_backend_hist:
                    H, edges = backend.histogram(
                        samples, bins=bins, range=hist_range_1d, density=density
                    )
                    H = convert_to_numpy(H)
                    edges = convert_to_numpy(edges)
                else:
                    H, edges = _np.histogram(
                        samples, bins=bins, range=hist_range_1d, density=density
                    )
                results[m] = {"hist": H, "edges": edges, "type": "1d_real"}

        result_dict = {
            "distributions": results,
            "modes": modes,
            "bins": bins,
            "density": density,
        }

        return AnalysisResult(data_dict=result_dict, meta=result_dict)
