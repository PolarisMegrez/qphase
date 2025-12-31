"""qphase_viz: Visualization Configuration
---------------------------------------------------------
Defines the configuration schemas for the visualization engine and individual plotters.

Public API
----------
`BasePlotterConfig` : Base configuration for all plotters.
`TimeSeriesSpec` : Specification for a single Time Series plot.
`TimeSeriesConfig` : Configuration for Time Series Plotter (list of specs).
`PhasePlaneSpec` : Specification for a single Phase Plane plot.
`PhasePlaneConfig` : Configuration for Phase Plane Plotter (list of specs).
`PowerSpectrumSpec` : Specification for a single Power Spectrum plot.
`PowerSpectrumConfig` : Configuration for Power Spectrum Plotter (list of specs).
`VizEngineConfig` : Configuration for the Visualization Engine.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BasePlotterConfig(BaseModel):
    """Base configuration for all plotters.

    Contains common matplotlib styling and output settings.
    """

    title: str | None = Field(None, description="Figure title")
    xlabel: str | None = Field(None, description="X-axis label")
    ylabel: str | None = Field(None, description="Y-axis label")
    xlim: tuple[float, float] | None = Field(None, description="X-axis limits")
    ylim: tuple[float, float] | None = Field(None, description="Y-axis limits")
    grid: bool = Field(True, description="Show grid")
    figsize: tuple[float, float] = Field((6.0, 4.0), description="Figure size (w, h)")
    dpi: int = Field(100, description="Figure DPI")
    filename: str | None = Field(
        None, description="Output filename (without extension)"
    )

    model_config = ConfigDict(extra="allow")


class TimeSeriesSpec(BasePlotterConfig):
    """Specification for a single Time Series Plot.

    Plots state variables vs time.
    """

    kind: Literal["time_series"] = "time_series"
    channels: list[int] = Field(
        default_factory=lambda: [0], description="Channels/Modes to plot"
    )
    transform: Literal["real", "imag", "abs", "angle", "number"] = Field(
        "real", description="Data transformation"
    )
    trajectories: int | Literal["mean", "all"] = Field(
        "mean", description="Trajectory selection: int (index), 'mean', or 'all'"
    )
    legend: bool = Field(True, description="Show legend")


class TimeSeriesConfig(BaseModel):
    """Configuration for Time Series Plotter.

    Contains a list of plot specifications.
    """

    plots: list[TimeSeriesSpec] = Field(
        default_factory=list, description="List of time series plots to generate"
    )


class PhasePlaneSpec(BasePlotterConfig):
    """Specification for a single Phase Plane Plot.

    Plots Im(y) vs Re(y) or y_j vs y_i.
    """

    kind: Literal["phase_plane"] = "phase_plane"
    channel_x: int = Field(0, description="Channel for X-axis (or Re part)")
    channel_y: int | None = Field(
        None, description="Channel for Y-axis (if None, uses Im part of channel_x)"
    )
    mode: Literal["scatter", "hist2d", "kde"] = Field(
        "hist2d", description="Plotting mode"
    )
    bins: int = Field(50, description="Number of bins for hist2d")
    cmap: str = Field("viridis", description="Colormap")


class PhasePlaneConfig(BaseModel):
    """Configuration for Phase Plane Plotter.

    Contains a list of plot specifications.
    """

    plots: list[PhasePlaneSpec] = Field(
        default_factory=list, description="List of phase plane plots to generate"
    )


class PowerSpectrumSpec(BasePlotterConfig):
    """Specification for a single Power Spectrum Plot.

    Plots Power Spectral Density (PSD).
    """

    kind: Literal["power_spectrum"] = "power_spectrum"
    channels: list[int] = Field(
        default_factory=lambda: [0], description="Channels to analyze"
    )
    window: Literal["hann", "hamming", "blackman"] | None = Field(
        "hann", description="Window function"
    )
    scale: Literal["linear", "log", "dB"] = Field("log", description="Y-axis scale")
    detrend: bool = Field(True, description="Detrend data before FFT")
    nperseg: int | None = Field(
        None, description="Length of each segment for Welch's method"
    )


class PowerSpectrumConfig(BaseModel):
    """Configuration for Power Spectrum Plotter.

    Contains a list of plot specifications.
    """

    plots: list[PowerSpectrumSpec] = Field(
        default_factory=list, description="List of power spectrum plots to generate"
    )


class VizEngineConfig(BaseModel):
    """Configuration for the Visualization Engine."""

    output_dir: str = Field(default=".", description="Directory to save figures")
    format: str = Field(default="png", description="Output format (png, pdf, svg)")
    specs: list[dict[str, Any]] = Field(
        default_factory=list, description="List of plot specifications"
    )
    style_overrides: dict[str, Any] = Field(
        default_factory=dict, description="Global matplotlib style overrides"
    )

    model_config = ConfigDict(extra="allow")
