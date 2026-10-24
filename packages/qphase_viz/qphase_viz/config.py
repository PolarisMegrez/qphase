"""qphase_viz: Visualization Configuration
---------------------------------------------------------
Defines the configuration schemas for the visualization engine and individual plotters.

Public API
----------
`AnnotationSpec` : One text annotation placed on a figure.
`BasePlotterConfig` : Base configuration for all plotters.
`TimeSeriesSpec` : Specification for a single Time Series plot.
`TimeSeriesConfig` : Configuration for Time Series Plotter (list of specs).
`PhasePlaneSpec` : Specification for a single Phase Plane plot.
`PhasePlaneConfig` : Configuration for Phase Plane Plotter (list of specs).
`PowerSpectrumSpec` : Specification for a single Power Spectrum plot.
`PowerSpectrumConfig` : Configuration for Power Spectrum Plotter (list of specs).
`SpectralScanSpec` : Specification for one PSD-across-scan figure.
`SpectralScanConfig` : Configuration for the PSD scan plotter.
`HeatmapSpec` : Specification for one scalar heatmap figure.
`HeatmapContour` / `HeatmapSeries` : Heatmap overlays.
`HeatmapConfig` : Configuration for the heatmap plotter.
`DistributionSpec` : Specification for one sample-distribution figure.
`DistributionConfig` : Configuration for the distribution plotter.
`VizEngineConfig` : Configuration for the Visualization Engine.
"""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .styles import DEFAULT_STYLE, PALETTE, StyleName

#: Default overlay colour: the ``ink`` role of the semantic palette.
PALETTE_INK = PALETTE["ink"]


class AnnotationSpec(BaseModel):
    """One text annotation drawn on a figure.

    ``coords`` selects the coordinate frame of ``x``/``y``: ``"axes fraction"``
    is the Matplotlib axes frame (0..1 across the drawn area, independent of
    the data limits), ``"data"`` is the plotted data space. ``fontsize`` and
    ``color`` inherit the active style when left unset.
    """

    text: str = Field(..., description="Annotation text (mathtext allowed)")
    x: float = Field(..., description="Annotation x position")
    y: float = Field(..., description="Annotation y position")
    coords: Literal["axes fraction", "data"] = Field(
        "axes fraction",
        description="Coordinate frame of x/y: 'axes fraction' or 'data'",
    )
    fontsize: float | None = Field(
        None, gt=0.0, description="Font size; None inherits the active style"
    )
    color: str | None = Field(
        None, description="Text colour; None inherits the active style"
    )
    ha: Literal["left", "center", "right"] = Field(
        "left", description="Horizontal alignment"
    )
    va: Literal["top", "center", "bottom", "baseline"] = Field(
        "top", description="Vertical alignment"
    )
    rotation: float = Field(0.0, description="Rotation in degrees")


class BasePlotterConfig(BaseModel):
    """Base configuration for all plotters.

    Contains common matplotlib styling and output settings. ``style``,
    ``editable`` and ``style_overrides`` override the engine-level settings for
    this figure only; leaving them unset inherits the engine's choice.
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
    style: StyleName | None = Field(
        None, description="Figure style; None inherits the engine style"
    )
    editable: bool | None = Field(
        None, description="Editable text export; None inherits the engine setting"
    )
    style_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-figure Matplotlib overrides; applied after the style",
    )
    annotations: list[AnnotationSpec] = Field(
        default_factory=list, description="Text annotations drawn on this figure"
    )
    constrained_layout: bool = Field(
        False, description="Use Matplotlib's constrained layout for this figure"
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
    scale: Literal["linear", "log", "dB"] = Field("log", description="Y-axis scale")
    # Peak annotation
    annotate_peaks: bool = Field(False, description="Annotate peaks on plot")


class PowerSpectrumConfig(BaseModel):
    """Configuration for Power Spectrum Plotter.

    Contains a list of plot specifications.
    """

    plots: list[PowerSpectrumSpec] = Field(
        default_factory=list, description="List of power spectrum plots to generate"
    )


class ParameterEvolutionSpec(BasePlotterConfig):
    """Specification for Parameter Evolution Plot.

    Plots a metric (e.g. PSD peak frequency) vs a parameter.
    """

    kind: Literal["parameter_evolution"] = "parameter_evolution"
    parameter: str = Field(..., description="Name of the parameter to scan")
    metric: Literal["psd_peak_freq", "psd_peak_val", "mean", "variance"] = Field(
        "psd_peak_freq", description="Metric to plot"
    )
    channel: int = Field(0, description="Channel to analyze")

    # Curve fitting configuration
    fit_type: Literal[None, "power", "linear"] = Field(
        None,
        description="Type of curve fitting to apply (power: y=ax^b, linear: y=ax+b)",
    )


class ParameterEvolutionConfig(BaseModel):
    """Configuration for Parameter Evolution Plotter."""

    plots: list[ParameterEvolutionSpec] = Field(
        default_factory=list, description="List of evolution plots"
    )


class SpectralScanSpec(BasePlotterConfig):
    """Specification for a multi-panel PSD-across-scan figure.

    One panel per selected mode, all sharing the frequency axis: each panel
    draws the PSD of the chosen scan points in that mode, coloured by the scan
    coordinate. The scan coordinate is always explicit — the plotter never
    aggregates across scan points.
    """

    kind: Literal["spectral_scan"] = "spectral_scan"
    scan_parameter: str = Field(
        ...,
        description="Scan coordinate name to read from the PSD product "
        "(for example 'omega'); required, never inferred",
    )
    channels: list[int] = Field(
        default_factory=lambda: [0],
        description="Modes to draw, one panel each, top to bottom",
    )
    scan_indices: list[int] | None = Field(
        None, description="Scan points to draw, by flat index; None draws all"
    )
    scan_coordinates: list[float] | None = Field(
        None,
        description=(
            "Scan points to draw, by coordinate value; each value must land "
            "within scan_tolerance of a stored point"
        ),
    )
    scan_tolerance: float | None = Field(
        None,
        gt=0.0,
        description=(
            "Required with scan_coordinates: the largest accepted distance "
            "between a requested value and the nearest stored scan point. No "
            "default — a silent nearest-match would hide a mistyped target."
        ),
    )
    log_scale: bool = Field(True, description="Draw the PSD on a log y axis")
    cmap: str = Field(
        "scan",
        description="Curve colours: 'scan' (allan_scan_colormap), 'flow' "
        "(cam_flow_colormap), or any Matplotlib colormap name",
    )
    label_template: str = Field(
        "{name}={value:.6g}",
        description=(
            "Legend label for one scan point; Python str.format fields are "
            "'name' (scan parameter), 'value' (coordinate) and 'index'"
        ),
    )
    mode_label_template: str = Field(
        "Mode {mode}", description="Y label of one panel; format field 'mode'"
    )
    constrained_layout: bool = Field(
        True, description="Use constrained layout (multi-panel default)"
    )

    @model_validator(mode="after")
    def validate_scan_selection(self) -> Self:
        if self.scan_coordinates is not None and self.scan_tolerance is None:
            raise ValueError(
                "scan_coordinates requires an explicit scan_tolerance: a "
                "nearest-match without a bound would silently draw a different "
                "scan point than the one requested"
            )
        return self


class SpectralScanConfig(BaseModel):
    """Configuration for the PSD scan plotter."""

    plots: list[SpectralScanSpec] = Field(
        default_factory=list, description="List of PSD scan figures"
    )


class HeatmapContour(BaseModel):
    """Contour lines drawn over a heatmap.

    ``variable`` names a product variable to contour; ``None`` contours the
    boolean validity mask itself, which is the common way to outline the
    admissible region of a scan.
    """

    variable: str | None = Field(
        None, description="Product variable to contour; None contours the mask"
    )
    levels: list[float] = Field(
        default_factory=lambda: [0.5], description="Contour levels"
    )
    color: str = Field(PALETTE_INK, description="Line colour")
    linestyle: str = Field("--", description="Matplotlib line style")
    linewidth: float = Field(0.9, gt=0.0, description="Line width")


class HeatmapSeries(BaseModel):
    """An explicit point or line series drawn in heatmap data coordinates.

    The series comes either from product variables (``x_variable``/
    ``y_variable``) or from constant coordinates (``x``/``y``); exactly one
    source per axis is required. A pure marker series sets
    ``linestyle=None``.
    """

    x_variable: str | None = Field(None, description="Product variable for x")
    y_variable: str | None = Field(None, description="Product variable for y")
    x: list[float] = Field(default_factory=list, description="Constant x values")
    y: list[float] = Field(default_factory=list, description="Constant y values")
    color: str = Field(PALETTE_INK, description="Line and marker colour")
    linestyle: str | None = Field(
        "-", description="Line style; None draws markers only"
    )
    linewidth: float = Field(1.2, gt=0.0, description="Line width")
    marker: str | None = Field(None, description="Marker style; None draws a line only")
    markersize: float = Field(6.0, gt=0.0, description="Marker size")
    marker_face_color: str | None = Field(
        None, description="Marker face colour; None uses the line colour"
    )
    label: str | None = Field(None, description="Legend label")

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        for axis, variable, values in (
            ("x", self.x_variable, self.x),
            ("y", self.y_variable, self.y),
        ):
            if (variable is None) == (not values):
                raise ValueError(
                    f"heatmap series needs exactly one {axis} source: "
                    f"'{axis}_variable' or '{axis}'"
                )
        if self.x_variable is None and self.y_variable is None and len(self.x) != len(
            self.y
        ):
            raise ValueError("constant heatmap series needs equal x and y lengths")
        return self


class HeatmapSpec(BasePlotterConfig):
    """Specification for a two-dimensional scalar heatmap.

    ``x`` is the 1-D row coordinate and ``y`` the 1-D column coordinate of
    ``value[row, column]``; the plotter transposes once for Matplotlib, so the
    declared orientation and the rendered figure agree. A product may carry a
    boolean validity mask, which blanks disallowed cells instead of drawing
    them as zero.
    """

    kind: Literal["heatmap"] = "heatmap"
    product: str | None = Field(
        None, description="Product name to read from; None selects by fields"
    )
    x_variable: str = Field(..., description="1-D row coordinate variable")
    y_variable: str = Field(..., description="1-D column coordinate variable")
    value_variable: str = Field(..., description="2-D (row, column) value variable")
    valid_variable: str | None = Field(
        None, description="Boolean mask variable; False cells are left blank"
    )
    cmap: str = Field(
        "scan",
        description="'scan' (allan_scan_colormap), 'flow' (cam_flow_colormap) "
        "or any Matplotlib colormap name",
    )
    norm: Literal["linear", "log"] = Field("linear", description="Colour scaling")
    vmin: float | None = Field(None, description="Colour range minimum")
    vmax: float | None = Field(None, description="Colour range maximum")
    colorbar_label: str | None = Field(
        None, description="Colorbar label; None uses the value variable name"
    )
    rasterized: bool = Field(True, description="Rasterize the mesh on export")
    contours: list[HeatmapContour] = Field(
        default_factory=list, description="Contour overlays"
    )
    series: list[HeatmapSeries] = Field(
        default_factory=list, description="Point or line overlays"
    )
    constrained_layout: bool = Field(
        True, description="Use constrained layout (mesh plus colorbar default)"
    )


class HeatmapConfig(BaseModel):
    """Configuration for the heatmap plotter."""

    plots: list[HeatmapSpec] = Field(
        default_factory=list, description="List of heatmap figures"
    )


class DistributionSpec(BasePlotterConfig):
    """Specification for per-realization sample distributions.

    ``value`` holds one sample per realization, optionally per scan point
    (``value[scan, realization]``). Selecting more than one scan point draws
    one distribution per point; the samples are never pooled across points.

    ``center``/``scale`` compose the sample transform
    ``scale * value + center`` — the source of a two-series comparison is
    ``center: reference_mean`` with ``scale: -1`` (i.e. ``reference - value``),
    where the reference is the first selected series. The default is the
    identity, ``center: none`` with ``scale: 1``.
    """

    kind: Literal["distribution"] = "distribution"
    product: str | None = Field(
        None, description="Product name to read from; None selects by variable"
    )
    value_variable: str = Field(
        ..., description="Sample variable: [scan, realization] or [realization]"
    )
    scan_parameter: str | None = Field(
        None,
        description=(
            "Scan coordinate name; required when the value has a scan axis "
            "and points are selected by value"
        ),
    )
    scan_indices: list[int] | None = Field(
        None, description="Scan points to draw, by flat index; None draws all"
    )
    scan_coordinates: list[float] | None = Field(
        None, description="Scan points to draw, by coordinate value"
    )
    scan_tolerance: float | None = Field(
        None,
        gt=0.0,
        description=(
            "Required with scan_coordinates: the largest accepted distance "
            "between a requested value and the nearest stored scan point."
        ),
    )
    labels: list[str] | None = Field(
        None, description="Legend label per series; None draws no legend"
    )
    colors: list[str] | None = Field(
        None, description="Colour per series; None uses the semantic palette"
    )
    center: Literal["none", "reference_mean"] = Field(
        "none", description="Subtract nothing, or the first series' mean"
    )
    scale: float = Field(1.0, description="Non-zero scale applied after centring")
    bin_width: float | None = Field(
        None, gt=0.0, description="Fixed histogram bin width (needs xlim to match)"
    )
    bins: int | None = Field(
        None, ge=2, description="Histogram bin count (alternative to bin_width)"
    )
    show_means: bool = Field(False, description="Draw each series mean as a line")
    show_fwhm: bool = Field(
        False, description="Shade the half-maximum band and mark its bounds"
    )
    shift_label: str | None = Field(
        None,
        description=(
            "Annotate the mean-to-mean shift of the first two series; requires "
            "show_means and at least two series"
        ),
    )
    width_label: str | None = Field(
        None,
        description=(
            "Annotate the first series' FWHM; requires show_fwhm"
        ),
    )
    xticks: list[float] | None = Field(
        None, description="Explicit x tick positions"
    )
    xticklabels: list[str] | None = Field(
        None, description="Explicit x tick labels; one per tick"
    )
    hide_yticks: bool = Field(False, description="Hide the y tick labels and marks")

    @model_validator(mode="after")
    def validate_distribution(self) -> Self:
        if (self.bin_width is None) == (self.bins is None):
            raise ValueError(
                "set exactly one of 'bin_width' or 'bins' so the histogram "
                "edges are unambiguous"
            )
        if self.scale == 0.0:
            raise ValueError("'scale' must be non-zero")
        if self.scan_coordinates is not None and self.scan_tolerance is None:
            raise ValueError(
                "scan_coordinates requires an explicit scan_tolerance"
            )
        if self.shift_label is not None and not self.show_means:
            raise ValueError(
                "shift_label annotates the series means; enable 'show_means'"
            )
        if self.width_label is not None and not self.show_fwhm:
            raise ValueError(
                "width_label annotates the half-maximum width; enable 'show_fwhm'"
            )
        if self.xticklabels is not None and self.xticks is None:
            raise ValueError("xticklabels needs xticks")
        if (
            self.xticks is not None
            and self.xticklabels is not None
            and len(self.xticks) != len(self.xticklabels)
        ):
            raise ValueError("xticks and xticklabels must have the same length")
        for name, values in (("labels", self.labels), ("colors", self.colors)):
            if values is not None and not values:
                raise ValueError(f"'{name}' must not be empty; use None instead")
        return self


class DistributionConfig(BaseModel):
    """Configuration for the distribution plotter."""

    plots: list[DistributionSpec] = Field(
        default_factory=list, description="List of distribution figures"
    )


class VizEngineConfig(BaseModel):
    """Configuration for the Visualization Engine.

    ``style``/``editable`` set the figure-wide defaults; a plot spec may
    override either for its own figure. ``style_overrides`` is applied after
    the style table and the editable-text switch, so an explicit rc value
    always wins.
    """

    output_dir: str = Field(default=".", description="Directory to save figures")
    format: str = Field(default="png", description="Output format (png, pdf, svg)")
    specs: list[dict[str, Any]] = Field(
        default_factory=list, description="List of plot specifications"
    )
    style: StyleName = Field(
        DEFAULT_STYLE, description="Default figure style (publication, sm, sans)"
    )
    editable: bool = Field(
        True,
        description=(
            "Export text as editable text (svg fonttype 'none', pdf fonttype 42) "
            "instead of outlines"
        ),
    )
    style_overrides: dict[str, Any] = Field(
        default_factory=dict, description="Global matplotlib style overrides"
    )

    model_config = ConfigDict(extra="allow")
