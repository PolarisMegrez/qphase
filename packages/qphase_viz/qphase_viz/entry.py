"""qphase_viz: Entry Point
----------------------

Entry point for qphase_viz resource package.
"""

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError
from qphase_sde.analyser.psd import PsdAnalyzer, PsdAnalyzerConfig
from qphase_sde.core.errors import QPSConfigError, get_logger

from .plotters.phase_plane import PhasePortraitPlotter, PhasePortraitSpec
from .plotters.psd import PsdPlotter, PsdSpec
from .utils import _time_to_index, spec_short_hash


class VizEngineConfig(BaseModel):
    """Configuration schema for the VizEngine.

    Parameters
    ----------
    output_dir : str | Path, optional
        Directory to save rendered figures (default: current directory).
    specs : list[dict], optional
        List of visualization specifications to render.
    style_overrides : dict, optional
        Style overrides for matplotlib plots.

    """

    output_dir: str | Path = Field(
        ".", description="Directory to save rendered figures"
    )
    specs: list[dict] = Field(
        default_factory=list, description="List of visualization specifications"
    )
    style_overrides: dict | None = Field(
        default=None, description="Style overrides for matplotlib plots"
    )


def render_from_spec(
    spec: Mapping[str, Any],
    data: Any,
    *,
    t0: float,
    dt: float,
    outdir: Path,
    style_overrides: Mapping[str, Any] | None = None,
    save: bool = True,
) -> Mapping[str, Any]:
    """Render a figure from a validated visualizer spec."""
    # Try dispatch by explicit kind field
    spec_kind = str(spec.get("kind", "")) if isinstance(spec, Mapping) else ""

    visualizer: PhasePortraitPlotter | PsdPlotter | None = None
    psd_mode = False
    vspec: PhasePortraitSpec | PsdSpec | None = None

    # Determine renderer and validate spec
    if spec_kind in ("re_im", "Re_Im", "abs_abs", "Abs_Abs"):
        try:
            vspec = PhasePortraitSpec.model_validate(spec)
        except ValidationError as e:
            raise QPSConfigError(f"[530] Invalid phase-portrait spec: {e}") from e
        visualizer = PhasePortraitPlotter(config=vspec)
        psd_mode = False
    elif (
        spec_kind in ("complex", "modular") or "psd_type" in spec or spec_kind == "psd"
    ):
        # Allow back-compat
        payload = dict(spec)
        if payload.get("psd_type") and not payload.get("kind"):
            try:
                get_logger().warning(
                    "[992] 'psd_type' is deprecated; use 'kind' with values "
                    "'complex' or 'modular'."
                )
            except Exception:
                pass
            payload["kind"] = payload["psd_type"]
        if payload.get("kind") == "psd" and payload.get("psd_type"):
            try:
                get_logger().warning(
                    "[993] 'kind: psd' with 'psd_type' is deprecated; use 'kind' only."
                )
            except Exception:
                pass
            payload["kind"] = payload["psd_type"]
        try:
            vspec = PsdSpec.model_validate(payload)
        except ValidationError as e:
            raise QPSConfigError(f"[531] Invalid PSD spec: {e}") from e

        # Inject dt into config for PSD
        vspec = vspec.model_copy(update={"dt": dt})
        visualizer = PsdPlotter(config=vspec)
        psd_mode = True
    else:
        # Fallback: try to validate as PhasePortrait then PSD
        try:
            vspec = PhasePortraitSpec.model_validate(spec)
            visualizer = PhasePortraitPlotter(config=vspec)
            psd_mode = False
        except ValidationError:
            payload = dict(spec)
            if payload.get("psd_type") and not payload.get("kind"):
                payload["kind"] = payload["psd_type"]
            try:
                vspec = PsdSpec.model_validate(payload)
            except ValidationError as e:
                raise QPSConfigError(f"[532] Invalid visualizer spec: {e}") from e

            # Inject dt into config for PSD
            vspec = vspec.model_copy(update={"dt": dt})
            visualizer = PsdPlotter(config=vspec)
            psd_mode = True

    # Slice time range
    n_keep = getattr(data, "shape", [None, None, None])[1] or data.shape[1]
    t_range = getattr(vspec, "t_range", None)
    k0, k1 = _time_to_index(t0, dt, int(n_keep), t_range)
    sliced = data[:, k0 : k1 + 1, :]

    # Optional decimation for plotting-only (phase portraits)
    if not psd_mode:
        plot_every = getattr(vspec, "plot_every", None)
        if plot_every is not None:
            pe = max(1, int(plot_every))
            if pe > 1:
                sliced = sliced[:, ::pe, :]

    # Merge styles
    plot_style = dict(style_overrides or {})

    # Figure scaffolding
    figsize = plot_style.pop("figsize", (5, 5))
    dpi = plot_style.pop("dpi", 150)
    title = plot_style.pop("title", None)
    grid = plot_style.pop("grid", True)

    t_start = time.perf_counter()
    fig, ax = plt.subplots(figsize=figsize)

    # Render
    if psd_mode:
        # Run analysis
        assert isinstance(vspec, PsdSpec)
        analyzer_config = PsdAnalyzerConfig(
            kind=vspec.kind, modes=vspec.modes, convention=vspec.convention, dt=dt
        )
        analyzer = PsdAnalyzer(config=analyzer_config)
        # sliced is (n_traj, n_time, n_modes)
        analysis_result = analyzer.analyze(sliced)

        # Pass result to visualizer
        assert isinstance(visualizer, PsdPlotter)
        category = visualizer.render(ax, analysis_result, plot_style=plot_style)
    else:
        assert isinstance(visualizer, PhasePortraitPlotter)
        category = visualizer.render(ax, sliced, plot_style=plot_style)

    if title:
        ax.set_title(title)
    ax.grid(grid)

    # Hash and filename
    vdict = vspec.model_dump()
    if psd_mode:
        x_scale = plot_style.get("x_scale") if isinstance(plot_style, dict) else None
        y_scale = plot_style.get("y_scale") if isinstance(plot_style, dict) else None
        h = spec_short_hash(
            {
                "viz": "psd",
                "kind": vdict.get("kind"),
                "modes": vdict.get("modes"),
                "convention": vdict.get("convention"),
                "x_scale": x_scale,
                "y_scale": y_scale,
                "xlim": vdict.get("xlim"),
                "t_range": vdict.get("t_range"),
            }
        )
    else:
        h = spec_short_hash(
            {
                "kind": vdict.get("kind"),
                "modes": vdict.get("modes"),
                "t_range": vdict.get("t_range"),
                "plot_every": vdict.get("plot_every"),
            }
        )
    fname = f"{h}_{category}.png"
    fpath = outdir / fname
    if save:
        fig.tight_layout()
        fig.savefig(fpath, dpi=dpi)
    plt.close(fig)
    duration_ms = (time.perf_counter() - t_start) * 1000.0
    tags = ["psd", category] if psd_mode else ["phase_portrait", category]
    rend_name = "psd" if psd_mode else "phase_portrait"
    return {
        "generated_filename": fname,
        "path": fpath,
        "hash": h,
        "tags": tags,
        "duration_ms": duration_ms,
        "renderer": rend_name,
    }


class VizEngine:
    """Visualization Engine.

    Wraps the visualization logic in a class-based interface compatible with qphase.
    """

    name: ClassVar[str] = "viz"
    description: ClassVar[str] = (
        "Visualization engine for rendering phase portraits and PSD plots"
    )
    config_schema: ClassVar[type[VizEngineConfig]] = VizEngineConfig

    def __init__(
        self,
        config: VizEngineConfig | dict | None = None,
        plugins: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the VizEngine.

        Parameters
        ----------
        config : VizEngineConfig | dict | None
            Configuration for the engine. Can be a VizEngineConfig instance,
            a dictionary, or None (uses defaults).
        plugins : dict[str, Any] | None
            Dictionary of instantiated plugins (not used by viz engine).
        **kwargs : Any
            Additional keyword arguments for future extensibility.

        """
        if config is None:
            config = VizEngineConfig()  # type: ignore[call-arg]
        elif isinstance(config, dict):
            config = VizEngineConfig.model_validate(config)

        self.config = config
        self.plugins = plugins or {}

    def run(self, data: Any | None = None) -> Any:
        """Run visualization pipeline.

        Parameters
        ----------
        data : Any | None
            Input data to visualize.

        Returns
        -------
        Any
            Visualization results.

        """
        # Convert config to dict for main function
        config_dict = self.config.model_dump()
        return main(config_dict, {}, data=data)


def main(
    config: dict[str, Any],
    plugins: dict[str, Any],
    data: Any | None = None,
) -> Any:
    """Return visualization results (metadata) for Viz package."""
    # 1. Validate Input Data
    if data is None:
        raise ValueError("Visualization requires input data.")

    # Extract trajectory and metadata from input object
    if hasattr(data, "trajectory"):
        trajectory = data.trajectory
        meta = getattr(data, "meta", {})
    elif isinstance(data, dict):
        trajectory = data.get("trajectory")
        meta = data.get("meta", {})
    else:
        trajectory = data
        meta = {}

    # 2. Prepare Rendering Parameters
    t0 = getattr(trajectory, "t0", meta.get("t0", 0.0))
    dt = getattr(trajectory, "dt", meta.get("dt", 1.0))

    if trajectory is not None and hasattr(trajectory, "data"):
        raw_data = trajectory.data
    else:
        raw_data = trajectory

    outdir = Path(config.get("output_dir", "."))
    outdir.mkdir(parents=True, exist_ok=True)

    # 3. Render Figures
    specs = config.get("specs", [])
    if not specs:
        if "kind" in config:
            specs = [config]
        else:
            specs = config.get("params", {}).get("specs", [])

    results = []
    for spec in specs:
        try:
            res = render_from_spec(
                spec=spec, data=raw_data, t0=t0, dt=dt, outdir=outdir, save=True
            )
            results.append(res)
        except Exception as e:
            print(f"[qphase_viz] Error rendering spec {spec.get('kind')}: {e}")

    return {"status": "success", "package": "viz", "results": results}
