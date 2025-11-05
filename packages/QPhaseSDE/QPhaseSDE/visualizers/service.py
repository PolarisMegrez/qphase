"""
QPhaseSDE: Visualizer service
-----------------------------
High-level entry for turning a validated visualizer spec and data into a
saved figure plus metadata.

Behavior
- Validates the input spec (``PhasePortraitSpec`` or ``PsdSpec``) and
    normalizes deprecated fields when present.
- Slices time based on ``t0``, ``dt``, and optional ``t_range``; optionally
    decimates samples for plotting-only views.
- Selects a renderer from the registry (``visualizer:phase_portrait`` or
    ``visualizer:psd``), merges style overrides, and renders into a Matplotlib
    figure/axes.
- Optionally saves the figure and returns metadata (filename, path, hash,
    tags, duration, renderer name).

Notes
- Deprecated fields ``psd_type`` and ``kind: psd`` are accepted for backward
    compatibility and will emit warnings; prefer the newer ``kind`` values
    (``complex`` or ``modular``).
- Detailed parameter, return, and error semantics are documented in
    ``render_from_spec``.
"""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import time

import matplotlib.pyplot as plt

__all__ = [
    "render_from_spec",
]

from .specs import PhasePortraitSpec, PsdSpec
from .utils import _ensure_outdir, spec_short_hash, _time_to_index
from ..core.registry import registry
from ..core.errors import QPSConfigError, get_logger
from pydantic import ValidationError

def render_from_spec(
    spec: Mapping[str, Any],
    data: Any,
    *,
    t0: float,
    dt: float,
    outdir: Path,
    style_overrides: Optional[Mapping[str, Any]] = None,
    save: bool = True,
) -> Mapping[str, Any]:
    """
    Render a figure based on a validated visualizer spec.

    Dispatches to the appropriate renderer based on the spec content. Supports
    PhasePortraitSpec and PsdSpec. Handles validation, slicing, style merging,
    figure creation, saving, and metadata collection.

    Parameters
    ----------
    spec : Mapping[str, Any]
        Visualization specification dictionary.
    data : Any
        Data to visualize (typically ndarray or similar).
    t0 : float
        Initial time for slicing.
    dt : float
        Time step for slicing and PSD.
    outdir : Path
        Output directory for saving figures.
    style_overrides : Mapping[str, Any], optional
        Style overrides for rendering.
    save : bool, default True
        Whether to save the figure to disk.

    Returns
    -------
    dict
        Metadata including filename, path, hash, tags, duration, renderer name.

    Raises
    ------
    QPSConfigError
        - [530] Invalid phase-portrait spec
        - [531] Invalid PSD spec
        - [532] Invalid visualizer spec

    Examples
    --------
    >>> meta = render_from_spec(spec, data, t0=0.0, dt=0.01, outdir=Path("./out"))
    >>> print(meta["generated_filename"])
    """
    # Try dispatch by explicit kind field
    spec_kind = str(spec.get("kind", "")) if isinstance(spec, Mapping) else ""
    renderer_key: str
    if spec_kind in ("re_im", "Re_Im", "abs_abs", "Abs_Abs"):
        try:
            vspec = PhasePortraitSpec.model_validate(spec)
        except ValidationError as e:  # normalize to framework error
            raise QPSConfigError(f"[530] Invalid phase-portrait spec: {e}")
        renderer_key = "visualizer:phase_portrait"
        psd_mode = False
    elif spec_kind in ("complex", "modular") or "psd_type" in spec or spec_kind == "psd":
        # Allow back-compat: if psd_type provided or kind=='psd', map to new fields
        payload = dict(spec)
        if payload.get("psd_type") and not payload.get("kind"):
            try:
                get_logger().warning("[992] 'psd_type' is deprecated; use 'kind' with values 'complex'|'modular'.")
            except Exception:
                pass
            payload["kind"] = payload["psd_type"]
        if payload.get("kind") == "psd" and payload.get("psd_type"):
            try:
                get_logger().warning("[993] 'kind: psd' with 'psd_type' is deprecated; use 'kind' only.")
            except Exception:
                pass
            payload["kind"] = payload["psd_type"]
        try:
            vspec = PsdSpec.model_validate(payload)
        except ValidationError as e:
            raise QPSConfigError(f"[531] Invalid PSD spec: {e}")
        renderer_key = "visualizer:psd"
        psd_mode = True
    else:
        # Fallback: try to validate as PhasePortrait then PSD
        try:
            vspec = PhasePortraitSpec.model_validate(spec)
            renderer_key = "visualizer:phase_portrait"
            psd_mode = False
        except ValidationError:
            payload = dict(spec)
            if payload.get("psd_type") and not payload.get("kind"):
                payload["kind"] = payload["psd_type"]
            try:
                vspec = PsdSpec.model_validate(payload)  # type: ignore[arg-type]
            except ValidationError as e:
                raise QPSConfigError(f"[532] Invalid visualizer spec: {e}")
            renderer_key = "visualizer:psd"
            psd_mode = True

    # Slice time range
    n_keep = getattr(data, "shape", [None, None, None])[1] or data.shape[1]
    t_range = getattr(vspec, "t_range", None)
    k0, k1 = _time_to_index(t0, dt, int(n_keep), t_range)
    sliced = data[:, k0 : k1 + 1, :]

    # Optional decimation for plotting-only (phase portraits)
    if not 'psd_mode' in locals():
        psd_mode = False
    if not psd_mode:
        plot_every = getattr(vspec, 'plot_every', None)
        if plot_every is not None:
            pe = max(1, int(plot_every))
            if pe > 1:
                sliced = sliced[:, ::pe, :]

    # Resolve renderer via registry (function-style expected)
    # Namespace: use 'visualizer' consistently (was mistakenly 'renderer').
    # Map any legacy 'visualizer:*' keys to 'visualizer:*'.
    if renderer_key.startswith("visualizer:"):
        renderer_key = "visualizer:" + renderer_key.split(":", 1)[1]
    renderer = registry.create(renderer_key)
    # Merge styles (renderer defaults < profile < overrides). Currently we have no profile, so just overrides
    plot_style = dict(style_overrides or {})

    # Figure scaffolding
    figsize = plot_style.pop("figsize", (5, 5))
    dpi = plot_style.pop("dpi", 150)
    title = plot_style.pop("title", None)
    grid = plot_style.pop("grid", True)

    t_start = time.perf_counter()
    fig, ax = plt.subplots(figsize=figsize)
    # Inject dt for PSD renderer to avoid recomputing
    vdict = vspec.model_dump()
    if psd_mode:
        vdict["dt"] = dt
    category = renderer(ax, sliced, vdict, plot_style)
    if title:
        ax.set_title(title)
    ax.grid(grid)
    # Hash and filename
    if psd_mode:
        # include style scales if present
        x_scale = plot_style.get("x_scale") if isinstance(plot_style, dict) else None
        y_scale = plot_style.get("y_scale") if isinstance(plot_style, dict) else None
        h = spec_short_hash({
            "viz": "psd",
            "kind": vdict.get("kind"),
            "modes": vdict.get("modes"),
            "convention": vdict.get("convention"),
            "x_scale": x_scale,
            "y_scale": y_scale,
            "xlim": vdict.get("xlim"),
            "t_range": vdict.get("t_range"),
        })
    else:
        h = spec_short_hash({
            "kind": vdict.get("kind"),
            "modes": vdict.get("modes"),
            "t_range": vdict.get("t_range"),
            "plot_every": vdict.get("plot_every"),
        })
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
