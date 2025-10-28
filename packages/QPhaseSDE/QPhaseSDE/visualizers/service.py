from __future__ import annotations

"""Visualizer service layer.

Responsibilities:
- Validate specs
- Select renderer via registry (function or class)
- Merge style layers (renderer defaults < profile < call overrides)
- Call renderer and handle saving and metadata
"""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import time

import matplotlib.pyplot as plt

from .specs import PhasePortraitSpec, PsdSpec
from .utils import _ensure_outdir, spec_short_hash, _time_to_index
from ..core.registry import registry


def _merge_styles(*layers: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for layer in layers:
        if not layer:
            continue
        for k, v in layer.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = {**out[k], **v}  # shallow deep-merge
            else:
                out[k] = v
    return out


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
    """Render a figure based on a validated spec dict.

    Dispatches to the appropriate renderer based on the spec content. Supports
    PhasePortraitSpec and PsdSpec.
    """
    # Try dispatch by explicit kind field
    spec_kind = str(spec.get("kind", "")) if isinstance(spec, Mapping) else ""
    renderer_key: str
    if spec_kind in ("re_im", "Re_Im", "abs_abs", "Abs_Abs"):
        vspec = PhasePortraitSpec.model_validate(spec)
        renderer_key = "visualization:phase_portrait"
        psd_mode = False
    elif spec_kind in ("complex", "modular") or "psd_type" in spec or spec_kind == "psd":
        # Allow back-compat: if psd_type provided or kind=='psd', map to new fields
        payload = dict(spec)
        if payload.get("psd_type") and not payload.get("kind"):
            payload["kind"] = payload["psd_type"]
        if payload.get("kind") == "psd" and payload.get("psd_type"):
            payload["kind"] = payload["psd_type"]
        vspec = PsdSpec.model_validate(payload)
        renderer_key = "visualization:psd"
        psd_mode = True
    else:
        # Fallback: try to validate as PhasePortrait then PSD
        try:
            vspec = PhasePortraitSpec.model_validate(spec)
            renderer_key = "visualization:phase_portrait"
            psd_mode = False
        except Exception:
            payload = dict(spec)
            if payload.get("psd_type") and not payload.get("kind"):
                payload["kind"] = payload["psd_type"]
            vspec = PsdSpec.model_validate(payload)  # type: ignore[arg-type]
            renderer_key = "visualization:psd"
            psd_mode = True

    # Slice time range
    n_keep = getattr(data, "shape", [None, None, None])[1] or data.shape[1]
    t_range = getattr(vspec, "t_range", None)
    k0, k1 = _time_to_index(t0, dt, int(n_keep), t_range)
    sliced = data[:, k0 : k1 + 1, :]

    # Resolve renderer via registry (function-style expected)
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
        h = spec_short_hash({"kind": vdict.get("kind"), "modes": vdict.get("modes"), "t_range": vdict.get("t_range")})
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
