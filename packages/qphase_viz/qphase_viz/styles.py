"""qphase_viz: Named render styles
---------------------------------------------------------
The single source of truth for how a qphase figure looks: typography and axis
geometry per named style, the semantic colour roles, and the colormaps derived
from them. Nothing here draws: the render helpers in :mod:`qphase_viz.render`
apply these values inside a local rc context, so a figure never leaks settings
into the process-wide Matplotlib state.

Public API
----------
``StyleName`` : The available style names.
``RenderOptions`` : Immutable, resolved style for one figure.
``PALETTE`` : Semantic colour roles shared by papers, SM and the GUI.
``allan_scan_colormap`` : Perceptually ordered map for dense scan sequences.
``cam_flow_colormap`` : Diverging map for signed flows around zero.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "DEFAULT_STYLE",
    "PALETTE",
    "STYLE_NAMES",
    "RenderOptions",
    "StyleName",
    "allan_scan_colormap",
    "cam_flow_colormap",
    "rc_params",
]

#: Semantic colour roles. Names describe what a colour *means* in a figure, not
#: where it appears, so papers, SM and the GUI stay visually consistent.
PALETTE: dict[str, str] = {
    "ink": "#111827",
    "cam": "#0072B2",
    "sde": "#D55E00",
    "accent": "#E69F00",
    "green": "#009E73",
    "magenta": "#CC79A7",
    "purple": "#6F3CC3",
    "proxy": "#66717A",
    "noncritical": "#7A8288",
    "white_fm": "#DCEEFF",
    "flow_zero": "#FAFAF7",
}

StyleName = Literal["publication", "sm", "sans"]
STYLE_NAMES: tuple[StyleName, ...] = ("publication", "sm", "sans")
DEFAULT_STYLE: StyleName = "publication"

#: Axis geometry and export resolution shared by every style.
_SHARED_RC: dict[str, Any] = {
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi": 300,
}

#: Paper typography: STIX text and math, 12 pt body, 14.4 pt axis labels.
_PUBLICATION_RC: dict[str, Any] = {
    **_SHARED_RC,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 12.0,
    "axes.labelsize": 14.4,
    "axes.titlesize": 11.2,
    "legend.fontsize": 12.0,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
}

#: Supplemental-material typography: same families, one size step down.
_SM_RC: dict[str, Any] = {
    **_SHARED_RC,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.labelsize": 13.5,
    "axes.titlesize": 10.5,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
}

#: Sans typography (Arial with sans math) kept from the pre-style engine.
_SANS_RC: dict[str, Any] = {
    **_SHARED_RC,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "axes.labelsize": 18.0,
    "xtick.labelsize": 15.0,
    "ytick.labelsize": 15.0,
}

_STYLE_RC: dict[StyleName, dict[str, Any]] = {
    "publication": _PUBLICATION_RC,
    "sm": _SM_RC,
    "sans": _SANS_RC,
}

#: Editable text: keep glyphs as text so the file can be edited in Inkscape.
_EDITABLE_RC: dict[str, Any] = {"svg.fonttype": "none", "pdf.fonttype": 42}
#: Non-editable text: embed outlines (no font needed by the reader).
_OUTLINED_RC: dict[str, Any] = {"svg.fonttype": "path", "pdf.fonttype": 3}


def rc_params(style: StyleName) -> dict[str, Any]:
    """Return the Matplotlib parameters of one named style.

    A fresh dict is returned, so callers may mutate it without touching the
    style table.
    """
    try:
        table = _STYLE_RC[style]
    except KeyError:
        raise ValueError(
            f"unknown figure style {style!r}; expected one of {STYLE_NAMES}"
        ) from None
    return dict(table)


@dataclass(frozen=True)
class RenderOptions:
    """Resolved render style for one figure.

    Immutable on purpose: the engine resolves its own options once and hands
    them to plotters, and a single figure derives a new instance instead of
    mutating the engine's. Resolution order is always
    *single figure > engine > ``publication`` default*, and
    ``style_overrides`` is applied last so an explicit rc override always wins
    over the style table and the editable-text switch.
    """

    style: StyleName = DEFAULT_STYLE
    editable: bool = True
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def merged(
        self,
        *,
        style: StyleName | None = None,
        editable: bool | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> RenderOptions:
        """Return these options with per-figure settings layered on top."""
        return RenderOptions(
            style=self.style if style is None else style,
            editable=self.editable if editable is None else editable,
            overrides={**self.overrides, **(overrides or {})},
        )

    def rc_params(self) -> dict[str, Any]:
        """Return the Matplotlib parameters this figure renders with."""
        params = rc_params(self.style)
        params.update(_EDITABLE_RC if self.editable else _OUTLINED_RC)
        params.update(self.overrides)
        return params


def allan_scan_colormap() -> LinearSegmentedColormap:
    """Perceptually ordered map for dense scan sequences."""
    stops = mpl.colormaps["plasma"](np.linspace(0.12, 0.92, 256))
    return LinearSegmentedColormap.from_list("qphase_scan", stops)


def cam_flow_colormap() -> LinearSegmentedColormap:
    """Diverging map for signed flows, centred on the ``flow_zero`` tone."""
    return LinearSegmentedColormap.from_list(
        "qphase_cam_flow",
        [PALETTE["cam"], PALETTE["flow_zero"], PALETTE["sde"]],
    )
