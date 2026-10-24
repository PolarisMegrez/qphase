"""qphase_viz: Shared render helpers
---------------------------------------------------------
The one implementation of "figure in, styled file out". Every plotter draws
its own data, then calls these helpers for the parts that must not diverge
between plotters: title/labels/limits/grid/legend, annotations, and saving.

Styling is scoped: :func:`styled_figure` opens a local Matplotlib rc context,
so neither the engine nor a direct ``plotter.plot`` call changes the
process-wide ``rcParams``.

Public API
----------
``styled_figure`` : Context manager yielding a styled, exception-safe figure.
``apply_axes_style`` : Title, labels, limits, grid and legend.
``draw_annotations`` : Place typed annotations on an axes.
``save_figure`` : Write one figure and return the path.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .config import AnnotationSpec
from .styles import RenderOptions

__all__ = [
    "apply_axes_style",
    "draw_annotations",
    "save_figure",
    "styled_figure",
    "styled_subplots",
]


@contextmanager
def styled_subplots(
    options: RenderOptions,
    *,
    rows: int = 1,
    cols: int = 1,
    figsize: tuple[float, float] = (6.0, 4.0),
    dpi: int = 100,
    sharex: bool = False,
    sharey: bool = False,
    constrained_layout: bool = False,
) -> Iterator[tuple[Figure, np.ndarray]]:
    """Yield ``(figure, axes)`` under *options* and always close the figure.

    ``axes`` always has shape ``(rows, cols)``, so a single-panel plotter and a
    stacked-panel plotter share one rendering path. The rc parameters live only
    for the duration of the block, so a caller may render any number of figures
    without leaking style into the process.
    """
    with mpl.rc_context(options.rc_params()):
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=figsize,
            dpi=dpi,
            sharex=sharex,
            sharey=sharey,
            constrained_layout=constrained_layout,
            squeeze=False,
        )
        try:
            yield fig, axes
        finally:
            plt.close(fig)


@contextmanager
def styled_figure(
    options: RenderOptions,
    *,
    figsize: tuple[float, float],
    dpi: int,
) -> Iterator[tuple[Figure, Axes]]:
    """Yield a single styled ``(figure, axes)`` pair and always close it."""
    with styled_subplots(options, figsize=figsize, dpi=dpi) as (fig, axes):
        yield fig, axes[0, 0]


def apply_axes_style(
    ax: Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    grid: bool = False,
    legend: bool = False,
    grid_which: Literal["major", "minor", "both"] = "major",
) -> None:
    """Apply the labels, limits, grid and legend of one figure.

    ``xlabel``/``ylabel``/``title`` are already resolved by the caller, which
    owns the per-plotter fallback text.
    """
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if grid:
        ax.grid(True, alpha=0.3, which=grid_which)
    if legend:
        ax.legend()


def draw_annotations(ax: Axes, annotations: Sequence[AnnotationSpec]) -> None:
    """Place each annotation on *ax* in its declared coordinate frame."""
    for note in annotations:
        transform = ax.transAxes if note.coords == "axes fraction" else ax.transData
        extra: dict[str, Any] = {}
        if note.fontsize is not None:
            extra["fontsize"] = note.fontsize
        if note.color is not None:
            extra["color"] = note.color
        ax.text(
            note.x,
            note.y,
            note.text,
            transform=transform,
            ha=note.ha,
            va=note.va,
            rotation=note.rotation,
            **extra,
        )


def save_figure(
    fig: Figure,
    output_dir: Path,
    filename: str,
    format: str,
) -> Path:
    """Write *fig* as ``output_dir/filename.format`` and return the path."""
    out_path = output_dir / f"{filename}.{format}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format=format, bbox_inches="tight")
    return out_path
