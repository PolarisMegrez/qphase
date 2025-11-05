"""
QPhaseSDE: PSD Renderer
-----------------------
Render power spectral density (PSD) for one or more modes from multi-trajectory
time series using analysis utilities and plot onto Matplotlib axes.

Behavior
--------
- Delegate PSD computation to analysis.psd; choose plot scales (linear/log)
    and sidedness based on spec and style. Details are documented by functions.
"""

__all__ = [
    "render_psd",
]

from typing import Any, Dict, List, Optional
import numpy as np
from matplotlib.axes import Axes
from ...analysis.psd import compute_psd_for_modes
from ...core.errors import QPSConfigError


def render_psd(
    ax: Axes,
    data: np.ndarray,
    spec: Dict[str, Any],
    plot_style: Optional[Dict] = None
) -> str:
    """
    Render power spectral density (PSD) for one or more modes on a matplotlib Axes.

    This function computes and plots the PSD for specified modes, using the provided
    data and configuration spec. It supports various conventions and plot styles.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    data : np.ndarray
        Array of shape (n_traj, n_time, n_modes), complex time series data.
    spec : dict
        Visualization spec validated via PsdSpec, with keys: kind, modes, convention, t_range, etc.
    plot_style : dict, optional
        Optional dict for styling (color, linestyle, alpha, yscale, xlim, ylim, legend).

    Returns
    -------
    str
        Category string used in filenames (e.g., 'psd_complex').

    Raises
    ------
    QPSConfigError
        [526] If spec['modes'] is empty.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> category = render_psd(ax, data, spec, plot_style={"x_scale": "log", "legend": True})
    >>> plt.show()
    """
    kind: str = str(spec.get("kind"))  # 'complex' | 'modular'
    convention: str = str(spec.get("convention", "symmetric"))
    modes: List[int] = list(spec.get("modes", []))
    # effective plot type: style override wins, then spec, then default by kind
    # Derive plot type from style x_scale/y_scale or defaults by kind
    x_scale = (plot_style or {}).get("x_scale")
    y_scale = (plot_style or {}).get("y_scale")
    if x_scale or y_scale:
        xs = (x_scale or "linear").lower()
        ys = (y_scale or "linear").lower()
        if xs == "log" and ys == "log":
            eff_plot_type = "loglog"
        elif xs == "log" and ys != "log":
            eff_plot_type = "semilogx"
        elif xs != "log" and ys == "log":
            eff_plot_type = "semilogy"
        else:
            eff_plot_type = "linear"
    else:
        eff_plot_type = "linear" if kind == "complex" else "loglog"
    xlim_spec = spec.get("xlim")
    # one- vs two-sided policy by kind
    sided = "two" if kind == "complex" else "one"

    if len(modes) == 0:
        raise QPSConfigError("[526] PsdSpec.modes must include at least one mode index")

    # Draw
    # Compute PSDs in batch
    res_all = compute_psd_for_modes(data, spec.get("dt", 1.0), modes, kind=kind, convention=convention)
    axis = res_all["axis"]
    Pmat = res_all["psd"]  # (n_freq, n_modes)
    # Plot each mode line
    for j, m in enumerate(modes):
        P = Pmat[:, j]
        label = f"mode {m}"

        # two-sided handling: sort by frequency for continuous line; for log-x, only positive side possible
        if sided == "two":
            if eff_plot_type in ("loglog", "semilogx"):
                mask = axis > 0
                axis_p, P_p = axis[mask], P[mask]
                # plot positive side only (log x cannot show negative)
                if eff_plot_type == "loglog":
                    ax.loglog(axis_p, P_p, label=label)
                elif eff_plot_type == "semilogx":
                    ax.semilogx(axis_p, P_p, label=label)
            else:
                # linear x: show both sides; sort for continuity
                idx = np.argsort(axis)
                axis_s, P_s = axis[idx], P[idx]
                if eff_plot_type == "semilogy":
                    ax.semilogy(axis_s, P_s, label=label)
                else:  # linear
                    ax.plot(axis_s, P_s, label=label)
        else:
            # one-sided: keep non-negative for linear/semilogy, positive for log-x
            if eff_plot_type in ("loglog", "semilogx"):
                mask = axis > 0
            else:
                mask = axis >= 0
            axis_1, P_1 = axis[mask], P[mask]
            if eff_plot_type == "loglog":
                ax.loglog(axis_1, P_1, label=label)
            elif eff_plot_type == "semilogx":
                ax.semilogx(axis_1, P_1, label=label)
            elif eff_plot_type == "semilogy":
                ax.semilogy(axis_1, P_1, label=label)
            else:
                ax.plot(axis_1, P_1, label=label)

    # Style
    if plot_style:
        # plot_type 决定坐标标度，这里不再根据样式强制 yscale/xscale
        if "xlim" in plot_style:
            ax.set_xlim(*plot_style["xlim"])  # type: ignore[arg-type]
        if "ylim" in plot_style:
            ax.set_ylim(*plot_style["ylim"])  # type: ignore[arg-type]
        if "legend" in plot_style and bool(plot_style["legend"]):
            ax.legend()
    # apply xlim from spec if not overridden by style
    if xlim_spec is not None and not (plot_style and "xlim" in plot_style):
        try:
            ax.set_xlim(*xlim_spec)  # type: ignore[arg-type]
        except Exception:
            pass
    # Show legend by default if any modes>1
    if not plot_style or not plot_style.get("legend", False):
        if len(modes) > 1:
            ax.legend()

    # Labels
    if convention in ("symmetric", "unitary"):
        ax.set_xlabel("ω [rad/s]")
    else:
        ax.set_xlabel("f [Hz]")
    ax.set_ylabel("PSD (a.u.)")

    category = f"psd_{kind}"
    return category
