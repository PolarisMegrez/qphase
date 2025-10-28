from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from matplotlib.axes import Axes
from ...analysis.psd import compute_psd_for_modes


def _compute_psd(
    x: np.ndarray,
    dt: float,
    *,
    mode: str = "complex",
    convention: str = "symmetric",
) -> Dict[str, np.ndarray]:
    """
    Compute two-sided PSD for a batch of trajectories and a single mode.

    x: shape (n_traj, n_time) complex array (time series for one mode across trajectories)
    dt: sample spacing
    mode: 'complex' uses x directly; 'modular' uses |x|
    convention:
      - 'symmetric' or 'unitary': use unitary FFT (norm='ortho') and angular frequency axis ω=2πf
      - 'pragmatic': use standard FFT (norm=None) and frequency axis f
    Returns dict with keys 'axis' (ω or f) and 'psd'
    """
    if mode not in ("complex", "modular"):
        raise ValueError("mode must be 'complex' or 'modular'")
    if convention not in ("symmetric", "unitary", "pragmatic"):
        raise ValueError("convention must be 'symmetric'|'unitary'|'pragmatic'")

    x_proc = np.abs(x) if mode == "modular" else x
    n_traj, n_time = x_proc.shape

    if convention in ("symmetric", "unitary"):
        X = np.fft.fft(x_proc, axis=1, norm="ortho")
        P = np.mean(np.abs(X) ** 2, axis=0)
        f = np.fft.fftfreq(n_time, d=dt)
        w = 2.0 * np.pi * f
        axis = w
    else:  # pragmatic
        X = np.fft.fft(x_proc, axis=1, norm=None)
        # Scale by N for Parseval-like invariance
        P = np.mean(np.abs(X) ** 2, axis=0) / float(n_time)
        axis = np.fft.fftfreq(n_time, d=dt)

    return {"axis": axis, "psd": P}


def render_psd(ax: Axes, data: np.ndarray, spec: Dict[str, Any], plot_style: Optional[Dict] = None) -> str:
    """
    Render PSD for one or more modes on a single Axes and return a category string
    used in filenames (e.g., 'psd_complex').

    data: shape (n_traj, n_time, n_modes) complex
    spec: dict validated via PsdSpec with keys: psd_type, modes, convention, t_range
    plot_style: optional dict for styling (color/linestyle/alpha/yscale/xlim/ylim)
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
        raise ValueError("PsdSpec.modes must include at least one mode index")

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
