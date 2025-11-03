"""
QPhaseSDE: Phase Portrait Renderer
----------------------------------
Render phase portraits from multi-trajectory time series for one or two modes.

Behavior
--------
- Support 're_im' (Re vs Im for one mode) and 'abs_abs' (|m1| vs |m2|) kinds.
    Spec validation and rendering semantics are documented by the functions.

Notes
-----
- Avoid importing heavy plotting libraries outside of rendering contexts.
"""

__all__ = [
    "render_phase_portrait",
    "validate_phase_spec",
]

from ..utils import _time_to_index
from typing import Dict, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from QPhaseSDE.core.errors import QPSConfigError

def render_phase_portrait(
    ax: Axes,
    data: np.ndarray,
    spec: Dict[str, Any],
    plot_style: Optional[Dict] = None
) -> str:
    """
    Render a phase portrait onto an existing matplotlib Axes.

    Supports two figure kinds:
        - 're_im': Re(alpha_m) vs Im(alpha_m) for a single mode m
        - 'abs_abs': |alpha_m1| vs |alpha_m2| for a pair of modes

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    data : np.ndarray
        Complex array of shape (n_traj, n_keep, n_modes), time series data.
    spec : dict
        Dict with keys 'kind' ('re_im'|'abs_abs') and 'modes'.
    plot_style : dict, optional
        Matplotlib Line2D keyword arguments for styling.

    Returns
    -------
    str
        Category tag for filenames ('pp_reim' or 'pp_abs').

    Raises
    ------
    QPSConfigError
        [521] If 're_im' kind and modes does not contain exactly one index.
        [522] If 'abs_abs' kind and modes does not contain exactly two indices.
        [523] If kind is unknown.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> tag = render_phase_portrait(ax, data, {"kind": "re_im", "modes": [0]})
    >>> plt.show()
    """
    style = dict(plot_style or {})
    n_traj, n_keep, n_modes = data.shape
    kind = spec.get('kind')
    if kind == 're_im':
        modes = spec.get('modes', [])
        if len(modes) != 1:
            raise QPSConfigError('[521] re_im requires exactly one mode index')
        m = modes[0]
        x = data[:, :, m].real
        y = data[:, :, m].imag
        for i in range(n_traj):
            ax.plot(x[i], y[i], **style)
        ax.set_xlabel(f'Re(Mode {m})')
        ax.set_ylabel(f'Im(Mode {m})')
        return 'pp_reim'
    elif kind == 'abs_abs':
        modes = spec.get('modes', [])
        if len(modes) != 2:
            raise QPSConfigError('[522] abs_abs requires exactly two mode indices')
        m1, m2 = modes
        x = np.abs(data[:, :, m1])
        y = np.abs(data[:, :, m2])
        for i in range(n_traj):
            ax.plot(x[i], y[i], **style)
        ax.set_xlabel(f'|Mode {m1}|')
        ax.set_ylabel(f'|Mode {m2}|')
        return 'pp_abs'
    else:
        raise QPSConfigError(f'[523] Unknown phase kind: {kind}')


def validate_phase_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and validate a phase-portrait spec dict.

    Ensures keys:
        - kind ('re_im'|'abs_abs')
        - modes (list[int])
        - optional t_range (list[float] length 2 with t_end > t_start)

    Parameters
    ----------
    spec : dict
        Input spec dictionary to validate and normalize.

    Returns
    -------
    dict
        Normalized spec dict with keys: kind, modes, t_range.

    Raises
    ------
    QPSConfigError
        [524] If kind is unsupported.
        [525] If 're_im' kind and modes does not contain exactly one index.
        [526] If 'abs_abs' kind and modes does not contain exactly two indices.
        [527] If t_range is not a valid [t_start, t_end] list.
        [528] If t_range does not satisfy t_end > t_start.

    Examples
    --------
    >>> spec = {"kind": "re_im", "modes": [0], "t_range": [0.0, 10.0]}
    >>> out = validate_phase_spec(spec)
    >>> print(out)
    {'kind': 're_im', 'modes': [0], 't_range': [0.0, 10.0]}
    """
    out: Dict[str, Any] = {}
    kind = str(spec.get("kind"))
    if kind in ("Re_Im", "re_im"):
        kind = "re_im"
    elif kind in ("Abs_Abs", "abs_abs"):
        kind = "abs_abs"
    else:
        raise QPSConfigError("[524] Unsupported kind for phase portrait")
    modes = list(spec.get("modes", []))
    if kind == "re_im" and len(modes) != 1:
        raise QPSConfigError("[525] re_im requires exactly one mode index in modes")
    if kind == "abs_abs" and len(modes) != 2:
        raise QPSConfigError("[526] abs_abs requires exactly two mode indices in modes")
    t_range = spec.get("t_range")
    if t_range is not None:
        if not (isinstance(t_range, (list, tuple)) and len(t_range) == 2):
            raise QPSConfigError("[527] t_range must be [t_start, t_end]")
        t0, t1 = float(t_range[0]), float(t_range[1])
        if not (t1 > t0):
            raise QPSConfigError("[528] t_range must satisfy t_end > t_start")
        out["t_range"] = [t0, t1]
    else:
        out["t_range"] = None
    out["kind"] = kind
    out["modes"] = modes
    return out