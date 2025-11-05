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


## Note: Validation moved to Pydantic models in visualizers/specs.py.