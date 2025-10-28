from __future__ import annotations
from ..utils import _time_to_index

"""Phase-portrait visualization utilities for time series outputs.

Supported figure kinds:
- 're_im': Re(alpha_m) vs Im(alpha_m) for a single mode m
- 'abs_abs': |alpha_m1| vs |alpha_m2| for a pair of modes
"""

from typing import Dict, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from QPhaseSDE.core.errors import ConfigError

def render_phase_portrait(ax: Axes, data: np.ndarray, spec: Dict[str, Any], plot_style: Optional[Dict] = None) -> str:
    """Render a phase portrait onto an existing Axes; return category tag.

    data: complex ndarray (n_traj, n_keep, n_modes)
    spec: dict with keys kind ('re_im'|'abs_abs') and modes
    plot_style: matplotlib Line2D kwargs
    Returns: category tag for filenames ('pp_reim' or 'pp_abs')
    """
    style = dict(plot_style or {})
    n_traj, n_keep, n_modes = data.shape
    kind = spec.get('kind')
    if kind == 're_im':
        modes = spec.get('modes', [])
        if len(modes) != 1:
            raise ConfigError('[521] re_im requires exactly one mode index')
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
            raise ConfigError('[522] abs_abs requires exactly two mode indices')
        m1, m2 = modes
        x = np.abs(data[:, :, m1])
        y = np.abs(data[:, :, m2])
        for i in range(n_traj):
            ax.plot(x[i], y[i], **style)
        ax.set_xlabel(f'|Mode {m1}|')
        ax.set_ylabel(f'|Mode {m2}|')
        return 'pp_abs'
    else:
        raise ConfigError(f'[523] Unknown phase kind: {kind}')


def validate_phase_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize and validate a phase-portrait spec dict.

    Ensures keys: kind ('re_im'|'abs_abs'), modes (list[int]),
    and optional t_range (list[float] length 2 with t_end>t_start).
    Returns a normalized dict.
    """
    out: Dict[str, Any] = {}
    kind = str(spec.get("kind"))
    if kind in ("Re_Im", "re_im"):
        kind = "re_im"
    elif kind in ("Abs_Abs", "abs_abs"):
        kind = "abs_abs"
    else:
        raise ConfigError("[524] Unsupported kind for phase portrait")
    modes = list(spec.get("modes", []))
    if kind == "re_im" and len(modes) != 1:
        raise ConfigError("[525] re_im requires exactly one mode index in modes")
    if kind == "abs_abs" and len(modes) != 2:
        raise ConfigError("[526] abs_abs requires exactly two mode indices in modes")
    t_range = spec.get("t_range")
    if t_range is not None:
        if not (isinstance(t_range, (list, tuple)) and len(t_range) == 2):
            raise ConfigError("[527] t_range must be [t_start, t_end]")
        t0, t1 = float(t_range[0]), float(t_range[1])
        if not (t1 > t0):
            raise ConfigError("[528] t_range must satisfy t_end > t_start")
        out["t_range"] = [t0, t1]
    else:
        out["t_range"] = None
    out["kind"] = kind
    out["modes"] = modes
    return out