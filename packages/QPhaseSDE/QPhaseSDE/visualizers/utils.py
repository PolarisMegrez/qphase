"""
QPhaseSDE: Visualizer utilities
-------------------------------
Shared helpers used by the visualizer service and renderers.

Behavior
- Compute short, stable identifiers from spec content for figure/cache naming.
- Convert time ranges to inclusive index ranges for time-series slicing.
- Ensure output directories exist when saving figures.

Notes
- Only ``spec_short_hash`` is a public utility; other helpers are internal.
"""

__all__ = [
        "spec_short_hash",
]

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json
import math
import hashlib


def spec_short_hash(spec: Dict[str, Any], length: int = 8) -> str:
    """
    Compute a short, deterministic hash from a visualizer spec.

    The hash is generated using only the content of the spec (such as kind, modes, t_range),
    ignoring styling and cosmetic fields so that visual changes do not affect identity.

    Parameters
    ----------
    spec : dict
        Visualization spec dictionary. Only its content is used for hashing.
    length : int, optional
        Length of the returned hash string (default is 8).

    Returns
    -------
    str
        Short hash string representing the spec content.

    Examples
    --------
    >>> spec_short_hash({"kind": "phase", "modes": [0, 1], "t_range": [0, 10]})
    'a1b2c3d4'
    """
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha1(payload).hexdigest()
    return h[:length]


def _time_to_index(t0: float, dt: float, n_steps: int, t_range: Optional[Tuple[float, float]]):
    """
    Convert a time range to index range for time series data.
    """
    if t_range is None:
        return 0, n_steps - 1
    t_start, t_end = t_range
    k0 = max(0, int(math.floor((t_start - t0) / dt)))
    k1 = min(n_steps - 1, int(math.ceil((t_end - t0) / dt)))
    if k1 < k0:
        k0, k1 = 0, n_steps - 1
    return k0, k1


def _ensure_outdir(run_dir: Path, out_dir: Optional[Path] = None) -> Path:
    """
    Ensure the output directory exists for saving figures.
    """
    if out_dir is None:
        out_dir = run_dir / 'figures' / 'ic00'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
