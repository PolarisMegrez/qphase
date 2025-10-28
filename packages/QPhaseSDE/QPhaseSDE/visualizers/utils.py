from __future__ import annotations

"""Visualization utilities shared across visualization backends.

- spec_short_hash: Compute a short, deterministic hash from a visualization spec
  using only its content (e.g., kind/modes/t_range), ignoring styling so that
  cosmetic changes do not affect identity.
- validate_phase_spec: Normalize and validate a single phase-portrait spec.
- validate_phase_specs: Validate a list of specs.

These functions are independent of any particular figure renderer and can be
used by both core and CLI layers.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import hashlib

def spec_short_hash(spec: Dict[str, Any], length: int = 8) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha1(payload).hexdigest()
    return h[:length]

def _time_to_index(t0: float, dt: float, n_steps: int, t_range: Optional[Tuple[float, float]]):
    if t_range is None:
        return 0, n_steps - 1
    t_start, t_end = t_range
    k0 = max(0, int(math.floor((t_start - t0) / dt)))
    k1 = min(n_steps - 1, int(math.ceil((t_end - t0) / dt)))
    if k1 < k0:
        k0, k1 = 0, n_steps - 1
    return k0, k1

def _ensure_outdir(run_dir: Path, out_dir: Optional[Path] = None) -> Path:
    """Fallback ensure out_dir exists. Defaults to figures/ic00 under run_dir."""
    if out_dir is None:
        out_dir = run_dir / 'figures' / 'ic00'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
