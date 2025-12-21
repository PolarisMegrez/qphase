"""qphase_sde: Visualizers Subpackage
---------------------------------
Figure plotters and services for simulation outputs (phase portraits, PSD),
validated by specs and plugged via the central registry.

Usage
-----
Registry keys:
`visualizer:phase_portrait` | `visualizer:psd`

Service:
>>> from qphase_sde.visualizer.service import render_from_spec

Notes
-----
- Renderers are registered lazily; plotting dependencies are imported only
  when needed.

"""

__all__ = []
