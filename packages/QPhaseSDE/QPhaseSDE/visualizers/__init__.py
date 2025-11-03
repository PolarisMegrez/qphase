"""
QPhaseSDE: Visualizers Subpackage
---------------------------------
Figure renderers and services for simulation outputs (phase portraits, PSD),
validated by specs and plugged via the central registry.

Usage
-----
Registry keys:
`visualization:phase_portrait` | `visualization:psd`

Service:
>>> from QPhaseSDE.visualizers.service import render_from_spec

Notes
-----
- Renderers are registered lazily; plotting dependencies are imported only
  when needed.
"""

from ..core.registry import register_lazy

# Phase portrait renderer (function). We return the callable rather than
# instantiating anything at registration time.
register_lazy(
	"visualization",
	"phase_portrait",
	"QPhaseSDE.visualizers.renderers.phase_plane:render_phase_portrait",
	return_callable=True,
)

# Power Spectral Density renderer
register_lazy(
	"visualization",
	"psd",
	"QPhaseSDE.visualizers.renderers.psd:render_psd",
	return_callable=True,
)

__all__ = []
