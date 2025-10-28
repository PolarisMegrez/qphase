"""Visualizers package.

Registers visualization renderers in the global registry. We register lazily
so importing QPhaseSDE doesn't pull heavy plotting deps until needed.
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
