"""qphase_viz: Plotter Protocol
--------------------------

Protocol definition for visualizers/plotters.
"""

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Plotter",
]


@runtime_checkable
class Plotter(Protocol):
    """Protocol for visualizer implementations.

    Plotters are plugins that render data onto a matplotlib Axes.
    They must satisfy the PluginBase protocol (Config + __init__) implicitly,
    and implement the render method defined here.
    """

    def render(
        self,
        ax: Any,
        data: Any,
        plot_style: dict[str, Any] | None = None,
    ) -> str:
        """Render the visualization onto an existing matplotlib Axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot on.
        data : Any
            Data to visualize (numpy array, dict, etc. depending on implementation).
        plot_style : dict, optional
            Matplotlib styling arguments.

        Returns
        -------
        str
            Category tag for filenames.

        """
        ...
