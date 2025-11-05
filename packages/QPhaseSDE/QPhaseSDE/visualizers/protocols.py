"""
QPhaseSDE: Visualizer Protocols
-------------------------------
Contracts for renderer implementations that validate specs and render outputs
to figure-like targets or buffers, returning standardized metadata.

Behavior
--------
- Render targets may be Matplotlib axes, off-screen buffers, or backend-
  specific objects. 

Notes
-----
- Avoid importing heavy visualizer libraries at module import time.
"""

from typing import Any, Dict, Mapping, Protocol

__all__ = [
  "Renderer",
]

class Renderer(Protocol):
  """
  Protocol for visualizer renderer implementations.

  Defines the contract for validating visualizer specs and rendering outputs to figure-like targets or buffers.
  Implementations must return standardized metadata for downstream analysis.

  Parameters
  ----------
  Inherits all parameters from Protocol.

  Attributes
  ----------
  None (protocol only).

  Methods
  -------
  validate
  render

  Examples
  --------
  >>> class MyRenderer(Renderer):
  ...     def validate(self, spec): ...
  ...     def render(self, ax, data, spec, style): ...
  """

  def validate(self, spec: Mapping[str, Any]) -> None:
    """
    Validate the visualizer specification.

    Parameters
    ----------
    spec : Mapping[str, Any]
      Visualization specification dictionary.

    Returns
    -------
    None

    Raises
    ------
    QPSConfigError
      If the spec is invalid.
    """
    ...

  def render(self, ax_or_buffer: Any, data: Any, spec: Mapping[str, Any], style: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Render the visualizer to a target and return standardized metadata.

    Parameters
    ----------
    ax_or_buffer : Any
      Target for rendering (e.g., Matplotlib axes, buffer).
    data : Any
      Data to visualize.
    spec : Mapping[str, Any]
      Visualization specification.
    style : Mapping[str, Any]
      Style parameters for rendering.

    Returns
    -------
    dict
      Metadata including at least 'tags' (list[str]), and optionally 'generated_filename', 'duration_ms'.

    Examples
    --------
    >>> renderer = MyRenderer()
    >>> meta = renderer.render(ax, data, spec, style)
    """
    ...