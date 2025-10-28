from __future__ import annotations

"""Visualizer domain protocols.

Renderer implementations may be functions or classes. The service layer will
normalize and call them via this protocol.
"""

from typing import Any, Dict, Mapping, Protocol


class Renderer(Protocol):
    """Renderer contract.

    validate(spec) -> None (may raise ConfigError)
    render(ax_or_buffer, data, spec, style) -> Mapping[str, Any]

    Return value is standardized metadata containing at least:
    - generated_filename (optional)
    - tags: list[str]
    - duration_ms (optional)
    """

    def validate(self, spec: Mapping[str, Any]) -> None: ...

    def render(self, ax_or_buffer: Any, data: Any, spec: Mapping[str, Any], style: Mapping[str, Any]) -> Mapping[str, Any]: ...
