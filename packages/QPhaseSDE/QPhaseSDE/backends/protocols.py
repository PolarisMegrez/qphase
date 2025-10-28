from __future__ import annotations

"""Backend domain-level protocols.

Inherits minimal interfaces from core.protocols.BackendBase and declares
optional capabilities that concrete backends may implement for performance.

Implementer guide
-----------------
- Implement all methods in BackendBase.
- Optional methods should raise BackendCapabilityError if explicitly called
  when not supported, or be omitted entirely.
- Provide capability metadata via a 'capabilities() -> dict' method when
  possible to aid factories in selecting optimal paths.
"""

from typing import Any, Dict, Optional, Protocol, Tuple

from ..core.protocols import BackendBase


class ExtendedBackend(BackendBase, Protocol):
    """Optional/extended backend capabilities (may not be present).

    These are not required by core but can be used opportunistically by states
    and visualizers when available.
    """

    # Commonly used helpers
    def stack(self, arrays: Tuple[Any, ...], axis: int = 0) -> Any: ...
    def to_device(self, x: Any, device: Optional[str]) -> Any: ...
    def complex_view(self, x_realimag: Any) -> Any: ...  # zero-copy complex view if supported
    def real_imag_split(self, x_complex: Any) -> Tuple[Any, Any]: ...  # zero-copy split if supported

    # Introspection of capabilities
    def capabilities(self) -> Dict[str, Any]: ...
