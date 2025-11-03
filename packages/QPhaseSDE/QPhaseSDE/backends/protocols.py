"""
QPhaseSDE: Backend Protocols
----------------------------
Abstract contracts for backend implementations shared across the engine and
domain modules. Defines the minimal `BackendBase` interface and optional
`ExtendedBackend` capabilities without importing any concrete array library.

Behavior
--------
- (*Design principles*) Keep the interface surface minimal and stable;
  advanced helpers or specialized extensions should be implemented in
  `ExtendedBackend` rather than the base interface.
- (*Capability introspection*) The `capabilities()` method must expose backend
  features using unified, backend-agnostic keys for consistent discovery and
  compatibility checks.

Notes
-----
- This module is dependency-light and safe to import in any environment.
  Do not import `numpy`, `torch`, `cupy`, or similar libraries here.
"""

from typing import Any, Dict, Optional, Protocol, Tuple
from ..core.protocols import BackendBase

__all__ = [
  "ExtendedBackend",
]

class ExtendedBackend(BackendBase, Protocol):
    """Optional backend protocol for advanced capabilities.

    Provides non-essential helpers that backends may expose in addition to the
    minimal `BackendBase` surface. Core components don't require these methods,
    but states, visualizers, or analysis code can opportunistically use them
    when present to improve performance or ergonomics.

    Methods
    -------
    stack(arrays, axis=0) -> Any
        Stack arrays along a new axis.
    to_device(x, device) -> Any
        Move or adapt an array to the target device if supported.
    complex_view(x_realimag) -> Any
        Create a zero-copy complex view from real/imag storage if supported.
    real_imag_split(x_complex) -> Tuple[Any, Any]
        Split a complex array into real/imag zero-copy views if supported.
    capabilities() -> Dict[str, Any]
        Report optional features using backend-agnostic keys.
    """

    # Commonly used helpers
    def stack(self, arrays: Tuple[Any, ...], axis: int = 0) -> Any: ...
    def to_device(self, x: Any, device: Optional[str]) -> Any: ...
    def complex_view(self, x_realimag: Any) -> Any: ...  # zero-copy complex view if supported
    def real_imag_split(self, x_complex: Any) -> Tuple[Any, Any]: ...  # zero-copy split if supported

    # Introspection of capabilities
    def capabilities(self) -> Dict[str, Any]: ...
