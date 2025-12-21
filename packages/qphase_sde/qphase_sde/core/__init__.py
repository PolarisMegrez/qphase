"""qphase_sde: Core Subpackage
--------------------------
Lightweight core containing protocols (interfaces), registry, engine, errors,
and cross-package utilities.

Public API
----------
- Engine: Object-oriented engine class with dependency injection (v0.2)
- run: Functional interface for engine (legacy, but still supported)
"""

from .engine import Engine

__all__ = [
    "Engine",
]
