"""qphase_sde: Backend-Neutral Numerical Operations (2.0)
---------------------------------------------------------
Backend-agnostic numerical definitions shared across the SDE resource
package: canonical phase-space coordinate conventions and the backend-aware
primitives used by all integrators. Modules here hold mathematics only —
execution helpers live in ``qphase_sde.runtime``.

Submodules are imported on demand. Note that inside this package the stdlib
``math`` module is shadowed for relative imports only; submodules use absolute
imports, which keep resolving to the stdlib.

Public API
----------
coordinates
    Canonical phase-space real-coordinate conventions.
ops
    Backend-agnostic SDE operations (noise expansion and contraction).
"""

__all__ = [
    "coordinates",
    "ops",
]
