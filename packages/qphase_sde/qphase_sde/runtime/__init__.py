"""qphase_sde: Runtime Execution Helpers (2.0)
---------------------------------------------------------
Package-private execution helpers of the SDE resource package: fused scan
adapters, point-view splitting and the integration buffer cache. Modules here
serve the engine's execution path; they are not public data contracts — those
live in ``qphase_sde.contracts``.

Submodules are imported on demand so that importing this package never pulls
backend-specific machinery before it is needed.

Public API
----------
batch
    In-memory point views over fused SDE scan results.
buffers
    Backend-agnostic integration buffer cache.
migrate
    One-way offline 1.x result migration to 2.0 typed products.
scan
    ParameterGrid adapter and logical scan dataset container.
"""

__all__ = [
    "batch",
    "buffers",
    "migrate",
    "scan",
]
