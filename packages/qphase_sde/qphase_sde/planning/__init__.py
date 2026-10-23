"""Resource-aware execution planning for logical SDE scans.

The plan records live in :mod:`plans`, the memory accounting in :mod:`memory`
and the assembly of one plan in :mod:`builder`. This package is the import
surface: callers ask for a plan type or for ``build_execution_plan`` here and
never need to know which module holds it.
"""

from .builder import build_execution_plan, typed_batch_scope, typed_scan_scope
from .plans import (
    SDEExecutionPlan,
    SDEMemoryEstimate,
    SDEMemoryPlanningError,
    SDETimeGrid,
    resolve_time_grid,
)

__all__ = [
    "SDEExecutionPlan",
    "SDEMemoryEstimate",
    "SDEMemoryPlanningError",
    "SDETimeGrid",
    "build_execution_plan",
    "resolve_time_grid",
    "typed_batch_scope",
    "typed_scan_scope",
]
