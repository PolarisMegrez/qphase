"""qphase_sde: Public Contracts (2.0)
---------------------------------------------------------
Public protocol, quantity and capability contracts of the SDE resource
package. This package contains declarations only: it must never import
concrete plugins, models, backends or runtime helpers.

Public API
----------
quantities
    SDE quantities, frequency orientation and product schema templates.
bundle
    SDEDataBundle contract and provenance.
analyser
    The 2.0 analyser contract (capabilities, workspace, work estimate).
reducer
    The unified reducer lifecycle.
peaks
    Peak candidate, fit-parameter and path contracts.
coherence
    Coherence-frequency estimate contracts.
tasks
    Frozen SDE engine task profiles (approved design, Phase 2 enforcement).
migration
    1.x → 2.x migration tables and the one-shot config converter.
"""

from . import (
    analyser,
    bundle,
    coherence,
    migration,
    peaks,
    quantities,
    reducer,
    tasks,
)

__all__ = [
    "analyser",
    "bundle",
    "coherence",
    "migration",
    "peaks",
    "quantities",
    "reducer",
    "tasks",
]
