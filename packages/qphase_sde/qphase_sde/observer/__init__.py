"""qphase_sde: Observer Subpackage
---------------------------------------------------------
Online observers that watch the live SDE integration state. Observers follow
the ``initialize -> observe -> finalize`` lifecycle and may request
whole-batch control flow (early stop or job failure) without ever mutating
the numerical state or consuming RNG.

Public API
----------
``SDEObserverProtocol`` : Protocol for online observers.
``Observer`` : Base class for observers.
``ObserverContext`` : Run-scoped context handed to observers at initialize.
``ObserverDecision`` : Structured whole-batch control-flow request.
``ObserverTriggeredError`` : Generic observer failure.
``FirstPassageTriggeredError`` : Compatibility alias for observer failures.
``FirstPassageObserver`` : Online first-passage observer.
``FirstPassageObserverConfig`` : Configuration for the first-passage observer.
"""

from .base import (
    Observer,
    ObserverContext,
    ObserverDecision,
    ObserverTriggeredError,
    SDEObserverProtocol,
)
from .first_passage import (
    FirstPassageObserver,
    FirstPassageObserverConfig,
    FirstPassageTriggeredError,
)

__all__ = [
    "FirstPassageObserver",
    "FirstPassageObserverConfig",
    "FirstPassageTriggeredError",
    "Observer",
    "ObserverContext",
    "ObserverDecision",
    "ObserverTriggeredError",
    "SDEObserverProtocol",
]
