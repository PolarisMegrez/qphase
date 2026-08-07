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
``FirstPassageTriggeredError`` : Raised for ``action="fail_job"`` hits.
``FirstPassageObserver`` : Online first-passage observer.
``FirstPassageObserverConfig`` : Configuration for the first-passage observer.
"""

from .base import (
    FirstPassageTriggeredError,
    Observer,
    ObserverContext,
    SDEObserverProtocol,
)
from .first_passage import FirstPassageObserver, FirstPassageObserverConfig

__all__ = [
    "FirstPassageObserver",
    "FirstPassageObserverConfig",
    "FirstPassageTriggeredError",
    "Observer",
    "ObserverContext",
    "SDEObserverProtocol",
]
