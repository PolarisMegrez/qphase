"""qphase_sde: Observer Base Classes
---------------------------------------------------------
Base classes for online SDE observers. Observers watch the live integration
state during ``Engine.run_sde`` and follow the lifecycle
``initialize -> observe -> finalize``. They never mutate the state array and
never consume RNG, so attaching an observer does not change the numerics of
trajectories that do not trigger a control-flow action.

Public API
----------
``SDEObserverProtocol`` : Protocol for online observers.
``Observer`` : Base class for observers.
``ObserverContext`` : Run-scoped context handed to observers at initialize.
``FirstPassageTriggeredError`` : Raised for ``action="fail_job"`` hits.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from qphase.backend.base import BackendBase
from qphase.core.protocols import PluginBase, PluginConfigBase

__all__ = [
    "FirstPassageTriggeredError",
    "Observer",
    "ObserverContext",
    "SDEObserverProtocol",
]


class FirstPassageTriggeredError(RuntimeError):
    """Raised when an observer with ``action="fail_job"`` confirms a hit.

    Attributes
    ----------
    payload : dict[str, Any]
        Structured event details: the triggering time/step and, per observer,
        the rule, the hit trajectory ids, and their first-hit times.

    """

    def __init__(self, message: str, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.payload: dict[str, Any] = {} if payload is None else payload


@dataclass(frozen=True)
class ObserverContext:
    """Immutable run description passed to ``Observer.initialize``.

    Attributes
    ----------
    n_traj : int
        Number of trajectories integrated simultaneously.
    n_modes : int
        Number of modes of the live state array ``y``.
    record_modes : tuple[int, ...]
        Mode indices retained in the output trajectory.
    dt : float
        Fixed integration step. Under adaptive stepping this is the initial
        step and observer cadence counts accepted steps instead.
    integration_t0 : float
        Physical integration start time.
    observation_t0 : float
        Observation (recording) start time.
    total_steps : int
        Total number of integration steps of the run.
    backend : BackendBase
        Active backend, so observers can allocate device-side temporaries.

    """

    n_traj: int
    n_modes: int
    record_modes: tuple[int, ...]
    dt: float
    integration_t0: float
    observation_t0: float
    total_steps: int
    backend: BackendBase


@runtime_checkable
class SDEObserverProtocol(Protocol):
    """Protocol for online SDE observers.

    The engine calls ``initialize`` once before the integration loop,
    ``observe`` with the live state at the initial condition and at every due
    check step, ``note_end_of_run`` once when the loop ends (normally or via
    ``stop_batch``/``fail_job``), and ``finalize`` to collect the result
    payload.
    """

    @property
    def check_interval_steps(self) -> int:
        """Observer cadence in integration steps (>= 1)."""
        ...

    def initialize(self, context: ObserverContext) -> None:
        """Reset internal state for a new run described by ``context``."""
        ...

    def observe(self, y: Any, t: float, step: int) -> str | None:
        """Inspect the live state ``y`` at time ``t`` / integration ``step``.

        ``y`` is the backend state array of shape ``(n_traj, n_modes)``; it
        must not be mutated and no RNG may be consumed. Returns the configured
        action string (``"stop_batch"`` or ``"fail_job"``) when a newly
        confirmed hit requests control flow, else ``None`` (``"record"`` hits
        are tracked internally and also return ``None``).
        """
        ...

    def note_end_of_run(self, t: float, step: int) -> None:
        """Record the effective run end before ``finalize`` is called."""
        ...

    def finalize(self) -> dict[str, Any]:
        """Return the result payload for the current run."""
        ...


class Observer(PluginBase, ABC):
    """Base class for online SDE observers.

    All observers must inherit from this class and implement the
    ``initialize``/``observe``/``finalize`` lifecycle.
    """

    config_schema: ClassVar[type[PluginConfigBase]]
    #: Keys of the finalize payload that hold per-trajectory arrays of length
    #: ``n_traj``; the engine concatenates these across trajectory batches.
    per_trajectory_keys: ClassVar[tuple[str, ...]] = ()

    def __init__(self, config: PluginConfigBase | None = None, **kwargs: Any):
        """Initialize the observer.

        Parameters
        ----------
        config : PluginConfigBase, optional
            Configuration object. If None, created from kwargs.
        **kwargs : Any
            Configuration parameters if config is not provided.

        """
        if config is None:
            self.config = self.config_schema(**kwargs)
        else:
            self.config = config

    @property
    @abstractmethod
    def check_interval_steps(self) -> int:
        """Observer cadence in integration steps (>= 1)."""

    @abstractmethod
    def initialize(self, context: ObserverContext) -> None:
        """Reset internal state for a new run described by ``context``."""

    @abstractmethod
    def observe(self, y: Any, t: float, step: int) -> str | None:
        """Inspect the live state and return a control-flow action or None."""

    def note_end_of_run(self, t: float, step: int) -> None:
        """Record the effective run end; the default implementation is a no-op."""

    @abstractmethod
    def finalize(self) -> dict[str, Any]:
        """Return the result payload for the current run."""
