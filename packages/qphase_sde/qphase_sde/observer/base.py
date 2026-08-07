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
``ObserverDecision`` : Structured whole-batch control-flow request.
``ObserverTriggeredError`` : Raised for an observer ``fail_job`` decision.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np
from qphase.backend.base import BackendBase
from qphase.core.protocols import PluginBase, PluginConfigBase

__all__ = [
    "Observer",
    "ObserverContext",
    "ObserverDecision",
    "ObserverTriggeredError",
    "SDEObserverProtocol",
]


class ObserverTriggeredError(RuntimeError):
    """Raised when an observer requests the generic ``fail_job`` action.

    Attributes
    ----------
    payload : dict[str, Any]
        Structured triggering time/step and plugin-owned decision details,
        grouped by observer instance name.

    """

    def __init__(self, message: str, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.payload: dict[str, Any] = {} if payload is None else payload


@dataclass(frozen=True)
class ObserverDecision:
    """Structured control-flow request returned by an online observer."""

    action: Literal["stop_batch", "fail_job"]
    message: str
    details: dict[str, Any]


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

    def observe(self, y: Any, t: float, step: int) -> ObserverDecision | None:
        """Inspect the live state ``y`` at time ``t`` / integration ``step``.

        ``y`` is the backend state array of shape ``(n_traj, n_modes)``; it
        must not be mutated and no RNG may be consumed. Returns an
        :class:`ObserverDecision` when whole-batch control flow is requested,
        otherwise ``None``.
        """
        ...

    def note_end_of_run(self, t: float, step: int) -> None:
        """Record the effective run end before ``finalize`` is called."""
        ...

    def finalize(self) -> dict[str, Any]:
        """Return the result payload for the current run."""
        ...

    def merge_payloads(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge trajectory-batch payloads using observer-owned semantics."""
        ...

    def split_payload(
        self, payload: dict[str, Any], group_sizes: list[int]
    ) -> list[dict[str, Any]]:
        """Split a fused scan payload using observer-owned semantics."""
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
    #: Scalar/schema fields that must agree across trajectory batches.
    invariant_payload_keys: ClassVar[tuple[str, ...]] = ("status", "observer")

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
    def observe(self, y: Any, t: float, step: int) -> ObserverDecision | None:
        """Inspect the live state and return a control-flow action or None."""

    def note_end_of_run(self, t: float, step: int) -> None:
        """Record the effective run end; the default implementation is a no-op."""

    @abstractmethod
    def finalize(self) -> dict[str, Any]:
        """Return the result payload for the current run."""

    def merge_payloads(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        """Concatenate declared per-trajectory fields across ordered batches."""
        if not payloads:
            raise ValueError("no observer payloads to merge")
        merged = dict(payloads[0])
        for payload in payloads[1:]:
            for key in self.invariant_payload_keys:
                if payload.get(key) != merged.get(key):
                    raise RuntimeError(
                        "observer payload invariant mismatch across trajectory "
                        f"batches for key {key!r}: {merged.get(key)!r} != "
                        f"{payload.get(key)!r}"
                    )
        for key in self.per_trajectory_keys:
            missing = [
                index for index, payload in enumerate(payloads) if key not in payload
            ]
            if missing:
                raise RuntimeError(
                    f"observer payload key {key!r} is missing from batches {missing}"
                )
            merged[key] = np.concatenate(
                [np.asarray(payload[key]) for payload in payloads], axis=0
            )
        if self.per_trajectory_keys:
            merged["n_traj"] = int(
                np.asarray(merged[self.per_trajectory_keys[0]]).shape[0]
            )
        else:
            merged["n_traj"] = sum(
                int(payload.get("n_traj", 0)) for payload in payloads
            )
        return self._finalize_group_payload(merged)

    def split_payload(
        self, payload: dict[str, Any], group_sizes: list[int]
    ) -> list[dict[str, Any]]:
        """Create ordered per-scan-point payloads from a fused ensemble."""
        if not group_sizes or any(int(size) < 1 for size in group_sizes):
            raise ValueError("observer payload group sizes must be positive")
        expected = sum(int(size) for size in group_sizes)
        actual = int(payload.get("n_traj", expected))
        if actual != expected:
            raise RuntimeError(
                "observer payload trajectory count does not match scan groups: "
                f"{actual} != {expected}"
            )
        for key in self.per_trajectory_keys:
            if key not in payload:
                raise RuntimeError(
                    f"observer payload key {key!r} is required for scan splitting"
                )
            if int(np.asarray(payload[key]).shape[0]) != expected:
                raise RuntimeError(
                    f"observer payload key {key!r} has an invalid trajectory axis"
                )

        groups: list[dict[str, Any]] = []
        start = 0
        for raw_size in group_sizes:
            size = int(raw_size)
            stop = start + size
            group = dict(payload)
            for key in self.per_trajectory_keys:
                group[key] = np.asarray(payload[key])[start:stop].copy()
            group["n_traj"] = size
            groups.append(self._finalize_group_payload(group))
            start = stop
        return groups

    def _finalize_group_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Recompute observer-specific aggregate fields after merge/split."""
        return payload
