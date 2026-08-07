"""qphase_sde: First-Passage Observer
---------------------------------------------------------
Online first-passage (first-hit) observer for SDE trajectories.

A single observer instance watches one threshold rule against a scalar
per-trajectory observable computed from the live state ``y`` on device. A hit
is confirmed only after ``debounce_checks`` consecutive due checks satisfy the
condition; the first-hit time is the time of the first check of the
confirming run. Trajectories that never confirm a hit are right-censored:
their ``first_hit_time`` is NaN and ``first_hit_step`` is -1 (never an
infinite or zero sentinel). Only small per-event payloads (indices and values
of newly confirmed hits) are transferred from device to host.

Phase 1 guarantees record-only semantics and whole-batch control-flow actions
(``stop_batch`` / ``fail_job``); per-trajectory early stopping is Phase 2.

Public API
----------
``FirstPassageObserver`` : Online first-passage observer.
``FirstPassageObserverConfig`` : Configuration for the observer.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Literal, cast

import numpy as np
from pydantic import Field, model_validator
from qphase.backend.xputil import convert_to_numpy, get_xp
from qphase.core.protocols import PluginConfigBase

from qphase_sde.coordinates import (
    CANONICAL_COORDINATE_LAYOUT,
    R_CONVENTION,
    canonical_r_coordinates,
    canonical_vector,
)
from qphase_sde.observer.base import Observer, ObserverContext

__all__ = ["FirstPassageObserver", "FirstPassageObserverConfig"]


class FirstPassageObserverConfig(PluginConfigBase):
    """Configuration for the online first-passage observer.

    One instance watches a single rule; configure multiple observer instances
    to watch multiple rules. ``check_interval_steps`` is the cadence in
    integration steps (under adaptive stepping it counts accepted steps).
    """

    rule: Literal[
        "state_norm", "mode_magnitude", "linear_projection", "matrix_projection"
    ] = Field(..., description="Observed quantity family for the threshold rule")
    check_interval_steps: int = Field(
        ...,
        ge=1,
        description=(
            "Cadence in integration steps between checks; under adaptive "
            "stepping this counts accepted steps"
        ),
    )
    action: Literal["record", "stop_batch", "fail_job"] = Field(
        "record",
        description=(
            "What a confirmed hit requests: record only, stop the whole "
            "trajectory batch early, or fail the job"
        ),
    )
    debounce_checks: int = Field(
        1,
        ge=1,
        description=(
            "Consecutive due checks that must satisfy the condition before a "
            "hit is confirmed; the first-hit time is the first check of the "
            "confirming run"
        ),
    )
    direction: Literal["above", "below", "outside", "inside"] = Field(
        ...,
        description=(
            "Threshold direction: above/below for scalar thresholds, "
            "outside/inside for the mode_magnitude interval"
        ),
    )
    threshold: float | None = Field(
        None, description="Scalar threshold for state_norm and projection rules"
    )
    mode: int | None = Field(
        None, ge=0, description="Mode index for the mode_magnitude rule"
    )
    lower: float | None = Field(
        None, description="Inclusive lower interval bound (mode_magnitude)"
    )
    upper: float | None = Field(
        None, description="Inclusive upper interval bound (mode_magnitude)"
    )
    weights: list[complex] | None = Field(
        None,
        description="Complex weights w_i of the linear projection sum(w_i*alpha_i)",
    )
    component: Literal["abs", "real", "imag"] = Field(
        "abs", description="Component of the linear projection that is thresholded"
    )
    reference: list[float] | None = Field(
        None,
        description=(
            "Flat canonical R coordinate vector (length n_modes**2) used as "
            "projection reference; None means the zero matrix"
        ),
    )
    left_vector: list[float] | None = Field(
        None,
        description=(
            "Flat canonical R coordinate vector (length n_modes**2) contracted "
            "with vec(R) - vec(reference)"
        ),
    )

    @model_validator(mode="after")
    def _validate_rule_fields(self) -> FirstPassageObserverConfig:
        if self.rule == "mode_magnitude":
            if self.direction not in ("outside", "inside"):
                raise ValueError(
                    "mode_magnitude requires direction 'outside' or 'inside'"
                )
            if self.mode is None:
                raise ValueError("mode_magnitude requires 'mode'")
            if self.lower is None and self.upper is None:
                raise ValueError(
                    "mode_magnitude requires at least one of 'lower'/'upper'"
                )
            if (
                self.lower is not None
                and self.upper is not None
                and self.lower > self.upper
            ):
                raise ValueError("mode_magnitude requires lower <= upper")
        else:
            if self.direction not in ("above", "below"):
                raise ValueError(
                    f"rule {self.rule!r} requires direction 'above' or 'below'"
                )
            if self.threshold is None:
                raise ValueError(f"rule {self.rule!r} requires 'threshold'")
        if self.rule == "linear_projection":
            if not self.weights:
                raise ValueError("linear_projection requires non-empty 'weights'")
        if self.rule == "matrix_projection":
            if self.left_vector is None:
                raise ValueError("matrix_projection requires 'left_vector'")
            root = math.isqrt(len(self.left_vector))
            if root * root != len(self.left_vector):
                raise ValueError(
                    "matrix_projection 'left_vector' length must be a perfect "
                    "square (n_modes**2)"
                )
            if self.reference is not None and len(self.reference) != len(
                self.left_vector
            ):
                raise ValueError(
                    "matrix_projection 'reference' must have the same length as "
                    "'left_vector' (n_modes**2)"
                )
        return self


class FirstPassageObserver(Observer):
    """Online first-passage observer over a scalar per-trajectory observable.

    The observable is evaluated with backend array ops on the live state at
    every due check step (``step % check_interval_steps == 0``, including the
    initial condition at step 0). Per-trajectory debounce state is tracked on
    device; only the indices and values of newly confirmed hits are copied to
    the host. ``observe`` never mutates the state and never consumes RNG.
    """

    name: ClassVar[str] = "first_passage"
    description: ClassVar[str] = (
        "Online first-passage observer with debounced threshold rules"
    )
    config_schema: ClassVar[type[FirstPassageObserverConfig]] = (
        FirstPassageObserverConfig
    )
    per_trajectory_keys: ClassVar[tuple[str, ...]] = (
        "hit",
        "censored",
        "first_hit_time",
        "first_hit_step",
        "value_before_hit",
        "value_at_hit",
        "effective_end_time",
    )

    def __init__(
        self, config: FirstPassageObserverConfig | None = None, **kwargs: Any
    ) -> None:
        super().__init__(config, **kwargs)

    @property
    def _cfg(self) -> FirstPassageObserverConfig:
        return cast(FirstPassageObserverConfig, self.config)

    @property
    def check_interval_steps(self) -> int:
        """Observer cadence in integration steps."""
        return int(self._cfg.check_interval_steps)

    def initialize(self, context: ObserverContext) -> None:
        """Validate rule operands against the run and reset all state."""
        cfg = self._cfg
        n_modes = int(context.n_modes)
        if cfg.rule == "mode_magnitude":
            mode = int(cast(int, cfg.mode))
            if not 0 <= mode < n_modes:
                raise ValueError(
                    f"mode_magnitude mode={mode} is out of range for n_modes={n_modes}"
                )
        elif cfg.rule == "linear_projection":
            weights = cast(list[complex], cfg.weights)
            if len(weights) != n_modes:
                raise ValueError(
                    f"linear_projection 'weights' length {len(weights)} does "
                    f"not match n_modes={n_modes}"
                )
        elif cfg.rule == "matrix_projection":
            n_coordinates = n_modes * n_modes
            self._left_vector = canonical_vector(
                cast(list[float], cfg.left_vector),
                n_coordinates,
                "matrix_projection 'left_vector'",
            )
            self._reference = (
                np.zeros(n_coordinates)
                if cfg.reference is None
                else canonical_vector(
                    cfg.reference, n_coordinates, "matrix_projection 'reference'"
                )
            )

        n_traj = int(context.n_traj)
        self._n_traj = n_traj
        self._integration_t0 = float(context.integration_t0)
        default_end = float(context.integration_t0) + int(context.total_steps) * float(
            context.dt
        )
        self._hit = np.zeros(n_traj, dtype=bool)
        self._first_hit_time = np.full(n_traj, np.nan)
        self._first_hit_step = np.full(n_traj, -1, dtype=np.int64)
        self._value_before_hit = np.full(n_traj, np.nan)
        self._value_at_hit = np.full(n_traj, np.nan)
        self._effective_end_time = np.full(n_traj, default_end)
        self._checks_seen = 0
        self._device: dict[str, Any] | None = None

    def observe(self, y: Any, t: float, step: int) -> str | None:
        """Check the rule against the live state at a due step.

        Returns the configured action when at least one trajectory newly
        confirms a hit and the action is not ``"record"``; otherwise None.
        """
        cfg = self._cfg
        if step % int(cfg.check_interval_steps) != 0:
            return None
        xp = get_xp(y)
        state = self._device_state(xp, y)
        values = self._observable(xp, y, state)
        active = ~state["hit"]
        qualifying = self._qualifying(xp, values) & active
        run_start = qualifying & (state["run"] == 0)
        before = values if self._checks_seen == 0 else state["last_value"]
        state["value_before_run"] = xp.where(
            run_start, before, state["value_before_run"]
        )
        state["run_start_time"] = xp.where(run_start, float(t), state["run_start_time"])
        state["run_start_step"] = xp.where(
            run_start, int(step), state["run_start_step"]
        )
        state["run"] = xp.where(qualifying, state["run"] + 1, 0)
        newly = qualifying & (state["run"] == int(cfg.debounce_checks)) & active
        new_index = xp.nonzero(newly)[0]
        n_new = int(new_index.size)
        if n_new:
            state["hit"] = state["hit"] | newly
            host_index = convert_to_numpy(new_index)
            self._hit[host_index] = True
            self._first_hit_time[host_index] = convert_to_numpy(
                state["run_start_time"][new_index]
            )
            self._first_hit_step[host_index] = convert_to_numpy(
                state["run_start_step"][new_index]
            )
            self._value_before_hit[host_index] = convert_to_numpy(
                state["value_before_run"][new_index]
            )
            self._value_at_hit[host_index] = convert_to_numpy(values[new_index])
        state["last_value"] = xp.where(state["hit"], state["last_value"], values)
        self._checks_seen += 1
        if n_new and cfg.action != "record":
            return str(cfg.action)
        return None

    def note_end_of_run(self, t: float, step: int) -> None:
        """Record the effective run end time for every trajectory."""
        del step
        self._effective_end_time = np.full(self._n_traj, float(t))

    def finalize(self) -> dict[str, Any]:
        """Return per-trajectory first-passage results and the rule echo."""
        cfg = self._cfg
        hit = self._hit.copy()
        censored = ~hit
        payload: dict[str, Any] = {
            "status": "ok",
            "observer": self.name,
            "rule": cfg.rule,
            "action": cfg.action,
            "direction": cfg.direction,
            "debounce_checks": int(cfg.debounce_checks),
            "check_interval_steps": int(cfg.check_interval_steps),
            "observable": self._observable_description(),
            "n_traj": self._n_traj,
            "integration_t0": self._integration_t0,
            "time_unit": "seconds",
            "hit": hit,
            "censored": censored,
            "first_hit_time": self._first_hit_time.copy(),
            "first_hit_step": self._first_hit_step.copy(),
            "value_before_hit": self._value_before_hit.copy(),
            "value_at_hit": self._value_at_hit.copy(),
            "effective_end_time": self._effective_end_time.copy(),
            "n_hit": int(np.count_nonzero(hit)),
            "n_censored": int(np.count_nonzero(censored)),
        }
        if cfg.rule == "state_norm":
            payload["threshold"] = float(cast(float, cfg.threshold))
        elif cfg.rule == "mode_magnitude":
            payload.update(
                {
                    "mode": int(cast(int, cfg.mode)),
                    "lower": cfg.lower,
                    "upper": cfg.upper,
                }
            )
        elif cfg.rule == "linear_projection":
            payload.update(
                {
                    "weights": list(cast(list[complex], cfg.weights)),
                    "component": cfg.component,
                    "threshold": float(cast(float, cfg.threshold)),
                }
            )
        else:
            payload.update(
                {
                    "threshold": float(cast(float, cfg.threshold)),
                    "coordinate_layout": CANONICAL_COORDINATE_LAYOUT,
                    "convention": R_CONVENTION,
                    "left_vector": self._left_vector.copy(),
                    "reference_vector": self._reference.copy(),
                }
            )
        return payload

    def _observable_description(self) -> str:
        cfg = self._cfg
        if cfg.rule == "state_norm":
            return "state_norm"
        if cfg.rule == "mode_magnitude":
            return f"|alpha_{int(cast(int, cfg.mode))}|"
        if cfg.rule == "linear_projection":
            return f"{cfg.component}(w . alpha)"
        return "left_vector . (vec(R) - vec(reference))"

    def _device_state(self, xp: Any, y: Any) -> dict[str, Any]:
        """Lazily allocate per-trajectory device state on y's array namespace."""
        if self._device is not None:
            return self._device
        n_traj = self._n_traj
        real_dtype = y.real.dtype if hasattr(y, "real") else float
        state: dict[str, Any] = {
            "hit": xp.zeros(n_traj, dtype=bool),
            "run": xp.zeros(n_traj, dtype=np.int64),
            "value_before_run": xp.zeros(n_traj, dtype=real_dtype),
            "run_start_time": xp.zeros(n_traj, dtype=np.float64),
            "run_start_step": xp.zeros(n_traj, dtype=np.int64),
            "last_value": xp.zeros(n_traj, dtype=real_dtype),
        }
        rule = self._cfg.rule
        if rule == "linear_projection":
            state["weights"] = xp.asarray(
                np.asarray(cast(list[complex], self._cfg.weights), dtype=np.complex128)
            )
        elif rule == "matrix_projection":
            state["left_vector"] = xp.asarray(self._left_vector)
            state["reference"] = xp.asarray(self._reference)
        self._device = state
        return state

    def _observable(self, xp: Any, y: Any, state: dict[str, Any]) -> Any:
        """Scalar per-trajectory observable, computed on device."""
        cfg = self._cfg
        if cfg.rule == "state_norm":
            return xp.sqrt(xp.sum(xp.abs(y) ** 2, axis=-1))
        if cfg.rule == "mode_magnitude":
            return xp.abs(y[:, int(cast(int, cfg.mode))])
        if cfg.rule == "linear_projection":
            projection = xp.sum(y * state["weights"][None, :], axis=-1)
            if cfg.component == "real":
                return xp.real(projection)
            if cfg.component == "imag":
                return xp.imag(projection)
            return xp.abs(projection)
        coordinates = canonical_r_coordinates(y)
        return (coordinates - state["reference"]) @ state["left_vector"]

    def _qualifying(self, xp: Any, values: Any) -> Any:
        """Boolean mask of trajectories satisfying the threshold condition."""
        cfg = self._cfg
        if cfg.rule == "mode_magnitude":
            if cfg.direction == "outside":
                result = xp.zeros(values.shape, dtype=bool)
                if cfg.lower is not None:
                    result = result | (values < float(cfg.lower))
                if cfg.upper is not None:
                    result = result | (values > float(cfg.upper))
                return result
            result = xp.ones(values.shape, dtype=bool)
            if cfg.lower is not None:
                result = result & (values >= float(cfg.lower))
            if cfg.upper is not None:
                result = result & (values <= float(cfg.upper))
            return result
        threshold = float(cast(float, cfg.threshold))
        if cfg.direction == "above":
            return values > threshold
        return values < threshold
