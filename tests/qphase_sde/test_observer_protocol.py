"""Generic online-observer contracts independent of first-passage semantics."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.protocols import PluginConfigBase
from qphase_sde.engine import Engine, EngineConfig
from qphase_sde.integrator.euler_maruyama import EulerMaruyama
from qphase_sde.observer import (
    Observer,
    ObserverContext,
    ObserverDecision,
    ObserverTriggeredError,
)

pytestmark = pytest.mark.integration


class _Config(PluginConfigBase):
    check_interval_steps: int = 1


class _GenericObserver(Observer):
    name: ClassVar[str] = "generic"
    config_schema: ClassVar[type[_Config]] = _Config
    per_trajectory_keys: ClassVar[tuple[str, ...]] = ("score",)

    @property
    def check_interval_steps(self) -> int:
        return self.config.check_interval_steps

    def initialize(self, context: ObserverContext) -> None:
        self.n_traj = context.n_traj

    def observe(self, y: Any, t: float, step: int) -> ObserverDecision | None:
        del y
        if step != 1:
            return None
        return ObserverDecision(
            action="fail_job",
            message="generic condition reached",
            details={"metric": "custom", "event_time": t},
        )

    def finalize(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "observer": self.name,
            "n_traj": self.n_traj,
            "score": np.arange(self.n_traj, dtype=float),
        }


class _StaticModel:
    name = "static"
    n_modes = 1
    noise_dim = 1
    noise_basis = "real"
    params: dict[str, Any] = {}

    @staticmethod
    def drift(y, t, params):
        del t, params
        return np.zeros_like(y)

    @staticmethod
    def diffusion(y, t, params):
        del t, params
        return np.zeros(y.shape + (1,))


def test_generic_observer_controls_engine_without_specialized_fields():
    engine = Engine(
        config=EngineConfig(t0=0.0, t1=0.2, dt=0.1, n_traj=2, ic=[[0.0]]),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": _StaticModel(),
            "observer": {"generic": _GenericObserver()},
        },
    )

    with pytest.raises(ObserverTriggeredError) as excinfo:
        engine.run()

    assert "generic condition reached" in str(excinfo.value)
    assert excinfo.value.payload["observers"]["generic"] == {
        "metric": "custom",
        "event_time": pytest.approx(0.1),
    }


def test_generic_observer_owns_trajectory_batch_payload_merge():
    observer = _GenericObserver()
    merged = observer.merge_payloads(
        [
            {"status": "ok", "observer": "generic", "n_traj": 2, "score": [1, 2]},
            {"status": "ok", "observer": "generic", "n_traj": 1, "score": [3]},
        ]
    )

    assert merged["n_traj"] == 3
    np.testing.assert_array_equal(merged["score"], [1, 2, 3])


def test_generic_observer_owns_fused_scan_payload_split():
    observer = _GenericObserver()
    payloads = observer.split_payload(
        {
            "status": "ok",
            "observer": "generic",
            "n_traj": 5,
            "score": np.arange(5),
        },
        [2, 3],
    )

    assert [payload["n_traj"] for payload in payloads] == [2, 3]
    np.testing.assert_array_equal(payloads[0]["score"], [0, 1])
    np.testing.assert_array_equal(payloads[1]["score"], [2, 3, 4])
