"""Integration tests for the qphase_viz engine."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.core.errors import QPhaseRuntimeError
from qphase_viz.config import VizEngineConfig
from qphase_viz.engine import VizEngine
from qphase_viz.plotters.evolution import TimeSeriesPlotter

pytestmark = pytest.mark.integration


class _FakeTrajectory:
    """Minimal TrajectorySet-like object (no qphase_sde dependency)."""

    def __init__(self, data: np.ndarray, dt: float = 0.1):
        self._data = data
        self.dt = dt

    def to_numpy(self) -> np.ndarray:
        return self._data

    @property
    def times(self) -> np.ndarray:
        return np.arange(self._data.shape[1]) * self.dt


def _trajectory() -> _FakeTrajectory:
    t = np.arange(64) * 0.1
    data = np.exp(-0.1 * t)[None, :, None] * np.exp(1j * t)[None, :, None]
    return _FakeTrajectory(np.repeat(data, 2, axis=0))


def test_viz_engine_manifest():
    assert VizEngine.name == "viz"
    assert VizEngine.manifest.required_plugins == {"visualizer"}
    assert VizEngine.config_schema is VizEngineConfig


def test_viz_engine_run_renders_files(tmp_path):
    engine = VizEngine(
        config=VizEngineConfig(output_dir=str(tmp_path), format="png"),
        plugins={
            "visualizer": TimeSeriesPlotter(
                plots=[{"channels": [0], "transform": "real"}]
            )
        },
    )

    result = engine.run(_trajectory())

    assert result.data, "engine returned no files"
    assert result.metadata["count"] == len(result.data)
    for path in result.data:
        assert path.exists()
        assert path.stat().st_size > 0


def test_viz_engine_requires_input(tmp_path):
    engine = VizEngine(config=VizEngineConfig(output_dir=str(tmp_path)))
    with pytest.raises(QPhaseRuntimeError):
        engine.run(None)
