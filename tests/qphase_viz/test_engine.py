"""Integration tests for the qphase_viz engine."""

from __future__ import annotations

import pytest
from qphase.core.errors import QPhaseRuntimeError
from qphase_viz.config import VizEngineConfig
from qphase_viz.engine import VizEngine
from qphase_viz.plotters.evolution import TimeSeriesPlotter

from tests.qphase_viz.typed_fixtures import restored, trajectory_bundle

pytestmark = pytest.mark.integration


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

    result = engine.run(trajectory_bundle())

    assert result.data, "engine returned no files"
    assert result.metadata["count"] == len(result.data)
    for path in result.data:
        assert path.exists()
        assert path.stat().st_size > 0


def test_viz_engine_run_renders_restored_bundle(tmp_path):
    artifact = restored(trajectory_bundle(), tmp_path / "artifact")
    engine = VizEngine(
        config=VizEngineConfig(output_dir=str(tmp_path / "plots"), format="png"),
        plugins={
            "visualizer": TimeSeriesPlotter(
                plots=[{"channels": [0], "transform": "real"}]
            )
        },
    )

    result = engine.run(artifact)

    assert result.data and result.data[0].is_file()


def test_viz_engine_requires_input(tmp_path):
    engine = VizEngine(config=VizEngineConfig(output_dir=str(tmp_path)))
    with pytest.raises(QPhaseRuntimeError):
        engine.run(None)
