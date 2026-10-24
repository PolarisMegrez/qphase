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


def test_viz_result_carries_only_generated_files(tmp_path):
    engine = VizEngine(
        config=VizEngineConfig(output_dir=str(tmp_path), format="png"),
        plugins={"visualizer": TimeSeriesPlotter(plots=[{"channels": [0]}])},
    )

    result = engine.run(trajectory_bundle())

    assert result.data and all(path.is_file() for path in result.data)
    assert result.metadata == {"count": len(result.data)}
    # The visualization layer has no analysis payload: it only renders files.
    assert not hasattr(result, "analysis")


def test_viz_result_save_is_a_no_op(tmp_path):
    engine = VizEngine(
        config=VizEngineConfig(output_dir=str(tmp_path), format="png"),
        plugins={"visualizer": TimeSeriesPlotter(plots=[{"channels": [0]}])},
    )
    result = engine.run(trajectory_bundle())
    before = sorted(path.name for path in tmp_path.iterdir())

    result.save(tmp_path / "unused")
    result.save(tmp_path / "unused.txt")

    assert sorted(path.name for path in tmp_path.iterdir()) == before
    assert not (tmp_path / "unused").exists()
    assert not (tmp_path / "unused.txt").exists()


def test_viz_engine_config_has_no_specs_field():
    """The engine is configured by plugin instances, not by a spec list."""
    config = VizEngineConfig()

    assert "specs" not in config.model_dump()
