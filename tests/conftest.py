"""Pytest configuration and shared fixtures.

Packages under ``packages/`` are installed editable via the uv workspace, and the
repository root is placed on ``sys.path`` automatically by pytest (``tests/`` is a
package), so no manual ``sys.path`` manipulation is needed here.
"""

import os
from pathlib import Path

import pytest
from qphase.core.project import ProjectContext
from qphase.core.registry import registry
from qphase.core.system_config import load_system_config

# Layer markers used for test selection (registered in pyproject.toml).
_LAYER_MARKERS = ("unit", "integration", "e2e", "gpu", "slow")


def pytest_collection_modifyitems(items):
    """Assign the ``unit`` marker to tests without an explicit layer marker."""
    for item in items:
        if not any(marker in item.keywords for marker in _LAYER_MARKERS):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="session", autouse=True)
def setup_env():
    """Set up environment variables for testing."""
    # Ensure we don't accidentally use user's config
    os.environ["QPHASE_CONFIG"] = ""
    os.environ["QPHASE_SYSTEM_CONFIG"] = ""
    os.environ["QPHASE_PROJECT"] = str(Path(__file__).resolve().parents[1])
    yield


@pytest.fixture(autouse=True)
def isolated_user_home(tmp_path, monkeypatch):
    """Keep per-user private state (tags, views, locations) out of the real home."""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "user-home"))


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace with config and output directories."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    config_dir = workspace / "configs"
    workflow_dir = config_dir / "workflows"
    workflow_dir.mkdir(parents=True)
    (config_dir / "defaults.yaml").write_text("{}\n", encoding="utf-8")

    output_dir = workspace / "runs"
    output_dir.mkdir()

    (workspace / "models").mkdir()
    (workspace / "qphase.toml").write_text(
        'schema = "qphase.project/2"\n'
        'project_id = "test-project"\n'
        'name = "Test Project"\n\n'
        "[paths]\n"
        'workflows = "configs/workflows"\n'
        'defaults = "configs/defaults.yaml"\n'
        'plugins = ["models"]\n'
        'sessions = "runs"\n',
        encoding="utf-8",
    )

    # Create a machine-policy override for this test.
    system_config_path = workspace / "system.yaml"

    # We need to mock the system config loading to use this path
    # For now, we can just set the env var, as load_system_config checks it
    os.environ["QPHASE_SYSTEM_CONFIG"] = str(system_config_path)

    # Create a basic system config file
    import yaml

    with open(system_config_path, "w") as f:
        yaml.dump(
            {
                "scan_runtime": {
                    "storage_layout": "auto",
                    "auto_shard_threshold_mib": 512,
                    "shard_target_mib": 128,
                }
            },
            f,
        )

    # Force reload of system config
    load_system_config(force_reload=True)
    previous_project = os.environ.get("QPHASE_PROJECT")
    os.environ["QPHASE_PROJECT"] = str(workspace)

    yield workspace

    # Cleanup
    os.environ.pop("QPHASE_SYSTEM_CONFIG", None)
    if previous_project is None:
        os.environ.pop("QPHASE_PROJECT", None)
    else:
        os.environ["QPHASE_PROJECT"] = previous_project
    # Reset system config cache
    load_system_config(force_reload=True)


@pytest.fixture
def temp_project(tmp_path):
    """Create an isolated Project for scheduler/service tests."""
    return ProjectContext.create(tmp_path / "project", name="Test Project")


@pytest.fixture
def mock_registry():
    """Reset registry before and after test."""
    # Save original state if needed (though registry is singleton)
    # For now, just ensure we start fresh-ish
    yield registry
    # Cleanup could involve clearing registered plugins if we added dynamic ones


@pytest.fixture
def sample_job_file(temp_workspace):
    """Create a sample job file in the temp workspace."""
    job_file = temp_workspace / "configs" / "workflows" / "test_job.yaml"
    job_file.parent.mkdir(parents=True, exist_ok=True)

    import yaml

    with open(job_file, "w") as f:
        yaml.dump(
            {
                "schema": "qphase.workflow/2",
                "id": "test_job",
                "title": "Test Job",
                "jobs": [
                    {
                        "name": "test_job",
                        "engine": {"dummy": {"param": 10.0}},
                        "model": {"dummy": {"param": 1.0}},
                        "backend": {"dummy": {"param": 1.0}},
                    }
                ],
            },
            f,
        )

    return job_file


@pytest.fixture
def dummy_model():
    """Return the dummy model class (registered by the tests/qphase conftest)."""
    from tests.plugins.dummy_plugin import DummyPlugin

    return DummyPlugin
