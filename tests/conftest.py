"""Pytest configuration and shared fixtures.

Packages under ``packages/`` are installed editable via the uv workspace, and the
repository root is placed on ``sys.path`` automatically by pytest (``tests/`` is a
package), so no manual ``sys.path`` manipulation is needed here.
"""

import os
from pathlib import Path

import pytest
from qphase.core.config_loader import load_system_config
from qphase.core.registry import registry

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
    yield


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace with config and output directories."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    config_dir = workspace / "configs"
    config_dir.mkdir()

    output_dir = workspace / "runs"
    output_dir.mkdir()

    # Create a dummy system config pointing to this workspace
    system_config_path = workspace / "system.yaml"

    # We need to mock the system config loading to use this path
    # For now, we can just set the env var, as load_system_config checks it
    os.environ["QPHASE_SYSTEM_CONFIG"] = str(system_config_path)

    # Create a basic system config file
    import yaml

    with open(system_config_path, "w") as f:
        yaml.dump(
            {
                "paths": {
                    "config_dirs": [str(config_dir)],
                    "output_dir": str(output_dir),
                    "global_file": str(config_dir / "global.yaml"),
                    "plugin_dirs": [str(workspace / "plugins")],
                },
                "parameter_scan": {"enabled": True, "method": "cartesian"},
            },
            f,
        )

    # Force reload of system config
    load_system_config(force_reload=True)

    yield workspace

    # Cleanup
    os.environ.pop("QPHASE_SYSTEM_CONFIG", None)
    # Reset system config cache
    load_system_config(force_reload=True)


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
    job_file = temp_workspace / "configs" / "jobs" / "test_job.yaml"
    job_file.parent.mkdir(parents=True, exist_ok=True)

    import yaml

    with open(job_file, "w") as f:
        yaml.dump(
            {
                "name": "test_job",
                "engine": {"dummy": {"param": 10.0}},
                "model": {"dummy": {"param": 1.0}},
                "backend": {"dummy": {"param": 1.0}},
            },
            f,
        )

    return job_file


@pytest.fixture
def dummy_model():
    """Return the dummy model class (registered by the tests/qphase conftest)."""
    from tests.plugins.dummy_plugin import DummyPlugin

    return DummyPlugin
