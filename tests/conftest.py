"""Pytest configuration and fixtures."""

import os
import sys
from pathlib import Path

import pytest

# Add package paths to sys.path
packages_dir = Path(__file__).parent.parent / "packages"
sys.path.insert(0, str(packages_dir / "qphase"))
# Note: We do NOT add qphase_sde to path to ensure core tests are independent

from qphase.core.config_loader import load_system_config  # noqa: E402
from qphase.core.registry import registry  # noqa: E402


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


@pytest.fixture(autouse=True)
def register_dummy_plugins():
    """Register dummy plugins for testing."""
    from qphase.core.registry import registry

    from tests.plugins.dummy_plugin import DummyPlugin

    # Register dummy engine
    registry.register(
        namespace="engine", name="dummy", builder=DummyPlugin, overwrite=True
    )

    # Register dummy backend
    registry.register(
        namespace="backend", name="dummy", builder=DummyPlugin, overwrite=True
    )

    # Register dummy model
    registry.register(
        namespace="model", name="dummy", builder=DummyPlugin, overwrite=True
    )

    # Register dummy viz engine
    registry.register(
        namespace="engine", name="viz", builder=DummyPlugin, overwrite=True
    )


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
    """Return the dummy model class (already registered)."""
    from tests.plugins.dummy_plugin import DummyPlugin

    return DummyPlugin
