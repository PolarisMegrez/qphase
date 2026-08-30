from pathlib import Path

import pytest
from qphase.core.config_loader import load_project_defaults
from qphase.core.errors import QPhaseConfigError
from qphase.core.system_config import (
    SystemConfig,
    SystemConfigStore,
    load_system_config,
    save_user_config,
)
from qphase.core.utils import load_yaml


def test_missing_user_system_config_is_not_created_on_read(tmp_path, monkeypatch):
    """Reading defaults must not persist a machine-specific config snapshot."""
    # Mock home directory to tmp_path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Ensure no config exists
    config_path = tmp_path / ".qphase" / "config.yaml"
    assert not config_path.exists()

    # Load system config without an explicit persistence request.
    config = load_system_config(force_reload=True)

    assert isinstance(config, SystemConfig)
    assert not config_path.exists()


def test_system_config_store_persists_sparse_user_override(tmp_path):
    package_path = tmp_path / "defaults.yaml"
    package_path.write_text(
        "auto_save_results: true\nreporting:\n  logging:\n    console_level: WARNING\n",
        encoding="utf-8",
    )
    user_path = tmp_path / "user" / "config.yaml"
    store = SystemConfigStore(
        package_default_path=package_path,
        site_path=tmp_path / "missing-site.yaml",
        user_path=user_path,
        environ={},
    )
    config = store.load()
    config.reporting.logging.console_level = "ERROR"

    store.save_user(config)

    assert load_yaml(user_path) == {
        "reporting": {"logging": {"console_level": "ERROR"}}
    }
    assert store.load().reporting.logging.console_level == "ERROR"


def test_system_config_override_precedence(tmp_path):
    package_path = tmp_path / "defaults.yaml"
    package_path.write_text("auto_save_results: true\n", encoding="utf-8")
    site_path = tmp_path / "site.yaml"
    site_path.write_text("auto_save_results: false\n", encoding="utf-8")
    user_path = tmp_path / "user.yaml"
    user_path.write_text(
        "reporting:\n  logging:\n    console_level: INFO\n", encoding="utf-8"
    )
    env_path = tmp_path / "environment.yaml"
    env_path.write_text(
        "reporting:\n  logging:\n    console_level: WARNING\n", encoding="utf-8"
    )
    explicit_path = tmp_path / "explicit.yaml"
    explicit_path.write_text(
        "reporting:\n  logging:\n    console_level: ERROR\n", encoding="utf-8"
    )
    store = SystemConfigStore(
        package_default_path=package_path,
        site_path=site_path,
        user_path=user_path,
        environ={"QPHASE_SYSTEM_CONFIG": str(env_path)},
    )

    config = store.load(config_path=explicit_path)

    assert config.auto_save_results is False
    assert config.reporting.logging.console_level == "ERROR"


def test_system_config_rejects_removed_parameter_scan_key(tmp_path):
    """Removed keys fail fast against the closed system schema."""
    package_path = tmp_path / "defaults.yaml"
    package_path.write_text("auto_save_results: true\n", encoding="utf-8")
    override_path = tmp_path / "override.yaml"
    override_path.write_text("parameter_scan: {}\n", encoding="utf-8")
    store = SystemConfigStore(
        package_default_path=package_path,
        site_path=tmp_path / "missing-site.yaml",
        user_path=tmp_path / "missing-user.yaml",
        environ={},
    )

    with pytest.raises(QPhaseConfigError, match="Extra inputs are not permitted"):
        store.load(config_path=override_path)


def test_silent_generation_project_defaults(tmp_path):
    """Missing project defaults are initialized as an empty mapping."""
    defaults_path = tmp_path / "defaults.yaml"
    assert not defaults_path.exists()

    # Load global config (should trigger generation)
    config = load_project_defaults(defaults_path)

    assert isinstance(config, dict)
    assert defaults_path.exists()
    assert config == {}


def test_config_reset_system(tmp_path, monkeypatch):
    """Test resetting system configuration."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Create a modified user config
    user_config_dir = tmp_path / ".qphase"
    user_config_dir.mkdir()
    user_config_path = user_config_dir / "config.yaml"

    with open(user_config_path, "w") as f:
        f.write("auto_save_results: false\n")

    # Verify it's loaded
    config = load_system_config(force_reload=True)
    assert config.auto_save_results is False

    # Reset logic (simulating the command)
    import importlib.resources as ilr

    system_yaml_path = ilr.files("qphase.core").joinpath("system.yaml")
    default_config_dict = load_yaml(Path(str(system_yaml_path)))
    config_obj = SystemConfig(**default_config_dict)
    save_user_config(config_obj)

    # Verify reset
    config = load_system_config(force_reload=True)
    assert config.auto_save_results is True
