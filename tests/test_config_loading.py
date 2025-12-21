from pathlib import Path

from qphase.core.config_loader import load_global_config, load_system_config
from qphase.core.system_config import SystemConfig, save_user_config
from qphase.core.utils import load_yaml


def test_silent_generation_system_config(tmp_path, monkeypatch):
    """Test that system config is silently generated if missing."""
    # Mock home directory to tmp_path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Ensure no config exists
    config_path = tmp_path / ".qphase" / "config.yaml"
    assert not config_path.exists()

    # Load system config (should trigger generation)
    config = load_system_config(force_reload=True)

    assert isinstance(config, SystemConfig)
    assert config_path.exists()

    # Verify content
    saved_data = load_yaml(config_path)
    assert "paths" in saved_data


def test_silent_generation_global_config(tmp_path):
    """Test that global config is silently generated if missing."""
    global_path = tmp_path / "global.yaml"
    assert not global_path.exists()

    # Load global config (should trigger generation)
    config = load_global_config(global_path)

    assert isinstance(config, dict)
    assert global_path.exists()
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
