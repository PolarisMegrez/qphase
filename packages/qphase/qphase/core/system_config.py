"""qphase: System Configuration Models
---------------------------------------------------------
Defines the Pydantic models for system-level configuration (``system.yaml``). This
includes settings for file paths (output directories, config locations), global
behavior flags (auto-save), and parameter scan defaults, serving as the root
configuration context for the framework.

Public API
----------
``SystemConfig`` : Root configuration model with paths, auto_save, and parameter_scan
``PathsConfig`` : Nested model for output_dir, global_file, plugin_dirs, config_dirs

Notes
-----
- System config controls framework behavior independent of individual jobs
- Supports multi-level override: package defaults → user config → environment

"""

import importlib.resources as ilr
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .errors import QPhaseConfigError, get_logger
from .utils import save_yaml, deep_merge_dicts, load_yaml

__all__ = ["SystemConfig", "PathsConfig"]

logger = get_logger()

class PathsConfig(BaseModel):
    """Unified path configuration for the system.

    All path-related configuration parameters are consolidated here
    with consistent naming conventions.
    """

    # Single-value paths (strings)
    output_dir: str = Field(
        default="./runs",
        description="Default output directory for simulation runs. Relative paths "
        "are resolved against CWD.",
    )

    global_file: str = Field(
        default="./configs/global.yaml",
        description="Path to the global plugin configuration file.",
    )

    # Multi-value paths (lists)
    plugin_dirs: list[str] = Field(
        default_factory=lambda: ["./plugins"],
        description="Paths to scan for plugin configuration files "
        "(.qphase_plugins.yaml).",
    )

    config_dirs: list[str] = Field(
        default_factory=lambda: ["./configs"],
        description="Directories to search for configuration files and job templates.",
    )

    @field_validator("output_dir", "global_file")
    @classmethod
    def validate_paths_not_empty(cls, v: str) -> str:
        """Validate that path fields are not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")
        return v

    @field_validator("plugin_dirs", "config_dirs")
    @classmethod
    def validate_path_lists_not_empty(cls, v: list[str]) -> list[str]:
        """Validate that path list fields are not empty and contain
        non-empty strings.
        """
        if not v:
            raise ValueError("Path list cannot be empty")
        for path in v:
            if not path or not path.strip():
                raise ValueError("Path in list cannot be empty")
        return v

    def get_output_dir(self) -> Path:
        """Get output directory as Path object, resolving relative paths."""
        return Path(self.output_dir).resolve()

    def get_global_file(self) -> Path:
        """Get global config file as Path object, resolving relative paths."""
        return Path(self.global_file).resolve()

    def get_plugin_dirs(self) -> list[Path]:
        """Get plugin directories as list of Path objects, resolving relative paths."""
        return [Path(p).resolve() for p in self.plugin_dirs]

    def get_config_dirs(self) -> list[Path]:
        """Get config directories as list of Path objects, resolving relative paths."""
        return [Path(p).resolve() for p in self.config_dirs]


class SystemConfig(BaseModel):
    """System-wide configuration parameters.

    These parameters control the global behavior of the QPhase system
    and should only be modified by experts. They are loaded from system.yaml
    and should NOT be included in per-run snapshots.

    Attributes
    ----------
    paths : PathsConfig
        Unified path configuration containing all path-related settings
    auto_save_results : bool
        Whether scheduler should automatically save job results to disk.
        If False, results are only passed to downstream jobs (if any).
        Default: True
    parameter_scan : dict
        Parameter scan configuration for batch execution.
        - enabled: Enable parameter scan expansion (default: True)
        - method: Expansion method - 'cartesian' or 'zipped' (default: 'cartesian')
        - numbered_outputs: Auto-number expanded job outputs (default: True)

    """

    paths: PathsConfig = Field(default_factory=PathsConfig)
    auto_save_results: bool = Field(
        default=True,
        description="Automatically save job results to disk. Set to False to "
        "disable automatic saving.",
    )
    parameter_scan: dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "method": "cartesian",
            "numbered_outputs": True,
        },
        description="Parameter scan configuration for batch execution",
    )

    class Config:
        """Pydantic config."""

        frozen = False
        extra = "forbid"

# Cache for system config
_SYSTEM_CONFIG_CACHE: SystemConfig | None = None

def load_system_config(
    *, force_reload: bool = False, config_path: str | Path | None = None
) -> SystemConfig:
    """Load system configuration with override chain.

    Search order (later overrides earlier):
    1. Package default (qphase.core/system.yaml)
    2. /etc/qphase/config.yaml (System-wide)
    3. ~/.qphase/config.yaml (User-specific)
    4. QPHASE_CONFIG environment variable
    5. Explicitly provided config_path

    Parameters
    ----------
    force_reload : bool
        If True, ignore cache and reload
    config_path : str or Path, optional
        Path to specific config file to override everything else

    Returns
    -------
    SystemConfig
        Loaded system configuration

    """
    global _SYSTEM_CONFIG_CACHE

    if _SYSTEM_CONFIG_CACHE is not None and not force_reload and config_path is None:
        return _SYSTEM_CONFIG_CACHE

    # 1. Load package default
    try:
        system_yaml_path = ilr.files("qphase.core").joinpath("system.yaml")
        config_dict = load_yaml(Path(str(system_yaml_path)))
    except Exception:
        logger.warning("Could not load default system.yaml from package")
        config_dict = {}

    # 2. System-wide config
    sys_path = Path("/etc/qphase/config.yaml")
    if sys_path.exists():
        try:
            sys_dict = load_yaml(sys_path)
            config_dict = deep_merge_dicts(config_dict, sys_dict)
        except Exception as e:
            logger.warning(f"Failed to load system config {sys_path}: {e}")

    # 3. User config
    user_path = Path.home() / ".qphase" / "config.yaml"
    if user_path.exists():
        try:
            user_dict = load_yaml(user_path)
            config_dict = deep_merge_dicts(config_dict, user_dict)
        except Exception as e:
            logger.warning(f"Failed to load user config {user_path}: {e}")
    else:
        # Silent Generation: Create user config from package default if missing
        try:
            user_path.parent.mkdir(parents=True, exist_ok=True)
            save_yaml(config_dict, user_path)
            logger.info(f"Created default user config at {user_path}")
        except Exception as e:
            logger.warning(f"Failed to create default user config at {user_path}: {e}")

    # 4. Environment variable
    env_path = os.environ.get("QPHASE_SYSTEM_CONFIG")
    if env_path:
        path = Path(env_path)
        if path.exists():
            try:
                env_dict = load_yaml(path)
                config_dict = deep_merge_dicts(config_dict, env_dict)
            except Exception as e:
                logger.warning(f"Failed to load env config {path}: {e}")

    # 5. Explicit path
    if config_path:
        path = Path(config_path)
        if path.exists():
            try:
                explicit_dict = load_yaml(path)
                config_dict = deep_merge_dicts(config_dict, explicit_dict)
            except Exception as e:
                raise QPhaseConfigError(
                    f"Failed to load explicit config {path}: {e}"
                ) from e

    try:
        _SYSTEM_CONFIG_CACHE = SystemConfig(**config_dict)
        return _SYSTEM_CONFIG_CACHE
    except Exception as e:
        raise QPhaseConfigError(f"Invalid system configuration: {e}") from e


def save_user_config(config: SystemConfig) -> None:
    """Save system configuration to user home directory."""
    user_config_dir = Path.home() / ".qphase"
    user_config_dir.mkdir(exist_ok=True)
    user_config_path = user_config_dir / "config.yaml"

    config_dict = config.model_dump()
    save_yaml(config_dict, user_config_path)

    global _SYSTEM_CONFIG_CACHE
    _SYSTEM_CONFIG_CACHE = None
