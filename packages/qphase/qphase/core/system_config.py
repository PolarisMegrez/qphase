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

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

__all__ = ["SystemConfig", "PathsConfig"]


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
