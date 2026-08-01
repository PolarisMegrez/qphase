"""qphase: System Configuration Models
---------------------------------------------------------
Defines the Pydantic models for system-level configuration (``system.yaml``). This
includes settings for file paths (output directories, config locations), global
behavior flags (auto-save), and parameter scan defaults, serving as the root
configuration context for the framework.

Public API
----------
SystemConfig
    Root configuration model with paths, auto-save, and scan runtime services.
PathsConfig
    Nested model for output_dir, global_file, plugin_dirs, config_dirs.
"""

import importlib.resources as ilr
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import QPhaseConfigError, get_logger
from .utils import deep_merge_dicts, load_yaml, save_yaml

__all__ = [
    "CheckpointConfig",
    "LoggingReportConfig",
    "PathsConfig",
    "ProgressReportConfig",
    "ReportingConfig",
    "ResourceHintsConfig",
    "ScanRuntimeConfig",
    "SystemConfig",
]

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


class CheckpointConfig(BaseModel):
    """Chunk-level checkpoint behavior for logical scans."""

    enabled: bool = False
    interval_chunks: int = Field(default=1, ge=1)
    keep_on_success: bool = False

    model_config = ConfigDict(extra="forbid")


class ResourceHintsConfig(BaseModel):
    """Workstation resource hints forwarded to resource engines."""

    cpu_worker_limit: int | None = Field(default=None, ge=1)
    memory_limit_mib: int | None = Field(default=None, ge=1)
    gpu_device: int | str | None = None
    gpu_memory_fraction: float | None = Field(default=None, gt=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class ScanRuntimeConfig(BaseModel):
    """Core storage and runtime services for engine-owned scans."""

    storage_layout: Literal["auto", "single", "sharded", "per_point"] = "auto"
    auto_shard_threshold_mib: int = Field(default=512, ge=1)
    shard_target_mib: int = Field(default=128, ge=1)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    resources: ResourceHintsConfig = Field(default_factory=ResourceHintsConfig)

    model_config = ConfigDict(extra="forbid")


class ProgressReportConfig(BaseModel):
    """Progress rendering and ETA estimation behavior."""

    refresh_interval: float = Field(
        default=0.5,
        ge=0.05,
        description="Minimum seconds between CLI progress line refreshes.",
    )
    non_tty_milestone_percent: float = Field(
        default=10.0,
        gt=0.0,
        le=100.0,
        description="Percent step between milestone lines on non-TTY output.",
    )
    eta_warmup_seconds: float = Field(
        default=2.0,
        ge=0.0,
        description="Minimum stage age before rate/ETA estimates are shown.",
    )
    eta_min_samples: int = Field(
        default=3,
        ge=1,
        description="Minimum number of rate samples before ETA is shown.",
    )
    eta_smoothing: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="EMA weight of the newest rate sample.",
    )

    model_config = ConfigDict(extra="forbid")


class LoggingReportConfig(BaseModel):
    """Session log file and console logging behavior."""

    session_file: bool = Field(
        default=True,
        description="Automatically write a full log file into each session dir.",
    )
    filename: str = Field(default="qphase.log", min_length=1)
    file_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"
    console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "WARNING"
    format: Literal["text", "json"] = "text"
    capture_warnings: bool = True

    model_config = ConfigDict(extra="forbid")


class ReportingConfig(BaseModel):
    """Progress, terminal output, and diagnostic logging configuration."""

    progress: ProgressReportConfig = Field(default_factory=ProgressReportConfig)
    logging: LoggingReportConfig = Field(default_factory=LoggingReportConfig)

    model_config = ConfigDict(extra="forbid")


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
    scan_runtime : ScanRuntimeConfig
        Storage, checkpoint, and workstation resource hints for logical scans.
    reporting : ReportingConfig
        Progress rendering, ETA estimation, and diagnostic logging behavior.

    """

    paths: PathsConfig = Field(default_factory=PathsConfig)
    auto_save_results: bool = Field(
        default=True,
        description="Automatically save job results to disk. Set to False to "
        "disable automatic saving.",
    )
    scan_runtime: ScanRuntimeConfig = Field(default_factory=ScanRuntimeConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    model_config = ConfigDict(frozen=False, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_keys(cls, data: Any) -> Any:
        """Map the removed ``progress_update_interval`` key onto reporting."""
        if isinstance(data, dict) and "progress_update_interval" in data:
            value = data.pop("progress_update_interval")
            logger.warning(
                "[992] DEPRECATED: system setting 'progress_update_interval' has "
                "moved to 'reporting.progress.refresh_interval'; applying the "
                "old value."
            )
            reporting = dict(data.get("reporting") or {})
            progress = dict(reporting.get("progress") or {})
            progress.setdefault("refresh_interval", value)
            reporting["progress"] = progress
            data["reporting"] = reporting
        return data

    @property
    def progress_update_interval(self) -> float:
        """Deprecated alias for ``reporting.progress.refresh_interval``."""
        return self.reporting.progress.refresh_interval


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

    if "parameter_scan" in config_dict:
        config_dict.pop("parameter_scan", None)
        logger.warning(
            "Ignoring removed system setting 'parameter_scan'; define explicit "
            "job.scan axes instead."
        )

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
