"""qphase: System Configuration Models
---------------------------------------------------------
Defines the Pydantic models for system-level configuration (``system.yaml``). This
includes machine policy, reporting behavior, and parameter scan defaults. Project
paths are intentionally owned by ``qphase.toml`` instead.

Public API
----------
SystemConfig
    Root configuration model with paths, auto-save, and scan runtime services.
"""

import importlib.resources as ilr
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import QPhaseConfigError, get_logger
from .utils import deep_merge_dicts, load_yaml, save_yaml

__all__ = [
    "CheckpointConfig",
    "LoggingReportConfig",
    "ProgressReportConfig",
    "ReportingConfig",
    "ResourceHintsConfig",
    "ScanRuntimeConfig",
    "SystemConfig",
    "SystemConfigStore",
    "load_system_config",
    "reset_user_config",
    "save_user_config",
]

logger = get_logger()


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
    auto_save_results : bool
        Whether scheduler should automatically save job results to disk.
        If False, results are only passed to downstream jobs (if any).
        Default: True
    scan_runtime : ScanRuntimeConfig
        Storage, checkpoint, and workstation resource hints for logical scans.
    reporting : ReportingConfig
        Progress rendering, ETA estimation, and diagnostic logging behavior.

    """

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


class SystemConfigStore:
    """Load and persist core system policy through one explicit boundary."""

    def __init__(
        self,
        *,
        package_default_path: str | Path | None = None,
        site_path: str | Path | None = None,
        user_path: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.package_default_path = (
            None if package_default_path is None else Path(package_default_path)
        )
        self.site_path = None if site_path is None else Path(site_path)
        self._explicit_site_path = site_path is not None
        self.user_path = None if user_path is None else Path(user_path)
        self.environ = os.environ if environ is None else environ
        self._cache: SystemConfig | None = None
        self._cache_sources: tuple[str, ...] | None = None

    def default_config(self) -> SystemConfig:
        """Return package defaults combined with an optional site policy."""
        payload = self._load_package_defaults()
        site_path = self._resolved_site_path()
        if site_path is not None and site_path.exists():
            payload = deep_merge_dicts(payload, self._load_mapping(site_path))
        return self._validate(payload)

    def load(
        self,
        *,
        force_reload: bool = False,
        config_path: str | Path | None = None,
    ) -> SystemConfig:
        """Resolve defaults, sparse user policy, environment, and explicit input."""
        sources = self._source_signature(config_path)
        if (
            self._cache is not None
            and not force_reload
            and config_path is None
            and sources == self._cache_sources
        ):
            return self._cache

        payload = self.default_config().model_dump()
        for path in self._override_paths(config_path):
            if path.exists():
                payload = deep_merge_dicts(payload, self._load_mapping(path))
            elif path == (Path(config_path) if config_path is not None else None):
                raise QPhaseConfigError(f"System config file not found: {path}")
        payload.pop("parameter_scan", None)
        result = self._validate(payload)
        if config_path is None:
            self._cache = result
            self._cache_sources = sources
        return result

    def save_user(self, config: SystemConfig) -> Path:
        """Persist only values differing from package and site defaults."""
        path = self._resolved_user_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        sparse = _sparse_difference(
            config.model_dump(), self.default_config().model_dump()
        )
        if sparse is _UNCHANGED:
            sparse = {}
        save_yaml(sparse, path)
        self.clear_cache()
        return path

    def reset_user(self) -> None:
        """Remove the user override; defaults remain inside the package."""
        path = self._resolved_user_path()
        if path.exists():
            path.unlink()
        self.clear_cache()

    def clear_cache(self) -> None:
        self._cache = None
        self._cache_sources = None

    def _load_package_defaults(self) -> dict[str, Any]:
        try:
            path = self.package_default_path
            if path is None:
                resource = ilr.files("qphase.core").joinpath("system.yaml")
                path = Path(str(resource))
            return self._load_mapping(path)
        except QPhaseConfigError:
            raise
        except Exception as exc:
            raise QPhaseConfigError(
                f"Could not load packaged system defaults: {exc}"
            ) from exc

    def _load_mapping(self, path: Path) -> dict[str, Any]:
        try:
            payload = load_yaml(path)
        except Exception as exc:
            raise QPhaseConfigError(
                f"Failed to load system config {path}: {exc}"
            ) from exc
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise QPhaseConfigError(f"System config {path} must contain a YAML mapping")
        return payload

    def _validate(self, payload: dict[str, Any]) -> SystemConfig:
        try:
            return SystemConfig.model_validate(payload)
        except Exception as exc:
            raise QPhaseConfigError(f"Invalid system configuration: {exc}") from exc

    def _resolved_user_path(self) -> Path:
        return self.user_path or Path.home() / ".qphase" / "config.yaml"

    def _resolved_site_path(self) -> Path | None:
        if self._explicit_site_path:
            return self.site_path
        if sys.platform == "win32":
            program_data = self.environ.get("PROGRAMDATA")
            return (
                Path(program_data) / "qphase" / "config.yaml" if program_data else None
            )
        return Path("/etc/qphase/config.yaml")

    def _override_paths(self, config_path: str | Path | None) -> list[Path]:
        paths = [self._resolved_user_path()]
        env_path = self.environ.get("QPHASE_SYSTEM_CONFIG")
        if env_path:
            paths.append(Path(env_path))
        if config_path is not None:
            paths.append(Path(config_path))
        return paths

    def _source_signature(self, config_path: str | Path | None) -> tuple[str, ...]:
        paths = [self._resolved_site_path(), *self._override_paths(config_path)]
        return tuple("" if path is None else str(path.resolve()) for path in paths)


def _sparse_difference(value: Any, baseline: Any) -> Any:
    """Return a recursively sparse mapping relative to ``baseline``."""
    if isinstance(value, dict) and isinstance(baseline, dict):
        result = {
            key: difference
            for key, item in value.items()
            if (difference := _sparse_difference(item, baseline.get(key)))
            is not _UNCHANGED
        }
        return result if result else _UNCHANGED
    return _UNCHANGED if value == baseline else value


_UNCHANGED = object()
_SYSTEM_CONFIG_STORE = SystemConfigStore()


def load_system_config(
    *, force_reload: bool = False, config_path: str | Path | None = None
) -> SystemConfig:
    """Load system configuration with override chain.

    Search order (later overrides earlier):
    1. Package default (qphase.core/system.yaml)
    2. /etc/qphase/config.yaml (System-wide)
    3. ~/.qphase/config.yaml (User-specific)
    4. QPHASE_SYSTEM_CONFIG environment variable
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
    return _SYSTEM_CONFIG_STORE.load(force_reload=force_reload, config_path=config_path)


def save_user_config(config: SystemConfig) -> None:
    """Save a sparse system configuration override in the user profile."""
    _SYSTEM_CONFIG_STORE.save_user(config)


def reset_user_config() -> None:
    """Remove the persisted user override and clear the config cache."""
    _SYSTEM_CONFIG_STORE.reset_user()
