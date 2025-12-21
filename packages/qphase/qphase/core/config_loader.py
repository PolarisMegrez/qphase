"""Configuration loading utilities.

This module handles loading and parsing of both Job configurations and System
configurations from YAML files. It serves as the unified entry point for
configuration I/O, replacing the legacy `loader.py` and `system_loader.py`.
"""

from __future__ import annotations

import importlib.resources as ilr
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .config import JobConfig, JobList
from .errors import QPhaseConfigError, QPhaseIOError, get_logger
from .system_config import SystemConfig
from .utils import deep_copy, deep_merge_dicts, load_yaml_file

if TYPE_CHECKING:
    from .registry import RegistryCenter

logger = get_logger()

# Cache for system config
_SYSTEM_CONFIG_CACHE: SystemConfig | None = None


# =============================================================================
# System Configuration
# =============================================================================


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
        config_dict = load_yaml_file(Path(str(system_yaml_path)))
    except Exception:
        logger.warning("Could not load default system.yaml from package")
        config_dict = {}

    # 2. System-wide config
    sys_path = Path("/etc/qphase/config.yaml")
    if sys_path.exists():
        try:
            sys_dict = load_yaml_file(sys_path)
            config_dict = deep_merge_dicts(config_dict, sys_dict)
        except Exception as e:
            logger.warning(f"Failed to load system config {sys_path}: {e}")

    # 3. User config
    user_path = Path.home() / ".qphase" / "config.yaml"
    if user_path.exists():
        try:
            user_dict = load_yaml_file(user_path)
            config_dict = deep_merge_dicts(config_dict, user_dict)
        except Exception as e:
            logger.warning(f"Failed to load user config {user_path}: {e}")

    # 4. Environment variable
    env_path = os.environ.get("QPHASE_SYSTEM_CONFIG")
    if env_path:
        path = Path(env_path)
        if path.exists():
            try:
                env_dict = load_yaml_file(path)
                config_dict = deep_merge_dicts(config_dict, env_dict)
            except Exception as e:
                logger.warning(f"Failed to load env config {path}: {e}")

    # 5. Explicit path
    if config_path:
        path = Path(config_path)
        if path.exists():
            try:
                explicit_dict = load_yaml_file(path)
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
    _save_yaml(config_dict, user_config_path)

    global _SYSTEM_CONFIG_CACHE
    _SYSTEM_CONFIG_CACHE = None


def get_system_param(path: str, default: Any = None) -> Any:
    """Get a specific system parameter by dot-separated path."""
    config = load_system_config()
    current: Any = config.model_dump()

    for segment in path.split("."):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        else:
            return default

    return current


# =============================================================================
# Job Configuration
# =============================================================================


def load_jobs_from_files(file_paths: list[Path]) -> JobList:
    """Load job configurations from multiple files.

    Parameters
    ----------
    file_paths : list[Path]
        List of paths to YAML job configuration files

    Returns
    -------
    JobList
        Container with all loaded jobs

    """
    jobs: list[JobConfig] = []

    for path in file_paths:
        if not path.exists():
            raise QPhaseIOError(f"Job file not found: {path}")

        try:
            logger.info(f"Loading job file: {path}")
            loaded = _load_single_job_file(path)
            if isinstance(loaded, list):
                jobs.extend(loaded)
            else:
                jobs.append(loaded)
        except Exception as e:
            raise QPhaseConfigError(f"Error loading {path}: {e}") from e

    if not jobs:
        raise QPhaseConfigError("No valid jobs found in provided files")

    return JobList(jobs=jobs)


def _load_single_job_file(path: Path) -> JobConfig | list[JobConfig]:
    """Load a single job file (can contain one job or a list)."""
    data = load_yaml_file(path)

    # Case 1: List of jobs
    if isinstance(data, list):
        return [JobConfig(**cast(dict[str, Any], job_data)) for job_data in data]

    # Case 2: Single job
    if isinstance(data, dict):
        # Check if it's a "job list" wrapper
        if "jobs" in data and isinstance(data["jobs"], list):
            return [
                JobConfig(**cast(dict[str, Any], job_data))
                for job_data in data["jobs"]
            ]

        # If name is missing, use filename as job name
        if "name" not in data:
            data["name"] = path.stem

        # Handle plugin fields extraction
        job_data, plugin_data = _extract_plugin_fields(data)

        # If plugins were already in data, merge them
        if "plugins" in job_data:
            plugin_data = deep_merge_dicts(job_data["plugins"], plugin_data)

        return JobConfig(**job_data, plugins=plugin_data)

    raise QPhaseConfigError(f"Invalid job file format in {path}")


def _extract_plugin_fields(
    config_data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract plugin fields from job configuration.

    Separates core job fields (name, package, input, output, etc.)
    from plugin fields (backend, integrator, model, etc.)
    """
    from .registry import registry

    # Core job fields that should not be treated as plugins
    core_fields = {
        "name",
        "package",
        "input",
        "input_loader",
        "output",
        "system",
        "engine",
        "params",
        "tags",
        "depends_on",
        "plugins",  # Explicit plugins field
    }

    job_data = {}
    plugin_data = {}

    # Get all registered plugin namespaces
    # We use a try-except block because registry might not be fully initialized
    try:
        all_namespaces = registry.list(namespace=None)
    except Exception:
        all_namespaces = {}

    for key, value in config_data.items():
        if key in core_fields:
            # Core job field
            job_data[key] = value
        elif key in all_namespaces and isinstance(value, dict):
            # This is a plugin namespace with plugin configs
            # The format is: plugin_type -> {plugin_name -> config}
            plugin_data[key] = value
        else:
            # Unknown field - keep in job_data, Pydantic will handle extra fields
            job_data[key] = value

    return job_data, plugin_data


def list_available_jobs(system_config: SystemConfig) -> list[str]:
    """List all available jobs in the configuration paths."""
    jobs = []
    for config_dir in system_config.paths.config_dirs:
        config_path = Path(config_dir)
        jobs_dir = config_path / "jobs"

        if jobs_dir.exists():
            for ext in [".yaml", ".yml"]:
                for job_file in jobs_dir.glob(f"*{ext}"):
                    job_name = job_file.stem
                    if job_name not in jobs:
                        jobs.append(job_name)
    return sorted(jobs)


# =============================================================================
# Global & Plugin Configuration
# =============================================================================


def load_global_config(global_config_path: Path) -> dict[str, Any]:
    """Load the global plugin configuration from YAML file."""
    if not global_config_path.exists():
        return {}
    try:
        return load_yaml_file(global_config_path)
    except (QPhaseIOError, QPhaseConfigError):
        return {}


def save_global_config(config: dict[str, Any], path: Path) -> None:
    """Save global configuration to YAML file."""
    _save_yaml(config, path)


def merge_configs(
    global_config: dict[str, Any],
    job_config: dict[str, Any],
) -> dict[str, Any]:
    """Merge global and job-specific configurations."""
    merged = deep_copy(global_config)
    merged = deep_merge_dicts(merged, job_config)
    return merged


def get_config_for_job(
    system_config: SystemConfig,
    job_name: str | None = None,
    job_config_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get the complete configuration for a job, including global defaults."""
    global_config_path = Path(system_config.paths.global_file)
    global_config = load_global_config(global_config_path)

    if job_config_dict is not None:
        job_config = job_config_dict
    elif job_name is not None:
        job_config_path = _find_job_config(system_config.paths.config_dirs, job_name)
        if job_config_path is None or not job_config_path.exists():
            raise QPhaseConfigError(
                f"Job configuration not found for '{job_name}'. "
                f"Searched in: {system_config.paths.config_dirs}"
            )
        job_config = load_yaml_file(job_config_path)
    else:
        job_config = {}

    return merge_configs(global_config, job_config)


def construct_plugins_config(reg: RegistryCenter) -> dict[str, dict[str, Any]]:
    """Construct plugins section from discovered plugins."""
    # Plugin namespaces to exclude from global config
    exclude_namespaces = {"engine", "resource", "loader", "default"}

    plugins_config: dict[str, dict[str, Any]] = {}
    all_namespaces = reg.list(namespace=None)

    for ns_name in all_namespaces:
        if ns_name in exclude_namespaces:
            continue

        ns_plugins = reg.list(namespace=ns_name)
        if not ns_plugins:
            continue

        for plugin_name in ns_plugins:
            try:
                reg.get_plugin_schema(ns_name, plugin_name)
                # This would require extract_defaults_from_schema from utils
                # For now, we skip implementation details as this is rarely used
                pass
            except Exception:
                continue

    return plugins_config


# =============================================================================
# Helpers
# =============================================================================


def _find_job_config(config_paths: list[str], job_name: str) -> Path | None:
    """Find a job configuration file in the given paths."""
    for config_dir in config_paths:
        config_path = Path(config_dir)
        job_dir = config_path / "jobs"

        for ext in [".yaml", ".yml"]:
            candidate = job_dir / f"{job_name}{ext}"
            if candidate.exists():
                return candidate
    return None


def _save_yaml(data: dict[str, Any], path: Path) -> None:
    """Save YAML using available library."""
    try:
        from ruamel.yaml import YAML

        y = YAML()
        with open(path, "w", encoding="utf-8") as f:
            y.dump(data, f)
    except ImportError:
        import yaml  # type: ignore[import-untyped]

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)
    except Exception as e:
        raise QPhaseIOError(f"Failed to save config to {path}: {e}") from e
