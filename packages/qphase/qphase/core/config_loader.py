"""System-policy and project-default configuration utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import get_logger
from .project import ProjectContext
from .system_config import load_system_config
from .utils import deep_copy, deep_merge_dicts, load_yaml, save_yaml

if TYPE_CHECKING:
    from .registry import RegistryCenter

logger = get_logger()


def get_system_param(path: str, default: Any = None) -> Any:
    """Get a machine-policy value by dot-separated path."""
    current: Any = load_system_config().model_dump()
    for segment in path.split("."):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        else:
            return default
    return current


def load_project_defaults(path: Path) -> dict[str, Any]:
    """Load project-scoped plugin defaults, creating an empty file if absent."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        save_yaml({}, path)
        logger.info("Created empty project defaults at %s", path)
        return {}
    payload = load_yaml(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def save_project_defaults(config: dict[str, Any], path: Path) -> None:
    """Save project-scoped plugin defaults."""
    save_yaml(config, path)


def merge_configs(
    project_defaults: dict[str, Any], job_config: dict[str, Any]
) -> dict[str, Any]:
    """Merge project defaults with a logical job override."""
    return deep_merge_dicts(deep_copy(project_defaults), job_config)


def registered_plugin_namespaces() -> set[str]:
    """Return ordinary plugin namespaces currently known to the registry."""
    from .registry import registry

    compatibility_namespaces = {
        "analyser",
        "analyzer",
        "backend",
        "integrator",
        "model",
        "observer",
        "visualizer",
        "cam_solver",
        "cam_postprocessor",
    }
    try:
        namespaces = set(registry.list(namespace=None))
    except Exception:
        namespaces = set()
    return (namespaces | compatibility_namespaces) - {
        "default",
        "engine",
        "loader",
        "resource",
    }


def merge_plugin_config_sections(config: dict[str, Any]) -> dict[str, Any]:
    """Merge top-level project plugin defaults with explicit job plugins."""
    plugins = deep_copy(config.get("plugins", {}))
    for namespace in registered_plugin_namespaces():
        inherited = config.get(namespace)
        if not isinstance(inherited, Mapping):
            continue
        override = plugins.get(namespace, {})
        plugins[namespace] = deep_merge_dicts(
            dict(inherited), dict(override) if isinstance(override, Mapping) else {}
        )
    return plugins


def get_config_for_job(
    project: ProjectContext,
    job_config_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a logical job configuration merged with project defaults."""
    defaults = load_project_defaults(project.defaults_path)
    return merge_configs(defaults, job_config_dict or {})


def construct_plugins_config(reg: RegistryCenter) -> dict[str, dict[str, Any]]:
    """Construct project defaults from discovered plugin schemas."""
    exclude_namespaces = {"resource", "loader", "default", "model"}
    plugins_config: dict[str, dict[str, Any]] = {}
    for namespace in reg.list(namespace=None):
        if namespace in exclude_namespaces:
            continue
        for plugin_name in reg.list(namespace=namespace):
            try:
                schema = reg.get_plugin_schema(namespace, plugin_name)
                if not schema:
                    continue
                from .utils import schema_to_yaml_map

                values = schema_to_yaml_map(schema, {}, plugin_name, mode="global")
                if values:
                    plugins_config.setdefault(namespace, {})[plugin_name] = values
            except Exception:
                continue
    return {key: value for key, value in plugins_config.items() if value}
