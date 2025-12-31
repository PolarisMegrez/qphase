"""qphase: Plugin Registry
---------------------------------------------------------
Implements the central registry for plugin management, supporting dynamic discovery,
registration, and factory-style instantiation. It handles both Python entry points
for installed packages and local ``.qphase_plugins.yaml`` files for development,
managing multiple namespaces (backend, integrator, engine) to keep the system
extensible.

Public API
----------
RegistryCenter
    Registry class managing plugin namespaces and entries.
registry
    Global singleton instance for application-wide plugin access.
"""

import importlib.metadata
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from typing import Any

from .errors import (
    QPhaseConfigError,
    QPhasePluginError,
)
from .system_config import load_system_config
from .utils import load_yaml

# Get UTC timezone
UTC = timezone.utc

Builder = Callable[..., Any]

__all__ = [
    "RegistryCenter",
    "DiscoveryService",
    "registry",
    "discovery",
]


@dataclass
class _Entry:
    """Internal record describing a registry entry."""

    kind: str  # "callable" | "dotted"
    builder: Builder | None = None
    target: str | None = None  # dotted path like "pkg.mod:Class"
    config_schema: type[Any] | None = None
    meta: dict[str, Any] | None = None

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}


Namespace = str
Name = str
FullName = str


class RegistryCenter:
    """Central registry for plugin types with factory-style lookup.

    Maintains per-namespace tables that map names to either callables or dotted
    import targets. Supports lazy loading via entry points and local plugin files.

    Examples
    --------
    >>> registry.register("backend", "numpy", NumpyBackend)
    >>> backend = registry.create("backend:numpy", config=config)

    """

    def __init__(self) -> None:
        self._tables: dict[Namespace, dict[Name, _Entry]] = {}

    def reset(self) -> None:
        """Reset the registry to its initial state."""
        self._tables.clear()

    # --------------------------- utilities ---------------------------
    @staticmethod
    def _split(full_name: FullName) -> tuple[Namespace, Name]:
        """Split a full key into namespace and name."""
        if ":" in full_name:
            ns, nm = full_name.split(":", 1)
            return ns.strip().lower(), nm.strip().lower()
        return "default", full_name.strip().lower()

    def _ensure_ns(self, namespace: Namespace) -> dict[Name, _Entry]:
        """Ensure a namespace table exists and return it."""
        ns = namespace.strip().lower()
        return self._tables.setdefault(ns, {})

    # --------------------------- registration ---------------------------
    def register(
        self,
        namespace: Namespace,
        name: Name,
        builder: Builder,
        *,
        overwrite: bool = False,
        **meta: Any,
    ) -> None:
        """Register a callable builder immediately under a namespace.

        Parameters
        ----------
        namespace : str
            Plugin namespace (e.g., "backend", "integrator")
        name : str
            Plugin name within the namespace
        builder : Callable
            Factory callable or class to instantiate the plugin
        overwrite : bool, optional
            If True, overwrite existing registration
        **meta : Any
            Additional metadata to store with the registration

        Raises
        ------
        ValueError
            If name already registered and overwrite is False

        """
        ns = namespace.strip().lower()
        nm = name.strip().lower()
        table = self._ensure_ns(ns)
        if not overwrite and nm in table:
            raise ValueError(f"Duplicate registration: {ns}:{nm}")

        full_meta = dict(meta or {})
        full_meta.setdefault("registered_at", datetime.now(UTC).isoformat())
        full_meta.setdefault("builder_type", self._infer_builder_type(builder))
        full_meta.setdefault("delayed_import", False)

        # Extract config schema if available on the builder
        config_schema = getattr(builder, "config_schema", None)

        table[nm] = _Entry(
            kind="callable",
            builder=builder,
            target=None,
            config_schema=config_schema,
            meta=full_meta,
        )

    def register_lazy(
        self,
        namespace: Namespace,
        name: Name,
        target: str,
        *,
        overwrite: bool = False,
        **meta: Any,
    ) -> None:
        """Register by dotted path without importing until ``create()``.

        Parameters
        ----------
        namespace : str
            Plugin namespace (e.g., "backend", "integrator")
        name : str
            Plugin name within the namespace
        target : str
            Dotted import path (e.g., "pkg.mod:ClassName")
        overwrite : bool, optional
            If True, overwrite existing registration
        **meta : Any
            Additional metadata to store with the registration

        """
        ns = namespace.strip().lower()
        nm = name.strip().lower()
        table = self._ensure_ns(ns)
        if not overwrite and nm in table:
            return

        full_meta = dict(meta or {})
        full_meta.setdefault("registered_at", datetime.now(UTC).isoformat())
        full_meta.setdefault("builder_type", "dotted")
        full_meta.setdefault("delayed_import", True)
        full_meta.setdefault("module_path", target)
        table[nm] = _Entry(
            kind="dotted",
            builder=None,
            target=str(target),
            config_schema=None,
            meta=full_meta,
        )

    def get_plugin_class(self, namespace: str, name: str) -> Any:
        """Retrieve the plugin class (or callable) without instantiation."""
        table = self._tables.get(namespace, {})
        entry = table.get(name)
        if entry is None:
            raise QPhasePluginError(
                f"Plugin '{name}' not found in namespace '{namespace}'"
            )

        if entry.kind == "callable":
            assert entry.builder is not None
            return entry.builder

        # dotted path import
        assert entry.target is not None
        try:
            obj = self._import_target(entry.target)
            return obj
        except Exception as e:
            raise QPhasePluginError(
                f"Failed to import plugin '{name}' from '{entry.target}': {e}"
            ) from e

    # --------------------------- factory ---------------------------
    def create(self, full_name: FullName, /, **kwargs: Any) -> Any:
        """Resolve and construct a plugin instance.

        Parameters
        ----------
        full_name : str
            Plugin identifier in "namespace:name" format
        **kwargs : Any
            Arguments passed to the plugin constructor

        Returns
        -------
        Any
            Instantiated plugin object

        Raises
        ------
        QPhasePluginError
            If plugin not found or import fails

        """
        ns, nm = self._split(full_name)
        table = self._tables.get(ns, {})
        entry = table.get(nm)
        if entry is None:
            raise QPhasePluginError(f"Plugin '{nm}' not found in namespace '{ns}'")

        if entry.kind == "callable":
            assert entry.builder is not None
            meta = entry.meta or {}
            if meta.get("return_callable"):
                return entry.builder
            return entry.builder(**kwargs)

        # dotted path import
        assert entry.target is not None
        try:
            obj = self._import_target(entry.target)
        except Exception as e:
            raise QPhasePluginError(
                f"Failed to import plugin '{nm}' from '{entry.target}': {e}"
            ) from e

        # Cache the schema if we just loaded the object
        if entry.config_schema is None and hasattr(obj, "config_schema"):
            entry.config_schema = obj.config_schema

        meta = entry.meta or {}
        if meta.get("return_callable"):
            return obj
        return obj(**kwargs) if callable(obj) else obj

    def _import_target(self, target: str) -> Any:
        """Import a dotted target supporting ``module:attr`` or ``module.attr``."""
        module_name: str
        attr_name: str | None = None
        if ":" in target:
            module_name, attr_name = target.split(":", 1)
        else:
            if "." in target:
                parts = target.rsplit(".", 1)
                module_name = parts[0]
                attr_name = parts[1]
            else:
                module_name = target
                attr_name = None

        try:
            mod = import_module(module_name)
        except ImportError as e:
            raise QPhasePluginError(
                f"Could not import module '{module_name}': {e}"
            ) from e

        if attr_name is None:
            return mod
        if not hasattr(mod, attr_name):
            raise QPhaseConfigError(
                f"Target '{target}' not found in module '{module_name}'"
            )
        return getattr(mod, attr_name)

    # --------------------------- plugin factory ---------------------------
    def create_plugin_instance(
        self, plugin_type: str, config: Any, **extra_kwargs: Any
    ) -> Any:
        """Create a plugin instance from a PluginConfig.

        Supports both dict and object configs.
        """
        # Extract plugin name and params from config
        # Try both attribute access (objects) and key access (dicts)
        if isinstance(config, dict):
            plugin_name = config.get("name")
            if plugin_name is None:
                raise QPhaseConfigError("PluginConfig must have a 'name' key")

            # Get params dict from dict
            params = config.get("params", {})
            # Include other config fields as params (excluding name)
            for k, v in config.items():
                if k not in ["name", "params"]:
                    params[k] = v
        else:
            # Object with attributes
            plugin_name = getattr(config, "name", None)
            if plugin_name is None:
                raise QPhaseConfigError("PluginConfig must have a 'name' attribute")

            # Get params dict
            if hasattr(config, "params"):
                params = getattr(config, "params", {})
            elif hasattr(config, "model_dump"):
                dump = config.model_dump(exclude={"name"})
                params = dump.get("params", dump)
            else:
                params = {}

        merged_kwargs = {**(params or {}), **extra_kwargs}
        full_name = f"{plugin_type}:{plugin_name}"

        schema = self.get_plugin_schema(plugin_type, plugin_name)
        if schema:
            # Validate/Create config object
            try:
                # Use model_validate for Pydantic v2, or direct instantiation
                if hasattr(schema, "model_validate"):
                    config_obj = schema.model_validate(merged_kwargs)
                else:
                    config_obj = schema(**merged_kwargs)

                return self.create(full_name, config=config_obj, **extra_kwargs)
            except Exception as e:
                raise QPhaseConfigError(
                    f"Invalid configuration for plugin "
                    f"'{plugin_type}:{plugin_name}': {e}"
                ) from e

        # Fallback for plugins without schema (should be avoided in strict mode)
        return self.create(full_name, **merged_kwargs)

    def get_plugin_schema(self, namespace: str, name: str) -> type[Any] | None:
        """Get the configuration schema class for a specific plugin."""
        table = self._tables.get(namespace)
        if not table or name not in table:
            return None

        entry = table[name]

        if entry.config_schema is not None:
            return entry.config_schema

        # Load plugin to inspect
        try:
            # Import the target class without instantiating
            assert entry.target is not None
            obj = self._import_target(entry.target)

            # Check for config_schema on the class/object
            if hasattr(obj, "config_schema"):
                entry.config_schema = obj.config_schema
                return entry.config_schema
        except Exception:
            pass

        return None

    def get_scanable_params(self, namespace: str, name: str) -> dict[str, bool]:
        """Get scanable parameters from plugin schema.

        Parameters
        ----------
        namespace : str
            Plugin namespace (e.g., 'model', 'backend', 'integrator')
        name : str
            Plugin name

        Returns
        -------
        dict[str, bool]
            Dictionary mapping parameter names to scanable status.
            Example: {'omega_a': True, 'omega_b': False, 'D': True}

        """
        schema = self.get_plugin_schema(namespace, name)
        if not schema:
            return {}

        scanable_params = {}

        try:
            # For Pydantic models, inspect the field metadata
            if hasattr(schema, "model_fields"):
                for field_name, field_info in schema.model_fields.items():
                    # Check if field has 'scanable' in metadata
                    is_scanable = False
                    if hasattr(field_info, "json_schema_extra"):
                        # Check json_schema_extra for scanable flag
                        extra = field_info.json_schema_extra
                        if callable(extra):
                            # If it's a function, call it to get the extra info
                            extra = extra()
                        if isinstance(extra, dict) and extra.get("scanable", False):
                            is_scanable = True

                    # Also check Field metadata for scanable
                    if hasattr(field_info, "field_info"):
                        field_info_obj = field_info.field_info
                        if hasattr(field_info_obj, "metadata"):
                            for meta in field_info_obj.metadata:
                                if hasattr(meta, "scanable") and meta.scanable:
                                    is_scanable = True

                    scanable_params[field_name] = is_scanable
        except Exception as e:
            print(f"DEBUG: get_scanable_params failed for {namespace}:{name}: {e}")
            # If we can't inspect the schema, return empty dict
            # The scheduler will fall back to heuristic detection
            pass

        return scanable_params

    def validate_plugin_config(
        self, plugin_type: str, config_data: dict[str, Any]
    ) -> Any:
        """Validate raw plugin configuration against its schema."""
        name = config_data.get("name")
        if not name:
            raise QPhaseConfigError(f"Plugin config for '{plugin_type}' missing 'name'")

        schema = self.get_plugin_schema(plugin_type, name)
        if not schema:
            raise QPhaseConfigError(
                f"No configuration schema found for plugin "
                f"'{plugin_type}:{name}'. All plugins must define a config_schema."
            )

        params = config_data.get("params", {}).copy()
        # Merge other fields into params if they are not name/params
        for k, v in config_data.items():
            if k not in ["name", "params"]:
                params[k] = v

        # Handle scanable parameters: use first value for validation if list provided
        scanable_params = self.get_scanable_params(plugin_type, name)
        for param_name, is_scanable in scanable_params.items():
            if is_scanable and param_name in params:
                value = params[param_name]
                if isinstance(value, list) and len(value) > 0:
                    # Use first value for validation
                    params[param_name] = value[0]

        try:
            if hasattr(schema, "model_validate"):
                return schema.model_validate(params)
            return schema(**params)
        except Exception as e:
            raise QPhaseConfigError(
                f"Invalid configuration for '{plugin_type}:{name}': {e}"
            ) from e

    # --------------------------- introspection ---------------------------
    def list(self, namespace: Namespace | None = None) -> dict[str, Any]:
        """List available entries with metadata."""
        if namespace is None:
            return {ns: sorted(list(tbl.keys())) for ns, tbl in self._tables.items()}
        ns = namespace.strip().lower()
        table = self._tables.get(ns, {})
        return {
            name: {
                "kind": ("callable" if e.kind == "callable" else "dotted"),
                **(e.meta or {}),
            }
            for name, e in table.items()
        }

    @staticmethod
    def _infer_builder_type(obj: Any) -> str:
        try:
            if callable(obj):
                return "class" if hasattr(obj, "__mro__") else "function"
        except Exception:
            pass
        return type(obj).__name__.lower()


class DiscoveryService:
    """Service for discovering plugins from entry points and local files."""

    def __init__(self, registry_center: RegistryCenter):
        self.registry = registry_center
        self._discovered_entry_points: set[str] = set()

    def reset(self) -> None:
        """Reset discovery state."""
        self._discovered_entry_points.clear()

    def discover_plugins(self, group: str = "qphase") -> None:
        """Automatically discover and register plugins from entry points.

        Expects entry points in the group 'qphase' with names in the format
        'category.name'.
        """
        eps = importlib.metadata.entry_points(group=group)

        for ep in eps:
            if ep.name in self._discovered_entry_points:
                continue

            self._discovered_entry_points.add(ep.name)

            # Parse entry point name
            # Format: "category.name"
            name_parts = ep.name.split(".")

            if len(name_parts) < 2:
                # Invalid format, skip
                continue

            namespace = name_parts[0]
            name = ".".join(name_parts[1:])

            # Extract package information from entry point
            package_name = None
            package_version = None
            try:
                # Get the distribution/package name from the entry point
                # Entry points are associated with a distribution
                dist = ep.dist
                if dist:
                    package_name = dist.metadata["name"]
                    package_version = dist.metadata["version"]
            except Exception:
                # If we can't get package info, just continue
                pass

            self.registry.register_lazy(
                namespace=namespace,
                name=name,
                target=ep.value,
                auto_discovered=True,
                package_name=package_name,
                package_version=package_version,
            )

    def discover_local_plugins(self) -> int:
        """Discover and register plugins from .qphase_plugins.yaml files.

        Scans plugin directories defined in system config for local plugin
        configuration files.

        Returns
        -------
        int
            Number of plugins discovered

        """
        discovered_count = 0

        try:
            system_config = load_system_config()
            plugin_dirs = system_config.paths.get_plugin_dirs()
        except Exception:
            # If system config fails to load, skip local discovery
            return 0

        for plugin_dir in plugin_dirs:
            if not plugin_dir.exists() or not plugin_dir.is_dir():
                continue

            # Look for .qphase_plugins.yaml in the directory
            plugins_file = plugin_dir / ".qphase_plugins.yaml"
            if not plugins_file.exists():
                continue

            try:
                data = load_yaml(plugins_file)
            except Exception:
                continue

            if not data or not isinstance(data, dict):
                continue

            plugins_list = data.get("plugins", [])
            if not isinstance(plugins_list, list):
                continue

            # Add plugin_dir to sys.path for imports
            plugin_dir_str = str(plugin_dir.parent)  # Parent dir for module imports
            if plugin_dir_str not in sys.path:
                sys.path.insert(0, plugin_dir_str)

            for plugin_entry in plugins_list:
                if not isinstance(plugin_entry, dict):
                    continue

                plugin_type = plugin_entry.get("type", "")
                target = plugin_entry.get("target", "")

                if not plugin_type or not target:
                    continue

                # Parse type: "namespace.name" format
                type_parts = plugin_type.split(".", 1)
                if len(type_parts) == 2:
                    namespace = type_parts[0]
                    name = type_parts[1]
                else:
                    namespace = "default"
                    name = type_parts[0]

                namespace = namespace.strip().lower()
                name = name.strip().lower()

                # Register the plugin
                self.registry.register_lazy(
                    namespace=namespace,
                    name=name,
                    target=target,
                    auto_discovered=True,
                    source_file=str(plugins_file),
                )
                discovered_count += 1

        return discovered_count


# Global singleton
registry = RegistryCenter()
discovery = DiscoveryService(registry)
