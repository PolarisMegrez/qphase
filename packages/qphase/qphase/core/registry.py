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
from datetime import datetime
from importlib import import_module
from typing import Any

from .errors import (
    QPhaseConfigError,
    QPhasePluginError,
)
from .protocols import PluginManifest
from .utils import load_yaml

Builder = Callable[..., Any]

__all__ = [
    "RegistryCenter",
    "RegistryView",
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
    resolved: Any | None = None

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

    def view(self) -> "RegistryView":
        """Return the read-only view used by control-plane compilation."""
        return RegistryView(self)

    def snapshot(self, *, include_local: bool = True) -> "RegistryCenter":
        """Copy registered asset definitions for one project runtime."""
        copied = RegistryCenter()
        copied._tables = {
            namespace: {
                name: _Entry(
                    kind=entry.kind,
                    builder=entry.builder,
                    target=entry.target,
                    config_schema=entry.config_schema,
                    meta=dict(entry.meta or {}),
                    resolved=entry.resolved,
                )
                for name, entry in table.items()
                if include_local or not (entry.meta or {}).get("local_import_root")
            }
            for namespace, table in self._tables.items()
        }
        return copied

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
        full_meta.setdefault("registered_at", datetime.now().isoformat())
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
        full_meta.setdefault("registered_at", datetime.now().isoformat())
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

    def _resolve_entry(self, entry: _Entry) -> Any:
        """Resolve an entry to a class/callable without instantiating it."""
        if entry.kind == "callable":
            assert entry.builder is not None
            return entry.builder
        assert entry.target is not None
        return self._import_target(entry.target)

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
            if entry.resolved is None:
                entry.resolved = self._resolve_entry(entry)
            obj = entry.resolved
            return obj
        except Exception as e:
            raise QPhasePluginError(
                f"Failed to import plugin '{name}' from '{entry.target}': {e}"
            ) from e

    def get_plugin_manifest(self, namespace: str, name: str) -> PluginManifest:
        """Return the manifest declared by a registered plugin."""
        plugin_class = self.get_plugin_class(namespace, name)
        manifest = getattr(plugin_class, "manifest", None)
        return manifest if isinstance(manifest, PluginManifest) else PluginManifest()

    def is_local_plugin(self, namespace: str, name: str) -> bool:
        """Return whether an entry is owned by the current Project."""
        entry = self._tables.get(namespace, {}).get(name)
        return bool(entry is not None and (entry.meta or {}).get("local_import_root"))

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

        try:
            obj = self.get_plugin_class(ns, nm)
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
        schema = self.get_plugin_schema(plugin_type, plugin_name)
        if schema:
            from .plugin_graph import PluginGraphResolver

            node = PluginGraphResolver(self).resolve(
                plugin_type,
                str(plugin_name),
                merged_kwargs,
                parent_kwargs=extra_kwargs,
            )
            return node.instance

        # Fallback for plugins without schema (should be avoided in strict mode)
        return self.create(f"{plugin_type}:{plugin_name}", **merged_kwargs)

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
            obj = self.get_plugin_class(namespace, name)

            # Check for config_schema on the class/object
            if hasattr(obj, "config_schema"):
                entry.config_schema = obj.config_schema
                return entry.config_schema
        except Exception:
            pass

        return None

    def validate_plugin_config(
        self, plugin_type: str, config_data: dict[str, Any]
    ) -> Any:
        """Validate a parent plugin and its declared child-plugin graph."""
        name = config_data.get("name")
        if not name:
            raise QPhaseConfigError(f"Plugin config for '{plugin_type}' missing 'name'")

        if not self.get_plugin_schema(plugin_type, name):
            raise QPhaseConfigError(
                f"No configuration schema found for plugin "
                f"'{plugin_type}:{name}'. All plugins must define a config_schema."
            )

        params = config_data.get("params", {}).copy()
        # Merge other fields into params if they are not name/params
        for k, v in config_data.items():
            if k not in ["name", "params"]:
                params[k] = v

        from .plugin_graph import PluginGraphResolver

        node = PluginGraphResolver(self).resolve(
            plugin_type,
            str(name),
            params,
            instantiate=False,
        )
        return node.config

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


class RegistryView:
    """Read-only registry facade for compiler and service consumers."""

    def __init__(self, source: RegistryCenter) -> None:
        self._source = source.snapshot()

    def list(self, namespace: str | None = None) -> dict[str, Any]:
        """List registered plugins without exposing mutation methods."""
        return self._source.list(namespace)

    def get_plugin_class(self, namespace: str, name: str) -> Any:
        """Resolve a plugin class through the trusted registry."""
        return self._source.get_plugin_class(namespace, name)

    def get_plugin_manifest(self, namespace: str, name: str) -> PluginManifest:
        """Read one plugin manifest."""
        return self._source.get_plugin_manifest(namespace, name)

    def is_local_plugin(self, namespace: str, name: str) -> bool:
        """Read local ownership without importing the plugin module."""
        return self._source.is_local_plugin(namespace, name)

    def get_plugin_schema(self, namespace: str, name: str) -> type[Any] | None:
        """Read one plugin configuration schema."""
        return self._source.get_plugin_schema(namespace, name)

    def validate_plugin_config(
        self, namespace: str, config: dict[str, Any]
    ) -> Any:
        """Validate plugin configuration without constructing the plugin."""
        return self._source.validate_plugin_config(namespace, config)


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

    def discover_local_plugins(self, project: Any | None = None) -> int:
        """Discover and register plugins from .qphase_plugins.yaml files.

        Scans plugin directories defined in the project manifest for local plugin
        configuration files.

        Returns
        -------
        int
            Number of plugins discovered

        """
        discovered_count = 0

        from .project import ProjectContext

        project = project or ProjectContext.discover()
        plugin_dirs = project.plugin_dirs

        for plugin_dir in plugin_dirs:
            if not plugin_dir.exists() or not plugin_dir.is_dir():
                continue

            # Look for .qphase_plugins.yaml in the directory
            plugins_file = plugin_dir / ".qphase_plugins.yaml"
            if not plugins_file.exists():
                continue

            data = load_yaml(plugins_file)
            if not isinstance(data, dict):
                raise QPhaseConfigError(
                    f"local plugin manifest must contain a mapping: {plugins_file}"
                )
            if "plugins" not in data:
                raise QPhaseConfigError(
                    f"local plugin manifest is missing 'plugins': {plugins_file}"
                )
            plugins_list = data["plugins"]
            if not isinstance(plugins_list, list):
                raise QPhaseConfigError(
                    f"local plugin manifest 'plugins' must be a list: {plugins_file}"
                )

            import_root = str(plugin_dir.parent.resolve())
            if import_root not in sys.path:
                sys.path.insert(0, import_root)

            for index, plugin_entry in enumerate(plugins_list):
                if not isinstance(plugin_entry, dict):
                    raise QPhaseConfigError(
                        f"local plugin entry {index} must be a mapping: {plugins_file}"
                    )

                plugin_type = plugin_entry.get("type", "")
                target = plugin_entry.get("target", "")

                if not isinstance(plugin_type, str) or not plugin_type.strip():
                    raise QPhaseConfigError(
                        f"local plugin entry {index} requires a string 'type': "
                        f"{plugins_file}"
                    )
                if not isinstance(target, str) or not target.strip():
                    raise QPhaseConfigError(
                        f"local plugin entry {index} requires a string 'target': "
                        f"{plugins_file}"
                    )

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

                # Collect optional batching metadata for engine entries.
                extra_meta: dict[str, Any] = {}
                if namespace == "engine":
                    for meta_key in ("batch_planner", "result_splitter"):
                        value = plugin_entry.get(meta_key)
                        if value:
                            extra_meta[meta_key] = value

                # Register the plugin
                self.registry.register_lazy(
                    namespace=namespace,
                    name=name,
                    target=target,
                    auto_discovered=True,
                    source_file=str(plugins_file),
                    local_import_root=str(plugin_dir.parent.resolve()),
                    **extra_meta,
                )
                discovered_count += 1

        return discovered_count


# Global singleton
registry = RegistryCenter()
discovery = DiscoveryService(registry)
