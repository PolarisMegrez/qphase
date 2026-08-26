"""Registry service facade."""

from __future__ import annotations

from typing import Any

from qphase.core.project import ProjectContext
from qphase.core.registry import DiscoveryService, RegistryCenter, registry
from qphase.core.system_config import SystemConfig

from .models import (
    PluginCatalog,
    PluginSummary,
    PluginTreeNode,
    SubpluginOptionSummary,
    SubpluginSlotSummary,
)


class RegistryService:
    """Structured API over the core plugin registry."""

    def __init__(
        self,
        registry_center: RegistryCenter | None = None,
        system_config: SystemConfig | None = None,
        project: ProjectContext | None = None,
    ):
        self.registry = registry_center or registry
        self.project = project or ProjectContext.discover()
        self.system_config = system_config
        self.discovery = DiscoveryService(self.registry)

    def discover(self, include_local: bool = True) -> PluginCatalog:
        self.discovery.discover_plugins()
        if include_local:
            self.discovery.discover_local_plugins(self.project)
        return self.get_catalog()

    def list_plugins(self, namespace: str | None = None) -> list[PluginSummary]:
        if namespace is not None:
            return [
                self._summary(namespace, name, meta)
                for name, meta in self.registry.list(namespace).items()
            ]

        summaries: list[PluginSummary] = []
        for ns_name, plugin_names in self.registry.list(namespace=None).items():
            for plugin_name in plugin_names:
                meta = self.registry.list(ns_name).get(plugin_name, {})
                summaries.append(self._summary(ns_name, plugin_name, meta))
        return summaries

    def get_catalog(self) -> PluginCatalog:
        plugins = self.list_plugins()
        packages = sorted(
            {plugin.package for plugin in plugins if plugin.package is not None}
        )
        namespaces = sorted({plugin.namespace for plugin in plugins})
        return PluginCatalog(packages=packages, namespaces=namespaces, plugins=plugins)

    def get_plugin_tree(self, namespace: str | None = None) -> list[PluginTreeNode]:
        """Return top-level plugins with their declared child-plugin slots."""
        return [
            self._tree_node(plugin.namespace, plugin.name)
            for plugin in self.list_plugins(namespace)
        ]

    def get_subplugin_options(
        self, parent_path: str, slot_name: str
    ) -> SubpluginSlotSummary:
        """List implementations accepted by one parent slot."""
        namespace, name, canonical = self.resolve_path(parent_path)
        manifest = self.registry.get_plugin_manifest(namespace, name)
        slot = manifest.subplugins.get(slot_name)
        if slot is None:
            raise ValueError(
                f"plugin {canonical!r} has no subplugin slot {slot_name!r}"
            )
        return self._slot_summary(canonical, slot_name, slot)

    def resolve_path(self, path: str) -> tuple[str, str, str]:
        """Resolve a canonical parent/slot.child path to a registry entry."""
        root, *segments = path.split("/")
        if "." not in root:
            raise ValueError("plugin path must start with namespace.name")
        namespace, name = root.split(".", 1)
        namespace = namespace.lower()
        name = name.lower()
        self.registry.get_plugin_class(namespace, name)
        canonical = f"{namespace}.{name}"
        for segment in segments:
            if "." not in segment:
                raise ValueError("child path segments must use slot.child")
            slot_name, child_name = segment.split(".", 1)
            manifest = self.registry.get_plugin_manifest(namespace, name)
            slot = manifest.subplugins.get(slot_name)
            if slot is None:
                raise ValueError(f"plugin {canonical!r} has no slot {slot_name!r}")
            namespace = slot.namespace
            name = child_name.lower()
            self.registry.get_plugin_class(namespace, name)
            canonical = f"{canonical}/{slot_name}.{name}"
        return namespace, name, canonical

    def get_composite_schema(self, plugin_path: str) -> dict[str, Any]:
        """Return one schema plus dynamically discovered child alternatives."""
        namespace, name, canonical = self.resolve_path(plugin_path)
        schema = self.get_schema(namespace, name) or {}
        manifest = self.registry.get_plugin_manifest(namespace, name)
        slots: dict[str, Any] = {}
        for slot_name, slot in manifest.subplugins.items():
            summary = self._slot_summary(canonical, slot_name, slot)
            slots[slot_name] = {
                "namespace": slot.namespace,
                "cardinality": slot.cardinality,
                "default": slot.default,
                "description": slot.description,
                "options": {
                    option.name: self.get_schema(slot.namespace, option.name) or {}
                    for option in summary.options
                },
            }
        return {"path": canonical, "schema": schema, "subplugins": slots}

    def build_template(
        self,
        plugin_path: str,
        *,
        selections: dict[str, str] | None = None,
        existing_values: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a recursively expanded default template for one plugin."""
        from qphase.core.utils import schema_to_yaml_map

        namespace, name, canonical = self.resolve_path(plugin_path)
        schema = self.registry.get_plugin_schema(namespace, name)
        if schema is None:
            raise ValueError(f"no config schema for {canonical}")
        existing = dict(existing_values or {})
        result = schema_to_yaml_map(schema, existing, name, mode="template")
        manifest = self.registry.get_plugin_manifest(namespace, name)
        for slot_name, slot in manifest.subplugins.items():
            selected = (selections or {}).get(slot_name)
            existing_selection = existing.get(slot_name)
            if selected is None and isinstance(existing_selection, dict):
                if len(existing_selection) == 1:
                    selected = next(iter(existing_selection))
            selected = selected or slot.default
            if selected is None:
                continue
            child_existing = (
                existing_selection.get(selected, {})
                if isinstance(existing_selection, dict)
                else {}
            )
            child_path = f"{canonical}/{slot_name}.{selected}"
            result[slot_name] = {
                selected: self.build_template(
                    child_path, existing_values=child_existing
                )
            }
        return result

    def get_schema(self, namespace: str, name: str) -> dict[str, Any] | None:
        schema = self.registry.get_plugin_schema(namespace, name)
        if schema is None:
            return None
        if hasattr(schema, "model_json_schema"):
            return schema.model_json_schema()
        if hasattr(schema, "schema"):
            return schema.schema()
        return None

    def validate_config(self, namespace: str, name: str, config: dict[str, Any]) -> Any:
        config_data = dict(config)
        config_data["name"] = name
        return self.registry.validate_plugin_config(namespace, config_data)

    def get_engine_manifest(self, engine_name: str) -> dict[str, Any] | None:
        engine_cls = self.registry.get_plugin_class("engine", engine_name)
        manifest = getattr(engine_cls, "manifest", None)
        if manifest is None:
            return None
        return {
            "required_plugins": sorted(manifest.required_plugins),
            "optional_plugins": sorted(manifest.optional_plugins),
            "defaults": dict(manifest.defaults),
        }

    def _summary(
        self, namespace: str, name: str, metadata: dict[str, Any]
    ) -> PluginSummary:
        schema = self.registry.get_plugin_schema(namespace, name)
        plugin_class = self.registry.get_plugin_class(namespace, name)
        return PluginSummary(
            namespace=namespace,
            name=name,
            package=metadata.get("package_name"),
            description=metadata.get("description")
            or getattr(plugin_class, "description", ""),
            schema_available=schema is not None,
            entry_point=metadata.get("module_path"),
            metadata=dict(metadata),
        )

    def _tree_node(self, namespace: str, name: str) -> PluginTreeNode:
        metadata = self.registry.list(namespace).get(name, {})
        summary = self._summary(namespace, name, metadata)
        path = f"{namespace}.{name}"
        manifest = self.registry.get_plugin_manifest(namespace, name)
        return PluginTreeNode(
            path=path,
            plugin=summary,
            slots=[
                self._slot_summary(path, slot_name, slot)
                for slot_name, slot in manifest.subplugins.items()
            ],
        )

    def _slot_summary(self, parent_path: str, slot_name: str, slot) -> Any:
        options = []
        for child_name, metadata in self.registry.list(slot.namespace).items():
            if slot.allowed is not None and child_name not in slot.allowed:
                continue
            options.append(
                SubpluginOptionSummary(
                    name=child_name,
                    path=f"{parent_path}/{slot_name}.{child_name}",
                    plugin=self._summary(slot.namespace, child_name, metadata),
                )
            )
        return SubpluginSlotSummary(
            name=slot_name,
            namespace=slot.namespace,
            cardinality=slot.cardinality,
            default=slot.default,
            description=slot.description,
            options=options,
        )
