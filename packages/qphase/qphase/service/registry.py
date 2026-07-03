"""Registry service facade."""

from __future__ import annotations

from typing import Any

from qphase.core.registry import RegistryCenter, discovery, registry

from .models import PluginCatalog, PluginSummary


class RegistryService:
    """Structured API over the core plugin registry."""

    def __init__(self, registry_center: RegistryCenter | None = None):
        self.registry = registry_center or registry

    def discover(self, include_local: bool = True) -> PluginCatalog:
        discovery.discover_plugins()
        if include_local:
            discovery.discover_local_plugins()
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

    def get_scanable_params(self, namespace: str, name: str) -> dict[str, bool]:
        return self.registry.get_scanable_params(namespace, name)

    def _summary(
        self, namespace: str, name: str, metadata: dict[str, Any]
    ) -> PluginSummary:
        schema = self.registry.get_plugin_schema(namespace, name)
        return PluginSummary(
            namespace=namespace,
            name=name,
            package=metadata.get("package_name"),
            description=metadata.get("description", ""),
            schema_available=schema is not None,
            entry_point=metadata.get("module_path"),
            metadata=dict(metadata),
        )
