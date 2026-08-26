import sys

import pytest
from qphase.core.errors import QPhaseConfigError
from qphase.core.project import ProjectContext
from qphase.core.registry import DiscoveryService, RegistryCenter
from qphase.service import RegistryService


def test_registry_service_catalog_includes_registered_dummy_plugins():
    service = RegistryService()

    catalog = service.get_catalog()

    assert "engine" in catalog.namespaces
    assert any(
        plugin.namespace == "engine"
        and plugin.name == "dummy"
        and plugin.schema_available
        for plugin in catalog.plugins
    )


def test_registry_service_returns_json_schema_and_rejects_list_scan_syntax():
    service = RegistryService()

    schema = service.get_schema("engine", "dummy")

    assert schema is not None
    assert "param" in schema["properties"]
    with pytest.raises(QPhaseConfigError):
        service.validate_config("engine", "dummy", {"param": [1.0, 2.0]})


def test_registry_snapshot_is_not_changed_by_source_mutation():
    class First:
        pass

    class Second:
        pass

    source = RegistryCenter()
    source.register("engine", "example", First)
    snapshot = source.snapshot()

    source.register("engine", "example", Second, overwrite=True)

    assert snapshot.get_plugin_class("engine", "example") is First
    assert source.get_plugin_class("engine", "example") is Second


def test_local_plugin_discovery_keeps_projects_out_of_sys_path_and_sys_modules(
    tmp_path,
):
    def make_project(name: str, marker: str) -> ProjectContext:
        project = ProjectContext.create(tmp_path / name)
        models = project.root / "models"
        (models / "__init__.py").write_text("", encoding="utf-8")
        (models / "plugin.py").write_text(
            f"class Model:\n    marker = {marker!r}\n", encoding="utf-8"
        )
        (models / ".qphase_plugins.yaml").write_text(
            "plugins:\n"
            "  - type: model.example\n"
            "    target: models.plugin:Model\n",
            encoding="utf-8",
        )
        return project

    first = make_project("first", "first")
    second = make_project("second", "second")
    first_registry = RegistryCenter()
    second_registry = RegistryCenter()
    before = list(sys.path)

    DiscoveryService(first_registry).discover_local_plugins(first)
    DiscoveryService(second_registry).discover_local_plugins(second)

    assert sys.path == before
    first_class = first_registry.get_plugin_class("model", "example")
    second_class = second_registry.get_plugin_class("model", "example")
    assert first_class.marker == "first"
    assert second_class.marker == "second"
    assert first_class is not second_class
