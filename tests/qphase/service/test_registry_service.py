import pytest
from qphase.core.errors import QPhaseConfigError
from qphase.core.registry import RegistryCenter
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
