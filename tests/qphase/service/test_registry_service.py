import pytest
from qphase.core.errors import QPhaseConfigError
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
