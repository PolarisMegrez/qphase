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


def test_registry_service_returns_json_schema_and_scanable_params():
    service = RegistryService()

    schema = service.get_schema("engine", "dummy")
    scanable = service.get_scanable_params("engine", "dummy")

    assert schema is not None
    assert "param" in schema["properties"]
    assert scanable == {"param": True, "description": False}
