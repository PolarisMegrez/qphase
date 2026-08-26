import pytest
from pydantic import BaseModel, ConfigDict
from qphase.core.compiler import WorkflowCompiler
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.errors import QPhaseConfigError, QPhasePluginError
from qphase.core.project import ProjectContext
from qphase.core.registry import DiscoveryService, RegistryCenter
from qphase.core.system_config import SystemConfig
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


def test_local_plugin_discovery_uses_worker_import_path_and_contract(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    package = project.root / "qphase_local_contract_plugin"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from pydantic import BaseModel, ConfigDict\n"
        "from qphase.core.protocols import EngineManifest\n"
        "\n"
        "class Config(BaseModel):\n"
        "    model_config = ConfigDict(extra='forbid')\n"
        "    value: int = 0\n"
        "\n"
        "class Engine:\n"
        "    config_schema = Config\n"
        "    manifest = EngineManifest(required_plugins={'analyser'})\n"
        "    marker = 'local'\n",
        encoding="utf-8",
    )
    (project.root / "models" / ".qphase_plugins.yaml").write_text(
        "plugins:\n"
        "  - type: engine.local_contract\n"
        "    target: qphase_local_contract_plugin:Engine\n",
        encoding="utf-8",
    )

    local_registry = RegistryCenter()
    DiscoveryService(local_registry).discover_local_plugins(project)

    plugin = local_registry.get_plugin_class("engine", "local_contract")
    assert plugin.marker == "local"
    manifest = local_registry.get_plugin_manifest("engine", "local_contract")
    assert manifest.required_plugins == {"analyser"}
    assert (
        local_registry.get_plugin_schema("engine", "local_contract")
        is plugin.config_schema
    )

    class StubConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class StubAnalyser:
        config_schema = StubConfig

    local_registry.register("analyser", "stub", StubAnalyser)
    workflow = WorkflowSpec(
        schema_="qphase.workflow/2",
        id="local-contract",
        title="Local contract",
        jobs=[JobConfig(name="job", engine={"local_contract": {}})],
    )
    with pytest.raises(QPhaseConfigError, match="required plugins"):
        WorkflowCompiler(
            project, SystemConfig(), registry_view=local_registry.view()
        ).compile(workflow)

    class NoManifestEngine:
        config_schema = plugin.config_schema

    local_registry.register(
        "engine",
        "no_manifest",
        NoManifestEngine,
        local_import_root=str(project.root),
    )
    no_manifest_workflow = WorkflowSpec(
        schema_="qphase.workflow/2",
        id="no-manifest",
        title="No manifest",
        jobs=[JobConfig(name="job", engine={"no_manifest": {}})],
    )
    with pytest.raises(QPhasePluginError, match="must declare an EngineManifest"):
        WorkflowCompiler(
            project, SystemConfig(), registry_view=local_registry.view()
        ).compile(no_manifest_workflow)

    with pytest.raises(QPhaseConfigError):
        local_registry.validate_plugin_config(
            "engine", {"name": "local_contract", "unknown": True}
        )


def test_local_plugin_discovery_propagates_project_errors(monkeypatch):
    error = QPhaseConfigError("invalid Project")

    def fail_discovery(cls, *args, **kwargs):
        del cls, args, kwargs
        raise error

    monkeypatch.setattr(ProjectContext, "discover", classmethod(fail_discovery))
    with pytest.raises(QPhaseConfigError, match="invalid Project"):
        DiscoveryService(RegistryCenter()).discover_local_plugins()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("plugins: [", "Failed to parse YAML"),
        ("- model.example", "must contain a mapping"),
        ("plugins: {}", "must be a list"),
        ("plugins:\n  - type: model.example\n", "requires a string 'target'"),
    ],
)
def test_local_plugin_discovery_rejects_malformed_manifests(
    tmp_path, payload, message
):
    project = ProjectContext.create(tmp_path / "project")
    manifest = project.root / "models" / ".qphase_plugins.yaml"
    manifest.write_text(payload, encoding="utf-8")

    with pytest.raises(QPhaseConfigError, match=message):
        DiscoveryService(RegistryCenter()).discover_local_plugins(project)
