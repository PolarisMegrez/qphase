from qphase.core.config import JobConfig
from qphase.core.registry import registry
from qphase.service import ConfigService

from tests.plugins.dummy_plugin import DummyPlugin


def test_config_service_previews_merged_config(temp_workspace):
    job = JobConfig(
        name="preview",
        engine={"dummy": {"param": 2.0}},
        plugins={"backend": {"dummy": {"param": 1.0}}},
    )
    service = ConfigService()

    preview = service.preview_merged_config(job)

    assert preview.job_name == "preview"
    assert preview.merged_config["engine"]["dummy"] == {"param": 2.0}
    assert preview.validation_issues == []


def test_config_service_reports_registry_validation_issue(temp_workspace):
    service = ConfigService()

    issues = service.validate_against_registry(
        {
            "name": "bad",
            "engine": {"missing": {}},
        }
    )

    assert len(issues) == 1
    assert issues[0].path == "engine.missing"


def test_config_service_previews_dynamic_project_plugin_defaults(temp_workspace):
    registry.register(
        namespace="research_solver",
        name="search",
        builder=DummyPlugin,
        overwrite=True,
    )
    global_file = temp_workspace / "configs" / "defaults.yaml"
    global_file.write_text(
        "research_solver:\n  search:\n    seed: 42\n",
        encoding="utf-8",
    )
    job = JobConfig(
        name="cam_preview",
        engine={"cam": {}},
        plugins={"research_solver": {"search": {"param": 3.0}}},
    )

    preview = ConfigService().preview_merged_config(job)

    assert preview.merged_config["plugins"]["research_solver"]["search"] == {
        "seed": 42,
        "param": 3.0,
    }
