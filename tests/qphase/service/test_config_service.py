from qphase.core.config import JobConfig
from qphase.service import ConfigService


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
