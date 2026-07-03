from fastapi.testclient import TestClient
from qphase.gui import create_app


def test_gui_console_resource_is_packaged():
    import importlib.resources as resources

    html = (
        resources.files("qphase.gui").joinpath("index.html").read_text(encoding="utf-8")
    )

    assert "QPhase Console" in html
    assert "Results" in html


def test_gui_api_health():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_gui_api_serves_web_console():
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "QPhase Console" in response.text
    assert "Jobs" in response.text
    assert "Results" in response.text


def test_gui_api_lists_and_loads_jobs(temp_workspace, sample_job_file):
    client = TestClient(create_app())

    list_response = client.get("/jobs")
    job_response = client.get("/jobs/test_job")

    assert list_response.status_code == 200
    assert "test_job" in list_response.json()["jobs"]
    assert job_response.status_code == 200
    assert job_response.json()["jobs"][0]["name"] == "test_job"


def test_gui_api_builds_plan(temp_workspace, sample_job_file):
    client = TestClient(create_app())

    response = client.post("/plans", json={"jobs": ["test_job"]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["original_jobs"][0]["name"] == "test_job"
    assert payload["expanded_jobs"][0]["engine"] == "dummy"


def test_gui_api_starts_run(temp_workspace, sample_job_file):
    client = TestClient(create_app())

    response = client.post("/runs", json={"jobs": ["test_job"]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["run"]["status"] == "completed"
    assert payload["run"]["session_id"] is not None
    assert payload["results"][0]["job_name"] == "test_job"
    assert payload["results"][0]["success"] is True


def test_gui_api_reads_run_manifest_and_artifacts(temp_workspace, sample_job_file):
    client = TestClient(create_app())

    run_response = client.post("/runs", json={"jobs": ["test_job"]})
    session_id = run_response.json()["run"]["session_id"]
    manifest_response = client.get(f"/runs/{session_id}")
    artifacts_response = client.get(f"/runs/{session_id}/artifacts")
    events_response = client.get(f"/runs/{session_id}/events")

    assert manifest_response.status_code == 200
    assert manifest_response.json()["session_id"] == session_id
    assert artifacts_response.status_code == 200
    artifacts = artifacts_response.json()["artifacts"]
    assert any(artifact["kind"] == "manifest" for artifact in artifacts)
    assert any(artifact["format"] == "json" for artifact in artifacts)
    assert events_response.status_code == 200
    assert any(
        event["message"] == "Starting job..."
        for event in events_response.json()["events"]
    )


def test_gui_api_reads_json_artifact(temp_workspace, sample_job_file):
    client = TestClient(create_app())

    run_response = client.post("/runs", json={"jobs": ["test_job"]})
    session_id = run_response.json()["run"]["session_id"]
    artifacts = client.get(f"/runs/{session_id}/artifacts").json()["artifacts"]
    manifest_path = next(
        artifact["path"] for artifact in artifacts if artifact["kind"] == "manifest"
    )
    artifact_response = client.get("/artifacts", params={"path": manifest_path})

    assert artifact_response.status_code == 200
    payload = artifact_response.json()
    assert payload["content_type"] == "application/json"
    assert payload["content"]["session_id"] == session_id


def test_gui_api_exposes_plugin_catalog_and_schema():
    client = TestClient(create_app())

    catalog_response = client.get("/plugins", params={"namespace": "engine"})
    schema_response = client.get("/plugins/engine/dummy/schema")

    assert catalog_response.status_code == 200
    assert any(
        plugin["name"] == "dummy" for plugin in catalog_response.json()["plugins"]
    )
    assert schema_response.status_code == 200
    assert "param" in schema_response.json()["properties"]


def test_gui_api_round_trips_global_config(temp_workspace):
    client = TestClient(create_app())

    put_response = client.put(
        "/config/global", json={"data": {"backend": {"dummy": {"param": 2.0}}}}
    )
    get_response = client.get("/config/global")

    assert put_response.status_code == 200
    assert get_response.status_code == 200
    assert get_response.json()["backend"]["dummy"]["param"] == 2.0
