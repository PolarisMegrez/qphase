import json
import os
import time
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from qphase.gui import create_app

pytestmark = pytest.mark.integration


def _execute_workflow(client: TestClient, workflow: str = "test_job") -> dict:
    response = client.post("/executions", json={"workflow": workflow})
    assert response.status_code == 202
    execution_id = response.json()["execution_id"]
    for _ in range(200):
        payload = client.get(f"/executions/{execution_id}").json()
        if payload["state"] in {"completed", "failed", "cancelled", "partial"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("execution did not finish")


def test_gui_console_resource_is_packaged():
    import importlib.resources as resources

    html = (
        resources.files("qphase.gui").joinpath("index.html").read_text(encoding="utf-8")
    )

    assert "QPhase Workbench" in html
    assert "Executions" in html


def test_gui_console_exposes_job_boundary_controls():
    import importlib.resources as resources

    html = (
        resources.files("qphase.gui").joinpath("index.html").read_text(encoding="utf-8")
    )

    assert "Pause at boundary" in html
    assert "Logical jobs" in html


def test_gui_api_health():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_gui_api_exposes_current_project(temp_workspace):
    client = TestClient(create_app())

    response = client.get("/project")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == "qphase.project/2"
    assert payload["root"] == str(temp_workspace)
    assert payload["paths"]["workflows"].endswith("configs\\workflows")


def test_gui_api_serves_web_console():
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "QPhase Workbench" in response.text
    assert "Workflows" in response.text
    assert "Sessions" in response.text


def test_gui_api_lists_and_loads_workflows(temp_workspace, sample_job_file):
    client = TestClient(create_app())

    list_response = client.get("/workflows")
    job_response = client.get("/workflows/test_job")

    assert list_response.status_code == 200
    assert "test_job" in [item["id"] for item in list_response.json()["workflows"]]
    assert job_response.status_code == 200
    assert job_response.json()["jobs"][0]["name"] == "test_job"


def test_gui_api_filters_workflow_catalog(temp_workspace, sample_job_file):
    client = TestClient(create_app())

    response = client.get("/workflows", params={"query": "test_job"})
    missing = client.get("/workflows", params={"tag": "not-present"})

    assert [item["id"] for item in response.json()["workflows"]] == ["test_job"]
    assert missing.json()["workflows"] == []


def test_gui_api_builds_plan(temp_workspace, sample_job_file):
    client = TestClient(create_app())

    response = client.post("/plans", json={"workflow": "test_job"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["jobs"][0]["name"] == "test_job"
    assert payload["jobs"][0]["engine"] == "dummy"


def test_gui_api_submits_asynchronous_execution(temp_workspace, sample_job_file):
    with TestClient(create_app()) as client:
        response = client.post("/executions", json={"workflow": "test_job"})
        assert response.status_code == 202
        execution_id = response.json()["execution_id"]

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            payload = client.get(f"/executions/{execution_id}").json()
            if payload["state"] in {"completed", "failed"}:
                break
            time.sleep(0.02)

        events = client.get(f"/executions/{execution_id}/events").json()["events"]

    assert payload["state"] == "completed"
    assert payload["session_id"] is not None
    assert payload["jobs"][0]["name"] == "test_job"
    assert {item["path"] for item in payload["jobs"][0]["plugins"]} == {
        "backend.dummy",
        "model.dummy",
    }
    assert any(event["payload"]["kind"] == "job_completed" for event in events)
    journal_path = next(
        path
        for path in (temp_workspace / "runs").rglob("events.jsonl")
        if path.parent.name == payload["session_id"]
    )
    journal = journal_path.read_text(encoding="utf-8")
    assert '"kind": "execution_queued"' in journal
    assert '"kind": "execution_started"' in journal


def test_gui_api_reads_run_manifest_and_artifacts(temp_workspace, sample_job_file):
    with TestClient(create_app()) as client:
        session_id = _execute_workflow(client)["session_id"]
        manifest_response = client.get(f"/sessions/{session_id}")
        artifacts_response = client.get(f"/sessions/{session_id}/artifacts")
        events_response = client.get(f"/sessions/{session_id}/events")

    assert manifest_response.status_code == 200
    assert manifest_response.json()["session_id"] == session_id
    assert artifacts_response.status_code == 200
    artifacts = artifacts_response.json()["artifacts"]
    assert any(artifact["kind"] == "manifest" for artifact in artifacts)
    assert any(artifact["format"] == "json" for artifact in artifacts)
    assert events_response.status_code == 200
    assert any(
        event["payload"].get("message") == "Starting job..."
        for event in events_response.json()["events"]
    )


def test_gui_api_reads_json_file_by_reference(temp_workspace, sample_job_file):
    with TestClient(create_app()) as client:
        session_id = _execute_workflow(client)["session_id"]
        artifacts = client.get(f"/sessions/{session_id}/artifacts").json()["artifacts"]
        manifest = next(
            artifact for artifact in artifacts if artifact["kind"] == "manifest"
        )
        artifact_response = client.get(
            f"/sessions/{session_id}/files/{manifest['file_ref']}"
        )

    assert artifact_response.status_code == 200
    payload = artifact_response.json()
    assert payload["content_type"] == "application/json"
    assert payload["content"]["session_id"] == session_id


def test_gui_api_reads_session_file_by_reference(temp_workspace, sample_job_file):
    with TestClient(create_app()) as client:
        session_id = _execute_workflow(client)["session_id"]
        artifacts = client.get(f"/sessions/{session_id}/artifacts").json()["artifacts"]
        manifest = next(item for item in artifacts if item["kind"] == "manifest")
        response = client.get(
            f"/sessions/{session_id}/files/{manifest['file_ref']}"
        )

    assert response.status_code == 200
    assert response.json()["content"]["session_id"] == session_id


def test_gui_api_preserves_session_note_when_alias_changes(
    temp_workspace, sample_job_file
):
    with TestClient(create_app()) as client:
        session_id = _execute_workflow(client)["session_id"]
        client.patch(f"/sessions/{session_id}", json={"note": "keep this"})
        response = client.patch(
            f"/sessions/{session_id}", json={"alias": "reference"}
        )

    assert response.status_code == 200
    assert response.json()["alias"] == "reference"
    assert response.json()["note"] == "keep this"


def test_gui_api_rejects_unknown_file_ref(temp_workspace, sample_job_file):
    with TestClient(create_app()) as client:
        session_id = _execute_workflow(client)["session_id"]
        response = client.get(f"/sessions/{session_id}/files/not-a-file.json")

    assert response.status_code == 404


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


def test_gui_api_round_trips_project_defaults(temp_workspace):
    client = TestClient(create_app())

    put_response = client.put(
        "/config/project-defaults",
        json={"data": {"backend": {"dummy": {"param": 2.0}}}},
    )
    get_response = client.get("/config/project-defaults")

    assert put_response.status_code == 200
    assert get_response.status_code == 200
    assert get_response.json()["backend"]["dummy"]["param"] == 2.0


def test_gui_api_manages_workflow_document_with_revision(
    temp_workspace, sample_job_file
):
    with TestClient(create_app()) as client:
        document = client.get("/workflow-docs/test_job.yaml").json()
        content = document["content"].replace("param: 10.0", "param: 11.0")
        response = client.put(
            "/workflow-docs/test_job.yaml",
            headers={"If-Match": document["revision"]},
            json={"content": content},
        )
        conflict = client.put(
            "/workflow-docs/test_job.yaml",
            headers={"If-Match": document["revision"]},
            json={"content": content},
        )

    assert response.status_code == 200
    assert response.json()["revision"] != document["revision"]
    assert conflict.status_code == 409


def test_gui_api_marks_stale_running_session_interrupted(temp_workspace):
    run_dir = temp_workspace / "runs" / "stale-session"
    run_dir.mkdir()
    (run_dir / "session_manifest.json").write_text(
        '{"session_id":"stale-session","start_time":"2026-01-01T00:00:00",'
        '"status":"running","jobs":{}}',
        encoding="utf-8",
    )

    with TestClient(create_app()) as client:
        response = client.get("/sessions/stale-session")

    assert response.status_code == 200
    assert response.json()["status"] == "interrupted"


def test_gui_api_preserves_live_running_session_status(temp_workspace):
    run_dir = temp_workspace / "runs" / "live-session"
    run_dir.mkdir()
    (run_dir / "session_manifest.json").write_text(
        '{"session_id":"live-session","start_time":"2026-01-01T00:00:00",'
        '"status":"running","jobs":{}}',
        encoding="utf-8",
    )
    (run_dir / "session.lock").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "session_id": "live-session",
                "heartbeat": datetime.now().astimezone().isoformat(),
            }
        ),
        encoding="utf-8",
    )

    with TestClient(create_app()) as client:
        response = client.get("/sessions/live-session")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_gui_api_lists_job_products(temp_workspace):
    import numpy as np
    from qphase.data import (
        AxisRole,
        AxisSchema,
        DataKind,
        ProductSchema,
        TimeSeriesDataset,
        VariableSchema,
        save_products,
    )

    run_dir = temp_workspace / "runs" / "products-session"
    job_dir = run_dir / "job1"
    job_dir.mkdir(parents=True)
    (run_dir / "session_manifest.json").write_text(
        '{"session_id":"products-session","start_time":"2026-01-01T00:00:00",'
        '"status":"completed","jobs":{}}',
        encoding="utf-8",
    )
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=2),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=4,
                coordinate="regular",
                start=0.0,
                step=0.1,
                units="s",
            ),
        ],
        variables=[
            VariableSchema(
                name="x",
                dtype="complex128",
                value_domain="complex",
                dims=("trajectory", "time"),
            ),
        ],
    )
    dataset = TimeSeriesDataset.from_arrays(
        schema,
        {"x": np.zeros((2, 4), dtype=np.complex128)},
        owner="engine.fake",
    )
    save_products(job_dir, {"trajectories": dataset})

    with TestClient(create_app()) as client:
        response = client.get("/sessions/products-session/jobs/job1/products")
        missing = client.get("/sessions/products-session/jobs/nope/products")

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact_id"]
    assert payload["size"] > 0
    product = payload["products"][0]
    assert product["name"] == "trajectories"
    assert product["kind"] == "time_series"
    assert product["backing"] == "artifact"
    assert product["chunk_count"] == 1
    axes = {axis["name"]: axis for axis in product["axes"]}
    assert axes["time"]["start"] == 0.0
    assert axes["time"]["step"] == 0.1
    assert missing.status_code == 404


def test_gui_api_job_products_maps_removed_hash_to_422(temp_workspace):
    import numpy as np
    from qphase.data import (
        AxisRole,
        AxisSchema,
        DataKind,
        ProductSchema,
        TimeSeriesDataset,
        VariableSchema,
        save_products,
    )

    run_dir = temp_workspace / "runs" / "corrupt-session"
    job_dir = run_dir / "job1"
    job_dir.mkdir(parents=True)
    (run_dir / "session_manifest.json").write_text(
        '{"session_id":"corrupt-session","start_time":"2026-01-01T00:00:00",'
        '"status":"completed","jobs":{}}',
        encoding="utf-8",
    )
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[AxisSchema(name="time", role=AxisRole.COORDINATE, size=2)],
        variables=[
            VariableSchema(
                name="x",
                dtype="float64",
                value_domain="real",
                dims=("time",),
            ),
        ],
    )
    dataset = TimeSeriesDataset.from_arrays(
        schema,
        {"x": np.zeros(2)},
        owner="engine.fake",
    )
    save_products(job_dir, {"trajectories": dataset})
    manifest_path = job_dir / "artifact_manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["content_hash"] = "removed"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with TestClient(create_app()) as client:
        response = client.get("/sessions/corrupt-session/jobs/job1/products")
        listing = client.get("/sessions/corrupt-session/artifacts")
        missing = client.get("/sessions/corrupt-session/jobs/nope/products")

    assert response.status_code == 422
    assert listing.status_code == 422
    assert missing.status_code == 404
