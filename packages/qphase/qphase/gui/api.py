"""FastAPI app factory for the local QPhase GUI backend."""

from __future__ import annotations

import importlib.resources as resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from qphase.core.errors import QPhaseError
from qphase.core.scheduler import JobProgressUpdate
from qphase.service import ConfigService, RegistryService, SchedulerService


class JobSelectionRequest(BaseModel):
    jobs: list[str] = Field(default_factory=list)


class RunRequest(JobSelectionRequest):
    resume_from: str | None = None


class JobValidationRequest(BaseModel):
    job: dict[str, Any]


class GlobalConfigRequest(BaseModel):
    data: dict[str, Any]


def create_app(
    *,
    config_service: ConfigService | None = None,
    registry_service: RegistryService | None = None,
    scheduler_service: SchedulerService | None = None,
) -> FastAPI:
    """Create a local FastAPI app backed by QPhase services."""
    config = config_service or ConfigService()
    registry = registry_service or RegistryService()
    scheduler = scheduler_service or SchedulerService(config.load_system_config())

    app = FastAPI(title="QPhase Local GUI API", version="0.1.0")
    app.state.run_events = {}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _dashboard_html()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/jobs")
    def list_jobs() -> dict[str, list[str]]:
        return {"jobs": scheduler.list_jobs()}

    @app.get("/jobs/{name}")
    def get_job(name: str) -> dict[str, Any]:
        try:
            job_list = scheduler.load_jobs([name])
        except Exception as exc:
            raise _http_error(exc, status_code=404) from exc
        return job_list.model_dump(mode="json")

    @app.post("/jobs/validate")
    def validate_job(request: JobValidationRequest) -> dict[str, Any]:
        try:
            issues = config.validate_against_registry(request.job)
        except Exception as exc:
            raise _http_error(exc) from exc
        return {
            "validation_issues": [issue.model_dump(mode="json") for issue in issues]
        }

    @app.post("/plans")
    def build_plan(request: JobSelectionRequest) -> dict[str, Any]:
        if not request.jobs:
            raise HTTPException(status_code=400, detail="At least one job is required")
        try:
            job_list = scheduler.load_jobs(request.jobs)
            plan = scheduler.build_plan(job_list)
        except Exception as exc:
            raise _http_error(exc) from exc
        return plan.model_dump(mode="json")

    @app.post("/runs")
    def start_run(request: RunRequest) -> dict[str, Any]:
        if not request.jobs:
            raise HTTPException(status_code=400, detail="At least one job is required")
        events: list[dict[str, Any]] = []
        try:
            job_list = scheduler.load_jobs(request.jobs)
            results = scheduler.run(
                job_list,
                progress_callback=_record_progress(events),
                resume_from=request.resume_from,
            )
        except Exception as exc:
            raise _http_error(exc) from exc
        if scheduler.last_run_handle and scheduler.last_run_handle.session_id:
            app.state.run_events[scheduler.last_run_handle.session_id] = events
        return {
            "run": (
                scheduler.last_run_handle.model_dump(mode="json")
                if scheduler.last_run_handle is not None
                else None
            ),
            "results": [
                {
                    "job_index": result.job_index,
                    "job_name": result.job_name,
                    "run_dir": str(result.run_dir),
                    "run_id": result.run_id,
                    "success": result.success,
                    "error": result.error,
                }
                for result in results
            ],
        }

    @app.get("/runs/{session_id}/events")
    def list_run_events(session_id: str) -> dict[str, Any]:
        return {"events": app.state.run_events.get(session_id, [])}

    @app.get("/plugins")
    def list_plugins(namespace: str | None = None) -> dict[str, Any]:
        plugins = registry.list_plugins(namespace)
        return {"plugins": [plugin.model_dump(mode="json") for plugin in plugins]}

    @app.get("/plugins/{namespace}/{name}/schema")
    def get_plugin_schema(namespace: str, name: str) -> dict[str, Any]:
        schema = registry.get_schema(namespace, name)
        if schema is None:
            raise HTTPException(status_code=404, detail="Plugin schema not found")
        return schema

    @app.get("/config/global")
    def get_global_config() -> dict[str, Any]:
        return config.load_global_config()

    @app.put("/config/global")
    def put_global_config(request: GlobalConfigRequest) -> dict[str, str]:
        config.save_global_config(request.data)
        return {"status": "saved"}

    @app.get("/runs/{session_id}")
    def get_run(session_id: str) -> dict[str, Any]:
        session_dir = Path(scheduler.system_config.paths.output_dir) / session_id
        try:
            return scheduler.load_session_manifest(session_dir)
        except Exception as exc:
            raise _http_error(exc, status_code=404) from exc

    @app.get("/runs/{session_id}/artifacts")
    def list_run_artifacts(session_id: str) -> dict[str, Any]:
        session_dir = Path(scheduler.system_config.paths.output_dir) / session_id
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Run session not found")
        artifacts = scheduler.list_artifacts(session_dir)
        return {
            "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts]
        }

    @app.get("/artifacts")
    def get_artifact(path: str) -> dict[str, Any]:
        try:
            return scheduler.load_artifact(path)
        except Exception as exc:
            raise _http_error(exc, status_code=404) from exc

    return app


def _http_error(exc: Exception, status_code: int = 400) -> HTTPException:
    if isinstance(exc, QPhaseError):
        return HTTPException(status_code=status_code, detail=str(exc))
    return HTTPException(status_code=status_code, detail=str(exc))


def _record_progress(events: list[dict[str, Any]]):
    def _on_progress(update: JobProgressUpdate) -> None:
        events.append(
            {
                "job_name": update.job_name,
                "job_index": update.job_index,
                "total_jobs": update.total_jobs,
                "message": update.message,
                "percent": update.percent,
                "job_eta": update.job_eta,
                "global_eta": update.global_eta,
                "stage": update.stage,
            }
        )

    return _on_progress


def _dashboard_html() -> str:
    return (
        resources.files("qphase.gui").joinpath("index.html").read_text(encoding="utf-8")
    )
