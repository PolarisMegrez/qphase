"""FastAPI app factory for the local QPhase GUI backend."""

from __future__ import annotations

import importlib.resources as resources
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from qphase.core.errors import QPhaseError
from qphase.core.system_config import load_system_config
from qphase.service import (
    ConfigService,
    ExecutionManager,
    ProjectService,
    RegistryService,
    SchedulerService,
)

from .application import ApplicationContext


class WorkflowSelectionRequest(BaseModel):
    workflow: str = Field(min_length=1)


class ExecutionRequest(WorkflowSelectionRequest):
    resume_from: str | None = None


class JobValidationRequest(BaseModel):
    job: dict[str, Any]


class ProjectDefaultsRequest(BaseModel):
    data: dict[str, Any]


class SessionUpdateRequest(BaseModel):
    alias: str | None = None
    note: str | None = None


class WorkflowDocumentRequest(BaseModel):
    content: str


class PendingJobRevisionRequest(BaseModel):
    job: dict[str, Any]


def create_app(
    *,
    config_service: ConfigService | None = None,
    registry_service: RegistryService | None = None,
    scheduler_service: SchedulerService | None = None,
    application_context: ApplicationContext | None = None,
) -> FastAPI:
    """Create a local FastAPI app backed by QPhase services."""
    owned_context = application_context is None
    if application_context is not None:
        context = application_context
    elif any((config_service, registry_service, scheduler_service)):
        system = (
            config_service.load_system_config()
            if config_service is not None
            else scheduler_service.system_config
            if scheduler_service is not None
            else load_system_config()
        )
        project = (
            scheduler_service.project
            if scheduler_service
            else application_context.project
            if application_context
            else None
        )
        if project is None:
            from qphase.core.project import ProjectContext

            project = ProjectContext.discover()
        config = config_service or ConfigService(project, system)
        registry = registry_service or RegistryService(
            system_config=system, project=project
        )
        scheduler = scheduler_service or SchedulerService(system, project=project)
        context = ApplicationContext(
            project,
            system,
            config,
            registry,
            scheduler,
            ExecutionManager(scheduler),
            ProjectService(project),
        )
    else:
        context = ApplicationContext.create()
    config = context.config
    registry = context.registry
    scheduler = context.scheduler

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if owned_context:
            context.close()

    app = FastAPI(title="QPhase Local GUI API", version="2.0.0", lifespan=lifespan)
    app.state.context = context

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _dashboard_html()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/project")
    def get_project() -> dict[str, Any]:
        project = context.project
        return {
            "schema": project.manifest.schema_,
            "project_id": project.project_id,
            "name": project.manifest.name,
            "root": str(project.root),
            "paths": {
                "workflows": str(project.workflow_root),
                "defaults": str(project.defaults_path),
                "plugins": [str(path) for path in project.plugin_dirs],
                "sessions": str(project.session_root),
            },
        }

    @app.get("/workflows")
    def list_workflows(
        collection: str | None = None,
        tag: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        return {
            "workflows": [
                item.__dict__
                for item in scheduler.catalog.search(
                    collection=collection, tag=tag, query=query
                )
            ]
        }

    @app.get("/workflows/{workflow_id}")
    def get_workflow(workflow_id: str) -> dict[str, Any]:
        try:
            workflow = scheduler.load_workflow(workflow_id)
        except Exception as exc:
            raise _http_error(exc, status_code=404) from exc
        return workflow.model_dump(mode="json", by_alias=True)

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
    def build_plan(request: WorkflowSelectionRequest) -> dict[str, Any]:
        try:
            workflow = scheduler.load_workflow(request.workflow)
            plan = scheduler.build_plan(workflow)
        except Exception as exc:
            raise _http_error(exc) from exc
        return plan.model_dump(mode="json")

    @app.post("/executions", status_code=202)
    def submit_execution(request: ExecutionRequest) -> dict[str, Any]:
        try:
            return context.executions.submit(
                request.workflow, resume_from=request.resume_from
            ).model_dump(mode="json")
        except Exception as exc:
            raise _http_error(exc, status_code=429) from exc

    @app.get("/executions")
    def list_executions() -> dict[str, Any]:
        return {
            "executions": [
                item.model_dump(mode="json")
                for item in context.executions.list_executions()
            ]
        }

    @app.get("/executions/{execution_id}")
    def get_execution(execution_id: str) -> dict[str, Any]:
        try:
            return context.executions.get(execution_id).model_dump(mode="json")
        except Exception as exc:
            raise _http_error(exc, status_code=404) from exc

    @app.get("/executions/{execution_id}/events")
    def get_execution_events(execution_id: str, after_seq: int = 0) -> dict[str, Any]:
        try:
            events = context.executions.events(execution_id, after=after_seq)
        except Exception as exc:
            raise _http_error(exc, status_code=404) from exc
        return {"events": [event.model_dump(mode="json") for event in events]}

    @app.post("/executions/{execution_id}/cancel")
    def cancel_execution(execution_id: str) -> dict[str, Any]:
        return _execution_action(context.executions.cancel, execution_id)

    @app.post("/executions/{execution_id}/pause")
    def pause_execution(execution_id: str) -> dict[str, Any]:
        return _execution_action(context.executions.request_pause, execution_id)

    @app.post("/executions/{execution_id}/resume")
    def resume_execution(execution_id: str) -> dict[str, Any]:
        return _execution_action(context.executions.resume, execution_id)

    @app.post("/executions/{execution_id}/jobs/{job_name}/cancel")
    def cancel_execution_job(execution_id: str, job_name: str) -> dict[str, Any]:
        try:
            return context.executions.cancel_job(execution_id, job_name).model_dump(
                mode="json"
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.put("/executions/{execution_id}/jobs/{job_name}")
    def revise_pending_job(
        execution_id: str, job_name: str, request: PendingJobRevisionRequest
    ) -> dict[str, Any]:
        try:
            return context.executions.revise_pending_job(
                execution_id, job_name, request.job
            ).model_dump(mode="json")
        except Exception as exc:
            raise _http_error(exc, status_code=409) from exc

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

    @app.get("/config/project-defaults")
    def get_project_defaults() -> dict[str, Any]:
        return config.load_project_defaults()

    @app.put("/config/project-defaults")
    def put_project_defaults(request: ProjectDefaultsRequest) -> dict[str, str]:
        config.save_project_defaults(request.data)
        return {"status": "saved"}

    @app.get("/sessions")
    def list_sessions() -> dict[str, Any]:
        return {
            "sessions": [
                item.model_dump(mode="json")
                for item in context.project_service.list_sessions()
            ]
        }

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        try:
            return context.project_service.get_session(session_id).model_dump(
                mode="json"
            )
        except Exception as exc:
            raise _http_error(exc, status_code=404) from exc

    @app.get("/sessions/{session_id}/events")
    def get_session_events(session_id: str, after_seq: int = 0) -> dict[str, Any]:
        try:
            return {
                "events": context.project_service.session_events(
                    session_id, after=after_seq
                )
            }
        except Exception as exc:
            raise _http_error(exc, status_code=404) from exc

    @app.patch("/sessions/{session_id}")
    def update_session(
        session_id: str, request: SessionUpdateRequest
    ) -> dict[str, Any]:
        try:
            fields = request.model_dump(exclude_unset=True)
            return context.project_service.update_session(
                session_id, **fields
            ).model_dump(mode="json")
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> Response:
        try:
            context.project_service.trash_session(session_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        return Response(status_code=204)

    @app.post("/sessions/purge")
    def purge_sessions() -> dict[str, int]:
        return {"purged": context.project_service.purge_trash()}

    @app.get("/sessions/{session_id}/artifacts")
    def list_session_artifacts(session_id: str) -> dict[str, Any]:
        root = context.project_service.session_dir(session_id)
        return {
            "artifacts": [
                artifact.model_dump(mode="json")
                for artifact in scheduler.list_artifacts(root)
            ]
        }

    @app.get("/sessions/{session_id}/artifacts/{artifact_id}")
    def get_session_artifact(session_id: str, artifact_id: str) -> dict[str, Any]:
        root = context.project_service.session_dir(session_id)
        try:
            return scheduler.load_artifact_by_id(artifact_id, session_dir=root)
        except Exception as exc:
            raise _http_error(exc, status_code=404) from exc

    @app.get("/sessions/{session_id}/jobs/{job_name}/products")
    def get_job_products(session_id: str, job_name: str) -> dict[str, Any]:
        root = context.project_service.session_dir(session_id)
        try:
            catalog = scheduler.describe_products(job_name, session_dir=root)
        except Exception as exc:
            raise _http_error(exc, status_code=404) from exc
        return catalog.model_dump(mode="json")

    @app.get("/workflow-docs")
    def list_workflow_documents() -> dict[str, Any]:
        return {
            "documents": [
                document.model_dump(mode="json")
                for document in context.project_service.list_workflow_documents()
            ]
        }

    @app.get("/workflow-docs/{doc_id:path}")
    def get_workflow_document(doc_id: str) -> dict[str, Any]:
        try:
            return context.project_service.get_workflow_document(doc_id).model_dump(
                mode="json"
            )
        except Exception as exc:
            raise _http_error(exc, status_code=404) from exc

    @app.put("/workflow-docs/{doc_id:path}")
    def put_workflow_document(
        doc_id: str,
        request: WorkflowDocumentRequest,
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        try:
            return context.project_service.put_workflow_document(
                doc_id, request.content, revision=if_match
            ).model_dump(mode="json")
        except RuntimeError as exc:
            raise _http_error(exc, status_code=409) from exc
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.delete("/workflow-docs/{doc_id:path}", status_code=204)
    def delete_workflow_document(
        doc_id: str,
        if_match: str = Header(alias="If-Match"),
    ) -> Response:
        try:
            context.project_service.delete_workflow_document(doc_id, revision=if_match)
        except RuntimeError as exc:
            raise _http_error(exc, status_code=409) from exc
        except Exception as exc:
            raise _http_error(exc) from exc
        return Response(status_code=204)

    return app


def _http_error(exc: Exception, status_code: int = 400) -> HTTPException:
    if isinstance(exc, QPhaseError):
        return HTTPException(status_code=status_code, detail=str(exc))
    return HTTPException(status_code=status_code, detail=str(exc))


def _execution_action(action: Any, execution_id: str) -> dict[str, Any]:
    try:
        return action(execution_id).model_dump(mode="json")
    except Exception as exc:
        raise _http_error(exc, status_code=409) from exc


def _dashboard_html() -> str:
    return (
        resources.files("qphase.gui").joinpath("index.html").read_text(encoding="utf-8")
    )
