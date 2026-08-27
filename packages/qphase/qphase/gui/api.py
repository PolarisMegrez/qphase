"""FastAPI app factory for the local QPhase GUI backend."""

from __future__ import annotations

import importlib.resources as resources
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from qphase.core.annotations import Lifecycle, RetentionPolicy
from qphase.core.catalog import OBJECT_KINDS, CatalogQuery
from qphase.core.errors import QPhaseError, QPhaseIOError
from qphase.core.system_config import load_system_config
from qphase.data.errors import ArtifactError
from qphase.service import (
    CatalogService,
    ConfigService,
    ExecutionManager,
    ProjectService,
    RegistryService,
    SchedulerService,
)
from qphase.service.catalog import parse_facet_filters, parse_range_filters
from qphase.service.private import UserPrivateStore

from .application import ApplicationContext


class WorkflowSelectionRequest(BaseModel):
    workflow: str = Field(min_length=1)


class ExecutionRequest(WorkflowSelectionRequest):
    resume_from: str | None = None
    tags: list[str] = Field(default_factory=list)


class JobValidationRequest(BaseModel):
    job: dict[str, Any]


class ProjectDefaultsRequest(BaseModel):
    data: dict[str, Any]


class SessionUpdateRequest(BaseModel):
    alias: str | None = None
    note: str | None = None
    lifecycle: Lifecycle | None = None
    retention: RetentionPolicy | None = None


class WorkflowDocumentRequest(BaseModel):
    content: str


class PendingJobRevisionRequest(BaseModel):
    job: dict[str, Any]


class TagsUpdateRequest(BaseModel):
    add: list[str] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)
    private: bool = False


class SavedViewRequest(BaseModel):
    object_kind: str
    tags_all: list[str] = Field(default_factory=list)
    lifecycle: str | None = None
    retention: str | None = None
    limit: int = 100


class PromoteTagRequest(BaseModel):
    tag: str = Field(min_length=1)


class ArtifactUpdateRequest(BaseModel):
    lifecycle: Lifecycle | None = None
    retention: RetentionPolicy | None = None


class OccurrenceUpdateRequest(BaseModel):
    retention: RetentionPolicy | None = None


class ExecutionTagsRequest(BaseModel):
    tags: list[str] = Field(default_factory=list)


class ProjectUpdateRequest(BaseModel):
    alias: str | None = None
    note: str | None = None
    private: bool = False


class PrivateAnnotationRequest(BaseModel):
    alias: str | None = None
    note: str | None = None


_TAGS_QUERY = Query([])
# Distinct Query instances per parameter: a shared instance shares its default
# list across fields, which leaks one parameter's values into the others.
_TAG_ANY_QUERY = Query([])
_TAG_WITHOUT_QUERY = Query([])
_FACET_QUERY = Query([])
_RANGE_QUERY = Query([], alias="range")


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
            CatalogService(project),
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

    # Record the project location for the recent-project list; private
    # bookkeeping must never break app startup.
    with suppress(Exception):
        context.catalog.private.record_location(context.project.root)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _dashboard_html()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/projects/recent")
    def list_recent_projects() -> dict[str, Any]:
        entries = UserPrivateStore.list_recent_projects(context.catalog.private.home)
        return {
            "projects": [
                {"project_id": project_id, "root": root, "last_seen": last_seen}
                for project_id, root, last_seen in entries
            ]
        }

    @app.get("/project")
    def get_project() -> dict[str, Any]:
        project = context.project
        annotations = context.catalog.project_annotations()
        return {
            "schema": project.manifest.schema_,
            "project_id": project.project_id,
            "name": project.manifest.name,
            "root": str(project.root),
            "alias": annotations.alias,
            "note": annotations.note,
            "paths": {
                "workflows": str(project.workflow_root),
                "defaults": str(project.defaults_path),
                "plugins": [str(path) for path in project.plugin_dirs],
                "sessions": str(project.session_root),
            },
        }

    @app.patch("/project")
    def update_project(request: ProjectUpdateRequest) -> dict[str, Any]:
        fields = request.model_dump(exclude_unset=True)
        fields.pop("private", None)
        if not fields:
            raise HTTPException(status_code=400, detail="no fields to update")
        try:
            if request.private:
                object_id = context.project.project_id
                if "alias" in fields:
                    context.catalog.private.set_private_alias(
                        "project", object_id, fields["alias"]
                    )
                if "note" in fields:
                    context.catalog.private.set_private_note(
                        "project", object_id, fields["note"]
                    )
            else:
                if "alias" in fields:
                    context.catalog.set_project_alias(fields["alias"])
                if "note" in fields:
                    context.catalog.set_project_note(fields["note"])
        except RuntimeError as exc:
            raise _http_error(exc, status_code=409) from exc
        except Exception as exc:
            raise _http_error(exc) from exc
        return get_project()

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
                request.workflow,
                resume_from=request.resume_from,
                tags=request.tags,
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

    @app.put("/executions/{execution_id}/tags")
    def update_execution_tags(
        execution_id: str, request: ExecutionTagsRequest
    ) -> dict[str, Any]:
        try:
            return context.executions.update_submission_tags(
                execution_id, request.tags
            ).model_dump(mode="json")
        except ValueError as exc:
            raise _http_error(exc, status_code=409) from exc
        except Exception as exc:
            raise _http_error(exc) from exc

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
            if "lifecycle" in fields:
                context.catalog.set_session_lifecycle(
                    session_id, fields.pop("lifecycle")
                )
            if "retention" in fields:
                context.catalog.set_session_retention(
                    session_id, fields.pop("retention")
                )
            summary = (
                context.project_service.update_session(session_id, **fields)
                if fields
                else context.project_service.get_session(session_id)
            )
            return summary.model_dump(mode="json")
        except RuntimeError as exc:
            raise _http_error(exc, status_code=409) from exc
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
        try:
            artifacts = scheduler.list_artifacts(root)
        except FileNotFoundError as exc:
            raise _http_error(exc, status_code=404) from exc
        except ArtifactError as exc:
            raise _http_error(exc, status_code=422) from exc
        return {
            "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts]
        }

    @app.get("/sessions/{session_id}/files/{file_ref:path}")
    def get_session_file(session_id: str, file_ref: str) -> dict[str, Any]:
        root = context.project_service.session_dir(session_id)
        try:
            return scheduler.load_file_by_ref(file_ref, session_dir=root)
        except FileNotFoundError as exc:
            raise _http_error(exc, status_code=404) from exc
        except QPhaseIOError as exc:
            raise _http_error(exc, status_code=422) from exc

    @app.get("/sessions/{session_id}/artifacts/{artifact_id}")
    def get_session_artifact(session_id: str, artifact_id: str) -> dict[str, Any]:
        root = context.project_service.session_dir(session_id)
        try:
            return scheduler.describe_artifact_by_id(
                artifact_id, session_dir=root
            ).model_dump(mode="json")
        except FileNotFoundError as exc:
            raise _http_error(exc, status_code=404) from exc
        except ArtifactError as exc:
            raise _http_error(exc, status_code=422) from exc

    @app.get("/sessions/{session_id}/jobs/{job_name}/products")
    def get_job_products(session_id: str, job_name: str) -> dict[str, Any]:
        root = context.project_service.session_dir(session_id)
        try:
            catalog = scheduler.describe_products(job_name, session_dir=root)
        except FileNotFoundError as exc:
            raise _http_error(exc, status_code=404) from exc
        except ArtifactError as exc:
            # Unsupported/corrupt manifests and descriptor failures are
            # unprocessable, not missing.
            raise _http_error(exc, status_code=422) from exc
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

    @app.get("/catalog/issues")
    def list_location_issues() -> dict[str, Any]:
        try:
            return {"issues": context.catalog.location_issues()}
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/catalog/{kind}")
    def list_catalog_objects(
        kind: str,
        tag: list[str] = _TAGS_QUERY,
        tag_any: list[str] = _TAG_ANY_QUERY,
        tag_without: list[str] = _TAG_WITHOUT_QUERY,
        tag_descendant: str | None = None,
        tag_namespace: str | None = None,
        facet: list[str] = _FACET_QUERY,
        range_: list[str] = _RANGE_QUERY,
        lifecycle: str | None = None,
        retention: str | None = None,
        direct: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        try:
            objects = context.catalog.query(
                CatalogQuery(
                    object_kind=kind,
                    facets=parse_facet_filters(facet),
                    ranges=parse_range_filters(range_),
                    tags_all=tuple(tag),
                    tags_any=tuple(tag_any),
                    tags_without=tuple(tag_without),
                    tag_descendant_of=tag_descendant,
                    tag_namespace=tag_namespace,
                    lifecycle=lifecycle,
                    retention=retention,
                    effective=not direct,
                    limit=limit,
                    offset=offset,
                )
            )
        except Exception as exc:
            raise _http_error(exc) from exc
        return {"objects": [item.model_dump(mode="json") for item in objects]}

    @app.get("/catalog/{kind}/{object_id:path}/tags")
    def get_catalog_tags(kind: str, object_id: str) -> dict[str, Any]:
        try:
            tags = context.catalog.effective_tags(kind, object_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        return {"effective_tags": [tag.model_dump(mode="json") for tag in tags]}

    @app.post("/catalog/{kind}/{object_id:path}/tags")
    def update_catalog_tags(
        kind: str, object_id: str, request: TagsUpdateRequest
    ) -> dict[str, Any]:
        """Add or remove tags on any catalog object kind."""
        try:
            return _update_catalog_tags(context.catalog, kind, object_id, request)
        except RuntimeError as exc:
            raise _http_error(exc, status_code=409) from exc
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.patch("/catalog/{kind}/{object_id:path}/private")
    def update_private_annotation(
        kind: str, object_id: str, request: PrivateAnnotationRequest
    ) -> dict[str, Any]:
        """Set the user-private alias/note of one catalog object."""
        fields = request.model_dump(exclude_unset=True)
        if not fields:
            raise HTTPException(status_code=400, detail="no fields to update")
        if kind not in OBJECT_KINDS:
            raise HTTPException(
                status_code=404, detail=f"unknown catalog object kind {kind!r}"
            )
        if "alias" in fields:
            context.catalog.private.set_private_alias(kind, object_id, fields["alias"])
        if "note" in fields:
            context.catalog.private.set_private_note(kind, object_id, fields["note"])
        alias, note = context.catalog.private.get_private_annotation(kind, object_id)
        return {
            "object_kind": kind,
            "object_id": object_id,
            "alias": alias,
            "note": note,
        }

    @app.post("/sessions/{session_id}/tags")
    def update_session_tags(
        session_id: str, request: TagsUpdateRequest
    ) -> dict[str, Any]:
        try:
            return context.catalog.tag_session(
                session_id,
                add=request.add,
                remove=request.remove,
                private=request.private,
            ).model_dump(mode="json")
        except RuntimeError as exc:
            raise _http_error(exc, status_code=409) from exc
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/artifacts/{artifact_id}/tags")
    def update_artifact_tags(
        artifact_id: str, request: TagsUpdateRequest
    ) -> dict[str, Any]:
        try:
            tags = context.catalog.tag_artifact(
                artifact_id,
                add=request.add,
                remove=request.remove,
                private=request.private,
            )
        except RuntimeError as exc:
            raise _http_error(exc, status_code=409) from exc
        except Exception as exc:
            raise _http_error(exc) from exc
        return {"effective_tags": [tag.model_dump(mode="json") for tag in tags]}

    @app.post("/sessions/{session_id}/occurrences/{artifact_id}/tags")
    def update_occurrence_tags(
        session_id: str,
        artifact_id: str,
        request: TagsUpdateRequest,
        job_name: str | None = None,
    ) -> dict[str, Any]:
        try:
            tags = context.catalog.tag_occurrence(
                session_id,
                artifact_id,
                job_name=job_name,
                add=request.add,
                remove=request.remove,
                private=request.private,
            )
        except RuntimeError as exc:
            raise _http_error(exc, status_code=409) from exc
        except Exception as exc:
            raise _http_error(exc) from exc
        return {"effective_tags": [tag.model_dump(mode="json") for tag in tags]}

    @app.patch("/artifacts/{artifact_id}")
    def update_artifact(
        artifact_id: str, request: ArtifactUpdateRequest
    ) -> dict[str, Any]:
        fields = request.model_dump(exclude_unset=True)
        if not fields:
            raise HTTPException(status_code=400, detail="no fields to update")
        try:
            result = None
            if "lifecycle" in fields:
                result = context.catalog.set_artifact_lifecycle(
                    artifact_id, fields["lifecycle"]
                )
            if "retention" in fields:
                result = context.catalog.set_artifact_retention(
                    artifact_id, fields["retention"]
                )
            assert result is not None
            return result.model_dump(mode="json")
        except RuntimeError as exc:
            raise _http_error(exc, status_code=409) from exc
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.patch("/sessions/{session_id}/occurrences/{artifact_id}")
    def update_occurrence(
        session_id: str,
        artifact_id: str,
        request: OccurrenceUpdateRequest,
        job_name: str | None = None,
    ) -> dict[str, Any]:
        fields = request.model_dump(exclude_unset=True)
        if "retention" not in fields:
            raise HTTPException(status_code=400, detail="no fields to update")
        try:
            return context.catalog.set_occurrence_retention(
                session_id, artifact_id, fields["retention"], job_name=job_name
            ).model_dump(mode="json")
        except RuntimeError as exc:
            raise _http_error(exc, status_code=409) from exc
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/project/reindex")
    def reindex_catalog() -> dict[str, Any]:
        try:
            return asdict(context.catalog.reindex())
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/tags/policy")
    def get_tag_policy() -> dict[str, Any]:
        try:
            return context.catalog.tag_policy().model_dump(mode="json")
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/views")
    def list_saved_views() -> dict[str, Any]:
        try:
            views = context.catalog.list_views()
        except Exception as exc:
            raise _http_error(exc) from exc
        return {
            "views": [
                {"name": name, "query": asdict(query)} for name, query in views
            ]
        }

    @app.put("/views/{name}")
    def save_view(name: str, request: SavedViewRequest) -> dict[str, str]:
        try:
            context.catalog.save_view(
                name,
                CatalogQuery(
                    object_kind=request.object_kind,
                    tags_all=tuple(request.tags_all),
                    lifecycle=request.lifecycle,
                    retention=request.retention,
                    limit=request.limit,
                ),
            )
        except Exception as exc:
            raise _http_error(exc) from exc
        return {"status": "saved"}

    @app.delete("/views/{name}", status_code=204)
    def delete_saved_view(name: str) -> Response:
        try:
            context.catalog.delete_view(name)
        except Exception as exc:
            raise _http_error(exc) from exc
        return Response(status_code=204)

    @app.get("/folders")
    def list_virtual_folders() -> dict[str, Any]:
        try:
            folders = context.catalog.virtual_folders()
        except Exception as exc:
            raise _http_error(exc) from exc
        return {
            "folders": [
                {"name": name, "count": count} for name, count in folders
            ]
        }

    @app.get("/folders/{name}")
    def get_virtual_folder(name: str) -> dict[str, Any]:
        try:
            objects = context.catalog.virtual_folder(name)
        except KeyError as exc:
            raise _http_error(exc, status_code=404) from exc
        except Exception as exc:
            raise _http_error(exc) from exc
        return {"objects": [item.model_dump(mode="json") for item in objects]}

    @app.post("/catalog/{kind}/{object_id:path}/tags/promote")
    def promote_tag(
        kind: str, object_id: str, request: PromoteTagRequest
    ) -> dict[str, Any]:
        try:
            tags = context.catalog.promote_tag(kind, object_id, request.tag)
        except RuntimeError as exc:
            raise _http_error(exc, status_code=409) from exc
        except Exception as exc:
            raise _http_error(exc) from exc
        return {"effective_tags": [tag.model_dump(mode="json") for tag in tags]}

    return app


def _http_error(exc: Exception, status_code: int = 400) -> HTTPException:
    if isinstance(exc, QPhaseError):
        return HTTPException(status_code=status_code, detail=str(exc))
    return HTTPException(status_code=status_code, detail=str(exc))


def _update_catalog_tags(
    catalog: CatalogService, kind: str, object_id: str, request: TagsUpdateRequest
) -> dict[str, Any]:
    """Dispatch a tag mutation to the service method of the object kind."""
    add, remove, private = request.add, request.remove, request.private
    tags: Any
    if kind == "project":
        tags = catalog.tag_project(add=add, remove=remove, private=private)
    elif kind == "workflow":
        tags = catalog.tag_workflow(object_id, add=add, remove=remove, private=private)
    elif kind == "job":
        tags = catalog.tag_job(object_id, add=add, remove=remove, private=private)
    elif kind == "execution":
        tags = catalog.tag_execution(object_id, add=add, remove=remove, private=private)
    elif kind == "session":
        return catalog.tag_session(
            object_id, add=add, remove=remove, private=private
        ).model_dump(mode="json")
    elif kind == "artifact":
        tags = catalog.tag_artifact(object_id, add=add, remove=remove, private=private)
    elif kind == "occurrence":
        artifact_id, session_id, job_name = object_id.split(":", 2)
        tags = catalog.tag_occurrence(
            session_id,
            artifact_id,
            job_name=job_name,
            add=add,
            remove=remove,
            private=private,
        )
    else:
        raise ValueError(f"unknown catalog object kind {kind!r}")
    return {"effective_tags": [tag.model_dump(mode="json") for tag in tags]}


def _execution_action(action: Any, execution_id: str) -> dict[str, Any]:
    try:
        return action(execution_id).model_dump(mode="json")
    except Exception as exc:
        raise _http_error(exc, status_code=409) from exc


def _dashboard_html() -> str:
    return (
        resources.files("qphase.gui").joinpath("index.html").read_text(encoding="utf-8")
    )
