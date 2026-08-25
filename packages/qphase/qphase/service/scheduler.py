"""Project-aware scheduler service facade."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.config_loader import (
    get_config_for_job,
    merge_plugin_config_sections,
    registered_plugin_namespaces,
)
from qphase.core.execution import CancellationController
from qphase.core.project import ProjectContext
from qphase.core.registry import registry
from qphase.core.scheduler import JobResult, Scheduler
from qphase.core.system_config import SystemConfig, load_system_config
from qphase.core.workflow import WorkflowCatalog

from .models import (
    ArtifactProductCatalog,
    ArtifactSummary,
    AxisSummary,
    ConfigValidationIssue,
    ExecutionPlan,
    ExecutionPlanEdge,
    ExecutionPlanJob,
    ProductSummary,
    SessionHandle,
    VariableSummary,
)


class SchedulerService:
    """Structured API over workflow planning and synchronous execution."""

    def __init__(
        self,
        system_config: SystemConfig | None = None,
        project: ProjectContext | None = None,
    ) -> None:
        self.project = project or ProjectContext.discover()
        self.system_config = system_config or load_system_config()
        self.catalog = WorkflowCatalog(self.project)
        self.last_session_handle: SessionHandle | None = None

    def list_workflows(self) -> list[str]:
        return [item.id for item in self.catalog.list()]

    def load_workflow(self, reference: str | Path) -> WorkflowSpec:
        return self.catalog.load(reference)

    def build_plan(self, workflow: WorkflowSpec) -> ExecutionPlan:
        scheduler = Scheduler(system_config=self.system_config, project=self.project)
        issues: list[ConfigValidationIssue] = []
        try:
            scheduler._validate_jobs(workflow)
        except Exception as exc:
            issues.append(
                ConfigValidationIssue(path="jobs", message=str(exc), source="scheduler")
            )
        return ExecutionPlan(
            jobs=[self._plan_job(job) for job in workflow.jobs],
            edges=self._build_edges(workflow.jobs),
            validation_issues=issues,
        )

    def run(
        self,
        workflow: WorkflowSpec,
        progress_callback: Any = None,
        resume_from: str | Path | None = None,
        cancellation: CancellationController | None = None,
        before_job: Any = None,
        on_scheduler: Any = None,
    ) -> list[JobResult]:
        scheduler = Scheduler(
            system_config=self.system_config,
            project=self.project,
            on_progress=progress_callback,
            cancellation=cancellation,
            before_job=before_job,
        )
        if on_scheduler is not None:
            on_scheduler(scheduler)
        results = scheduler.run(
            workflow,
            resume_from=Path(resume_from) if resume_from is not None else None,
        )
        statuses = {result.status for result in results}
        self.last_session_handle = SessionHandle(
            session_id=scheduler.session_id,
            session_dir=scheduler.session_dir,
            status="failed"
            if "failed" in statuses
            else "partial"
            if "skipped_dependency" in statuses
            else "completed",
        )
        return results

    def dry_run(self, workflow: WorkflowSpec) -> ExecutionPlan:
        return self.build_plan(workflow)

    def list_artifacts(self, session_dir: str | Path) -> list[ArtifactSummary]:
        root = Path(session_dir)
        return [
            ArtifactSummary(
                artifact_id=hashlib.sha256(
                    str(path.relative_to(root)).encode()
                ).hexdigest()[:16],
                path=path,
                kind=self._artifact_kind(path),
                format=path.suffix.lstrip(".") or None,
                job_name=path.parent.name if path.parent != root else None,
                size=path.stat().st_size,
            )
            for path in root.rglob("*")
            if path.is_file()
        ]

    def load_artifact_by_id(
        self, artifact_id: str, *, session_dir: str | Path
    ) -> dict[str, Any]:
        matches = [
            item
            for item in self.list_artifacts(session_dir)
            if item.artifact_id == artifact_id
        ]
        if len(matches) != 1:
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")
        return self.load_artifact(matches[0].path, session_dir=session_dir)

    def load_artifact(
        self, path: str | Path, *, session_dir: str | Path
    ) -> dict[str, Any]:
        root = Path(session_dir).expanduser().resolve()
        requested = Path(path).expanduser()
        artifact = (
            requested.resolve()
            if requested.is_absolute()
            else (root / requested).resolve()
        )
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise FileNotFoundError(f"Artifact not found: {path}")
        payload: dict[str, Any] = {
            "path": str(artifact),
            "size": artifact.stat().st_size,
        }
        if artifact.suffix.lower() == ".json":
            payload.update(
                content=json.loads(artifact.read_text(encoding="utf-8")),
                content_type="application/json",
            )
        elif artifact.suffix.lower() in {".txt", ".log", ".csv"}:
            payload.update(
                content=artifact.read_text(encoding="utf-8"), content_type="text/plain"
            )
        else:
            payload.update(content=None, content_type="application/octet-stream")
        return payload

    def describe_products(
        self, path: str | Path, *, session_dir: str | Path
    ) -> ArtifactProductCatalog:
        """Build the metadata-only product catalog of a v3 artifact directory.

        Never materializes payloads: all fields come from the manifest and
        the dataset schemas, so this is safe for GUI listings.
        """
        root = Path(session_dir).expanduser().resolve()
        requested = Path(path).expanduser()
        artifact = (
            requested.resolve()
            if requested.is_absolute()
            else (root / requested).resolve()
        )
        if not artifact.is_relative_to(root) or not artifact.is_dir():
            raise FileNotFoundError(f"Artifact directory not found: {path}")
        from ..data.store import ArtifactManifestV3, load_products

        try:
            manifest = ArtifactManifestV3.read(artifact)
        except ValueError as exc:
            raise FileNotFoundError(
                f"Not a qphase 2.x artifact directory: {path} ({exc})"
            ) from exc
        datasets = load_products(artifact)
        size = sum(
            item.stat().st_size for item in artifact.rglob("*") if item.is_file()
        )
        products: list[ProductSummary] = []
        for entry in manifest.products:
            summary = datasets[entry.name].summary()
            products.append(
                ProductSummary(
                    name=entry.name,
                    kind=summary["kind"],
                    axes=[AxisSummary(**axis) for axis in summary["axes"]],
                    variables=[
                        VariableSummary(**variable)
                        for variable in summary["variables"]
                    ],
                    backing="artifact",
                    devices=summary["devices"],
                    materializable=True,
                    nbytes=summary["nbytes"],
                    chunk_count=sum(
                        variable.chunk_count
                        for variable in entry.storage.summary.values()
                    ),
                    sha256=entry.sha256,
                    attributes=datasets[entry.name].attributes,
                )
            )
        return ArtifactProductCatalog(
            artifact_id=manifest.artifact_id,
            path=artifact,
            loader="+".join(
                sorted({entry.storage.adapter for entry in manifest.products})
            ),
            products=products,
            size=size,
            content_hash=manifest.content_hash,
        )

    def load_session_manifest(self, session_dir: str | Path) -> dict[str, Any]:
        return json.loads(
            (Path(session_dir) / "session_manifest.json").read_text(encoding="utf-8")
        )

    def _plan_job(self, job: JobConfig) -> ExecutionPlanJob:
        manifest = self._engine_manifest(job.get_engine_name())
        explicit = self._explicit_plugin_namespaces(job)
        inherited = self._inherited_defaults(
            job, manifest["required_plugins"], explicit
        )
        return ExecutionPlanJob(
            name=job.name,
            engine=job.get_engine_name(),
            plugins=job.plugins,
            required_plugins=manifest["required_plugins"],
            optional_plugins=manifest["optional_plugins"],
            explicit_plugins=sorted(explicit),
            inherited_project_defaults=inherited,
            optional_plugins_enabled=sorted(
                set(manifest["optional_plugins"]) & explicit
            ),
            scan_summary=job.scan.compile().summary() if job.scan else None,
            input=job.input.from_ if job.input else None,
            output=job.output,
            save=job.save,
            expected_job_subdir=job.name,
            expected_output_name=self._expected_output_name(job),
            configured_plugin_paths=[
                f"{ns}.{name}"
                for ns, entries in job.plugins.items()
                for name in entries
            ],
            reusable_output=job.save is not False,
        )

    @staticmethod
    def _build_edges(jobs: list[JobConfig]) -> list[ExecutionPlanEdge]:
        names = {job.name for job in jobs}
        edges: list[ExecutionPlanEdge] = []
        for job in jobs:
            if job.input:
                edges.append(
                    ExecutionPlanEdge(
                        source=job.input.from_,
                        target=job.name,
                        kind="input",
                        input_mode=job.input.mode,
                    )
                )
            edges.extend(
                ExecutionPlanEdge(source=dep, target=job.name, kind="depends_on")
                for dep in job.depends_on
            )
            if job.output and job.output in names:
                edges.append(
                    ExecutionPlanEdge(source=job.name, target=job.output, kind="output")
                )
        return edges

    @staticmethod
    def _engine_manifest(engine_name: str) -> dict[str, list[str]]:
        try:
            cls = registry.get_plugin_class("engine", engine_name)
            manifest = getattr(cls, "manifest", None)
            return {
                "required_plugins": sorted(manifest.required_plugins)
                if manifest
                else [],
                "optional_plugins": sorted(manifest.optional_plugins)
                if manifest
                else [],
            }
        except Exception:
            return {"required_plugins": [], "optional_plugins": []}

    @staticmethod
    def _explicit_plugin_namespaces(job: JobConfig) -> set[str]:
        explicit = set(job.plugins)
        extra = job.model_extra or {}
        explicit.update(key for key in registered_plugin_namespaces() if key in extra)
        return explicit

    def _inherited_defaults(
        self, job: JobConfig, required: list[str], explicit: set[str]
    ) -> dict[str, list[str]]:
        merged = get_config_for_job(
            self.project,
            job_config_dict={
                "plugins": job.plugins,
                "engine": job.engine,
                "params": job.params,
            },
        )
        plugins = merge_plugin_config_sections(merged)
        return {
            namespace: sorted(plugins[namespace])
            for namespace in required
            if namespace not in explicit
            and isinstance(plugins.get(namespace), dict)
            and plugins[namespace]
        }

    @staticmethod
    def _expected_output_name(job: JobConfig) -> str | None:
        if isinstance(job.save, str):
            return job.save
        if job.output:
            return job.output
        return None if job.save is False else job.name

    @staticmethod
    def _artifact_kind(
        path: Path,
    ) -> Literal["result", "figure", "table", "manifest", "log", "other"]:
        if path.name == "session_manifest.json":
            return "manifest"
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
            return "figure"
        if path.suffix.lower() in {".csv", ".parquet"}:
            return "table"
        if path.suffix.lower() in {".npz", ".npy", ".json"}:
            return "result"
        if path.suffix.lower() in {".log", ".txt"}:
            return "log"
        return "other"
