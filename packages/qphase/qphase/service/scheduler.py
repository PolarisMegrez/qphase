"""Scheduler service facade."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from qphase.core.config import JobConfig, JobList
from qphase.core.config_loader import (
    get_config_for_job,
    list_available_jobs,
    load_jobs_from_files,
    merge_plugin_config_sections,
    registered_plugin_namespaces,
)
from qphase.core.registry import registry
from qphase.core.scheduler import JobResult, Scheduler
from qphase.core.system_config import SystemConfig, load_system_config

from .models import (
    ArtifactSummary,
    ConfigValidationIssue,
    ExecutionPlan,
    ExecutionPlanEdge,
    ExecutionPlanJob,
    RunHandle,
)


class SchedulerService:
    """Structured API over scheduler execution and planning."""

    def __init__(self, system_config: SystemConfig | None = None):
        self.system_config = system_config or load_system_config()
        self.last_run_handle: RunHandle | None = None

    def list_jobs(self) -> list[str]:
        return list_available_jobs(self.system_config)

    def load_jobs(self, names_or_paths: Sequence[str | Path]) -> JobList:
        paths = [self._resolve_job_path(item) for item in names_or_paths]
        return load_jobs_from_files(paths)

    def build_plan(
        self,
        job_list: JobList,
        system_config: SystemConfig | None = None,
    ) -> ExecutionPlan:
        scheduler = Scheduler(system_config=system_config or self.system_config)
        validation_issues: list[ConfigValidationIssue] = []

        try:
            scheduler._validate_jobs(job_list)
        except Exception as exc:
            validation_issues.append(
                ConfigValidationIssue(path="jobs", message=str(exc), source="scheduler")
            )

        return ExecutionPlan(
            jobs=[self._plan_job(job) for job in job_list.jobs],
            edges=self._build_edges(job_list.jobs),
            validation_issues=validation_issues,
        )

    def run(
        self,
        job_list: JobList,
        progress_callback: Any = None,
        resume_from: str | Path | None = None,
    ) -> list[JobResult]:
        scheduler = Scheduler(
            system_config=self.system_config,
            on_progress=progress_callback,
        )
        results = scheduler.run(
            job_list,
            resume_from=Path(resume_from) if resume_from is not None else None,
        )
        failed = any(result.status == "failed" for result in results)
        skipped = any(result.status == "skipped_dependency" for result in results)
        self.last_run_handle = RunHandle(
            session_id=scheduler.session_id,
            session_dir=scheduler.session_dir,
            status="failed" if failed else "partial" if skipped else "completed",
        )
        return results

    def dry_run(self, job_list: JobList) -> ExecutionPlan:
        return self.build_plan(job_list)

    def list_artifacts(self, session_dir: str | Path) -> list[ArtifactSummary]:
        root = Path(session_dir)
        artifacts = []
        for path in root.rglob("*"):
            if path.is_file():
                artifacts.append(
                    ArtifactSummary(
                        path=path,
                        kind=self._artifact_kind(path),
                        format=path.suffix.lstrip(".") or None,
                        job_name=path.parent.name if path.parent != root else None,
                    )
                )
        return artifacts

    def load_artifact(self, path: str | Path) -> dict[str, Any]:
        artifact_path = Path(path).expanduser().resolve()
        if not artifact_path.exists() or not artifact_path.is_file():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        summary = ArtifactSummary(
            path=artifact_path,
            kind=self._artifact_kind(artifact_path),
            format=artifact_path.suffix.lstrip(".") or None,
        )
        payload: dict[str, Any] = {"artifact": summary.model_dump(mode="json")}

        if artifact_path.suffix.lower() == ".json":
            with open(artifact_path, encoding="utf-8") as file:
                payload["content"] = json.load(file)
            payload["content_type"] = "application/json"
        elif artifact_path.suffix.lower() in {".txt", ".log", ".csv"}:
            payload["content"] = artifact_path.read_text(encoding="utf-8")
            payload["content_type"] = "text/plain"
        else:
            payload["content"] = None
            payload["content_type"] = "application/octet-stream"

        return payload

    def load_session_manifest(self, session_dir: str | Path) -> dict[str, Any]:
        manifest_path = Path(session_dir) / "session_manifest.json"
        with open(manifest_path, encoding="utf-8") as file:
            return json.load(file)

    def _resolve_job_path(self, name_or_path: str | Path) -> Path:
        path = Path(name_or_path)
        if path.exists():
            return path

        for config_dir in self.system_config.paths.config_dirs:
            jobs_dir = Path(config_dir) / "jobs"
            for suffix in (".yaml", ".yml"):
                candidate = jobs_dir / f"{name_or_path}{suffix}"
                if candidate.exists():
                    return candidate
        return path

    def _plan_job(self, job: JobConfig) -> ExecutionPlanJob:
        manifest = self._engine_manifest(job.get_engine_name())
        explicit_plugins = self._explicit_plugin_namespaces(job)
        inherited_defaults = self._inherited_global_defaults(
            job, manifest["required_plugins"], explicit_plugins
        )
        optional_enabled = sorted(
            namespace
            for namespace in manifest["optional_plugins"]
            if namespace in explicit_plugins
        )
        return ExecutionPlanJob(
            name=job.name,
            engine=job.get_engine_name(),
            plugins=job.plugins,
            required_plugins=manifest["required_plugins"],
            optional_plugins=manifest["optional_plugins"],
            explicit_plugins=sorted(explicit_plugins),
            inherited_global_defaults=inherited_defaults,
            optional_plugins_enabled=optional_enabled,
            scan_summary=job.scan.compile().summary() if job.scan else None,
            input=job.input.from_ if job.input else None,
            output=job.output,
            save=job.save,
            expected_run_subdir=job.name,
            expected_output_name=self._expected_output_name(job),
        )

    def _build_edges(self, jobs: list[JobConfig]) -> list[ExecutionPlanEdge]:
        job_names = {job.name for job in jobs}
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
            for dependency in job.depends_on:
                edges.append(
                    ExecutionPlanEdge(
                        source=dependency, target=job.name, kind="depends_on"
                    )
                )
            if job.output and job.output in job_names:
                edges.append(
                    ExecutionPlanEdge(source=job.name, target=job.output, kind="output")
                )
        return edges

    def _engine_manifest(self, engine_name: str) -> dict[str, list[str]]:
        try:
            engine_cls = registry.get_plugin_class("engine", engine_name)
        except Exception:
            return {"required_plugins": [], "optional_plugins": []}

        manifest = getattr(engine_cls, "manifest", None)
        if manifest is None:
            return {"required_plugins": [], "optional_plugins": []}
        return {
            "required_plugins": sorted(manifest.required_plugins),
            "optional_plugins": sorted(manifest.optional_plugins),
        }

    def _explicit_plugin_namespaces(self, job: JobConfig) -> set[str]:
        explicit = set(job.plugins.keys())
        job_extra = job.model_extra or {}
        explicit.update(
            key for key in registered_plugin_namespaces() if key in job_extra
        )
        return explicit

    def _inherited_global_defaults(
        self, job: JobConfig, required_plugins: list[str], explicit_plugins: set[str]
    ) -> dict[str, list[str]]:
        job_override = {
            "plugins": job.plugins,
            "engine": job.engine,
            "params": job.params,
        }
        merged_config = get_config_for_job(
            job.system or self.system_config,
            job_name=job.name,
            job_config_dict=job_override,
        )
        merged_plugins = self._merged_plugin_config(merged_config)
        inherited: dict[str, list[str]] = {}
        for namespace in required_plugins:
            if namespace in explicit_plugins:
                continue
            namespace_config = merged_plugins.get(namespace)
            if isinstance(namespace_config, dict) and namespace_config:
                inherited[namespace] = sorted(namespace_config.keys())
        return inherited

    def _merged_plugin_config(self, merged_config: dict[str, Any]) -> dict[str, Any]:
        return merge_plugin_config_sections(merged_config)

    def _expected_output_name(self, job: JobConfig) -> str | None:
        if isinstance(job.save, str):
            return job.save
        if job.output:
            return job.output
        if job.save is False:
            return None
        return job.name

    def _artifact_kind(
        self, path: Path
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
