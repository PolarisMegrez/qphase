"""Scheduler service facade."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qphase.core.config import JobConfig, JobList
from qphase.core.config_loader import list_available_jobs, load_jobs_from_files
from qphase.core.scheduler import JobResult, Scheduler
from qphase.core.system_config import SystemConfig, load_system_config

from .models import (
    ArtifactSummary,
    ConfigValidationIssue,
    ExecutionPlan,
    ExecutionPlanEdge,
    ExecutionPlanJob,
)


class SchedulerService:
    """Structured API over scheduler execution and planning."""

    def __init__(self, system_config: SystemConfig | None = None):
        self.system_config = system_config or load_system_config()

    def list_jobs(self) -> list[str]:
        return list_available_jobs(self.system_config)

    def load_jobs(self, names_or_paths: list[str | Path]) -> JobList:
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

        try:
            expanded_jobs = scheduler._expand_parameter_scans(job_list)
        except Exception as exc:
            validation_issues.append(
                ConfigValidationIssue(
                    path="jobs.scan", message=str(exc), source="scheduler"
                )
            )
            expanded_jobs = job_list.jobs

        return ExecutionPlan(
            original_jobs=[self._plan_job(job) for job in job_list.jobs],
            expanded_jobs=[self._plan_job(job) for job in expanded_jobs],
            edges=self._build_edges(expanded_jobs),
            scan_groups=self._scan_groups(job_list.jobs, expanded_jobs),
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
        return scheduler.run(
            job_list,
            resume_from=Path(resume_from) if resume_from is not None else None,
        )

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
        return ExecutionPlanJob(
            name=job.name,
            base_name=job.name.rsplit("_", 1)[0] if "_" in job.name else job.name,
            engine=job.get_engine_name(),
            plugins=job.plugins,
            scan_params=self._scan_params(job),
            input=job.input,
            output=job.output,
            save=job.save,
            expected_run_subdir=job.name,
        )

    def _build_edges(self, jobs: list[JobConfig]) -> list[ExecutionPlanEdge]:
        job_names = {job.name for job in jobs}
        edges: list[ExecutionPlanEdge] = []
        for job in jobs:
            if job.input:
                edges.append(
                    ExecutionPlanEdge(source=job.input, target=job.name, kind="input")
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

    def _scan_groups(
        self, original_jobs: list[JobConfig], expanded_jobs: list[JobConfig]
    ) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        original_names = {job.name for job in original_jobs}
        for job in expanded_jobs:
            base_name = job.name.rsplit("_", 1)[0]
            if base_name in original_names and base_name != job.name:
                groups.setdefault(base_name, []).append(job.name)
        return groups

    def _scan_params(self, job: JobConfig) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for engine_name, engine_config in job.engine.items():
            for key, value in engine_config.items():
                if isinstance(value, list):
                    params[f"engine.{engine_name}.{key}"] = value
        for namespace, plugin_configs in job.plugins.items():
            for plugin_name, plugin_config in plugin_configs.items():
                for key, value in plugin_config.items():
                    if isinstance(value, list):
                        params[f"{namespace}.{plugin_name}.{key}"] = value
        return params

    def _artifact_kind(self, path: Path) -> str:
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
