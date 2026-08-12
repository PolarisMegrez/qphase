"""Project-aware configuration service facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.config_loader import (
    get_config_for_job,
    load_project_defaults,
    merge_plugin_config_sections,
    save_project_defaults,
)
from qphase.core.project import ProjectContext
from qphase.core.registry import RegistryCenter, registry
from qphase.core.system_config import SystemConfig, load_system_config
from qphase.core.workflow import WorkflowCatalog, load_workflow

from .models import ConfigValidationIssue, MergedConfigPreview


class ConfigService:
    """Load and validate workflows, project defaults, and machine policy."""

    def __init__(
        self,
        project: ProjectContext | None = None,
        system_config: SystemConfig | None = None,
        registry_center: RegistryCenter | None = None,
    ) -> None:
        self.project = project or ProjectContext.discover()
        self.system_config = system_config or load_system_config()
        self.registry = registry_center or registry

    def load_system_config(self) -> SystemConfig:
        return self.system_config

    def load_project_defaults(self, path: str | Path | None = None) -> dict[str, Any]:
        return load_project_defaults(Path(path) if path else self.project.defaults_path)

    def save_project_defaults(
        self, data: dict[str, Any], path: str | Path | None = None
    ) -> None:
        save_project_defaults(data, Path(path) if path else self.project.defaults_path)

    def load_workflow(self, reference: str | Path) -> WorkflowSpec:
        path = Path(reference)
        if path.exists():
            return load_workflow(path)
        return WorkflowCatalog(self.project).load(reference)

    def normalize_job_config(self, raw: dict[str, Any]) -> JobConfig:
        return JobConfig(**raw)

    def merge_for_job(self, job: JobConfig) -> dict[str, Any]:
        override = {"plugins": job.plugins, "engine": job.engine, "params": job.params}
        merged = get_config_for_job(self.project, job_config_dict=override)
        merged["plugins"] = merge_plugin_config_sections(merged)
        return merged

    def preview_merged_config(self, job: JobConfig) -> MergedConfigPreview:
        return MergedConfigPreview(
            job_name=job.name,
            raw_job_config=job.model_dump(mode="json"),
            global_defaults_used=self.load_project_defaults(),
            merged_config=self.merge_for_job(job),
            validation_issues=self.validate_against_registry(job),
        )

    def validate_against_registry(
        self, job_or_config: JobConfig | dict[str, Any]
    ) -> list[ConfigValidationIssue]:
        job = (
            job_or_config
            if isinstance(job_or_config, JobConfig)
            else self.normalize_job_config(job_or_config)
        )
        issues: list[ConfigValidationIssue] = []
        engine_name = job.get_engine_name()
        if engine_name:
            try:
                config = dict(job.engine.get(engine_name, {}))
                config["name"] = engine_name
                self.registry.validate_plugin_config("engine", config)
            except Exception as exc:
                issues.append(
                    ConfigValidationIssue(
                        path=f"engine.{engine_name}",
                        message=str(exc),
                        source="registry",
                    )
                )
        for namespace, configs in job.plugins.items():
            for plugin_name, plugin_config in configs.items():
                try:
                    data = dict(plugin_config)
                    data["name"] = plugin_name
                    self.registry.validate_plugin_config(namespace, data)
                except Exception as exc:
                    issues.append(
                        ConfigValidationIssue(
                            path=f"plugins.{namespace}.{plugin_name}",
                            message=str(exc),
                            source="registry",
                        )
                    )
        return issues
