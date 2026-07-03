"""Configuration service facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qphase.core.config import JobConfig, JobList
from qphase.core.config_loader import (
    get_config_for_job,
    load_global_config,
    load_jobs_from_files,
    save_global_config,
)
from qphase.core.registry import RegistryCenter, registry
from qphase.core.system_config import SystemConfig, load_system_config

from .models import ConfigValidationIssue, MergedConfigPreview


class ConfigService:
    """Python-first API for loading, merging, and validating configuration."""

    def __init__(
        self,
        system_config: SystemConfig | None = None,
        registry_center: RegistryCenter | None = None,
    ):
        self.system_config = system_config or load_system_config()
        self.registry = registry_center or registry

    def load_system_config(self) -> SystemConfig:
        return self.system_config

    def load_global_config(self, path: str | Path | None = None) -> dict[str, Any]:
        global_path = Path(path or self.system_config.paths.global_file)
        return load_global_config(global_path)

    def save_global_config(
        self, data: dict[str, Any], path: str | Path | None = None
    ) -> None:
        global_path = Path(path or self.system_config.paths.global_file)
        save_global_config(data, global_path)

    def load_job_files(self, paths: list[str | Path]) -> JobList:
        return load_jobs_from_files([Path(path) for path in paths])

    def normalize_job_config(self, raw: dict[str, Any]) -> JobConfig:
        return JobConfig(**raw)

    def merge_for_job(
        self,
        job: JobConfig,
        system_config: SystemConfig | None = None,
    ) -> dict[str, Any]:
        system_cfg = system_config or job.system or self.system_config
        job_override = {
            "plugins": job.plugins,
            "engine": job.engine,
            "params": job.params,
        }
        return get_config_for_job(
            system_cfg, job_name=job.name, job_config_dict=job_override
        )

    def preview_merged_config(self, job: JobConfig) -> MergedConfigPreview:
        if job.system:
            global_defaults = self.load_global_config(job.system.paths.global_file)
        else:
            global_defaults = self.load_global_config()
        merged = self.merge_for_job(job)
        return MergedConfigPreview(
            job_name=job.name,
            raw_job_config=job.model_dump(mode="json"),
            global_defaults_used=global_defaults,
            merged_config=merged,
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
                engine_config = dict(job.engine.get(engine_name, {}))
                engine_config["name"] = engine_name
                self.registry.validate_plugin_config("engine", engine_config)
            except Exception as exc:
                issues.append(
                    ConfigValidationIssue(
                        path=f"engine.{engine_name}",
                        message=str(exc),
                        source="registry",
                    )
                )

        for namespace, plugin_configs in job.plugins.items():
            for plugin_name, plugin_config in plugin_configs.items():
                try:
                    config_data = dict(plugin_config)
                    config_data["name"] = plugin_name
                    self.registry.validate_plugin_config(namespace, config_data)
                except Exception as exc:
                    issues.append(
                        ConfigValidationIssue(
                            path=f"plugins.{namespace}.{plugin_name}",
                            message=str(exc),
                            source="registry",
                        )
                    )

        return issues
