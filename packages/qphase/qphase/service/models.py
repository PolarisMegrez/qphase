"""Structured service-layer return models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ServiceModel(BaseModel):
    """Base model for service-layer DTOs."""

    model_config = {"arbitrary_types_allowed": True}


class PluginSummary(ServiceModel):
    namespace: str
    name: str
    package: str | None = None
    description: str = ""
    schema_available: bool = False
    entry_point: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PluginCatalog(ServiceModel):
    packages: list[str] = Field(default_factory=list)
    namespaces: list[str] = Field(default_factory=list)
    plugins: list[PluginSummary] = Field(default_factory=list)


class ConfigSource(ServiceModel):
    kind: Literal["system", "global", "job", "merged", "snapshot"]
    path: Path | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ConfigValidationIssue(ServiceModel):
    level: Literal["error", "warning", "info"] = "error"
    path: str = ""
    message: str
    source: str | None = None


class MergedConfigPreview(ServiceModel):
    job_name: str
    raw_job_config: dict[str, Any]
    global_defaults_used: dict[str, Any] = Field(default_factory=dict)
    merged_config: dict[str, Any]
    validation_issues: list[ConfigValidationIssue] = Field(default_factory=list)


class ExecutionPlanJob(ServiceModel):
    name: str
    base_name: str
    index: int | None = None
    engine: str
    plugins: dict[str, Any] = Field(default_factory=dict)
    required_plugins: list[str] = Field(default_factory=list)
    optional_plugins: list[str] = Field(default_factory=list)
    explicit_plugins: list[str] = Field(default_factory=list)
    inherited_global_defaults: dict[str, list[str]] = Field(default_factory=dict)
    optional_plugins_enabled: list[str] = Field(default_factory=list)
    scan_params: dict[str, Any] = Field(default_factory=dict)
    input: str | None = None
    output: str | None = None
    save: bool | str | None = None
    expected_run_subdir: str | None = None
    expected_output_name: str | None = None


class ExecutionPlanEdge(ServiceModel):
    source: str
    target: str
    kind: Literal["input", "output", "aggregate", "depends_on"]


class ArtifactSummary(ServiceModel):
    path: Path
    kind: Literal["result", "figure", "table", "manifest", "log", "other"] = "other"
    format: str | None = None
    job_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(ServiceModel):
    session_preview_id: str | None = None
    original_jobs: list[ExecutionPlanJob] = Field(default_factory=list)
    expanded_jobs: list[ExecutionPlanJob] = Field(default_factory=list)
    edges: list[ExecutionPlanEdge] = Field(default_factory=list)
    scan_groups: dict[str, list[str]] = Field(default_factory=dict)
    artifacts: list[ArtifactSummary] = Field(default_factory=list)
    validation_issues: list[ConfigValidationIssue] = Field(default_factory=list)


class RunHandle(ServiceModel):
    session_id: str | None = None
    session_dir: Path | None = None
    status: str
