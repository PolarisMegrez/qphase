"""Structured service-layer return models."""

from __future__ import annotations

from datetime import datetime
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


class SubpluginOptionSummary(ServiceModel):
    name: str
    path: str
    plugin: PluginSummary


class SubpluginSlotSummary(ServiceModel):
    name: str
    namespace: str
    cardinality: Literal["one", "optional", "many"]
    default: str | None = None
    description: str = ""
    options: list[SubpluginOptionSummary] = Field(default_factory=list)


class PluginTreeNode(ServiceModel):
    path: str
    plugin: PluginSummary
    slots: list[SubpluginSlotSummary] = Field(default_factory=list)


class ConfigSource(ServiceModel):
    kind: Literal["system", "project_defaults", "workflow", "merged", "snapshot"]
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
    engine: str
    plugins: dict[str, Any] = Field(default_factory=dict)
    required_plugins: list[str] = Field(default_factory=list)
    optional_plugins: list[str] = Field(default_factory=list)
    explicit_plugins: list[str] = Field(default_factory=list)
    inherited_project_defaults: dict[str, list[str]] = Field(default_factory=dict)
    optional_plugins_enabled: list[str] = Field(default_factory=list)
    scan_summary: dict[str, Any] | None = None
    input: str | None = None
    output: str | None = None
    save: bool | str | None = None
    expected_job_subdir: str | None = None
    expected_output_name: str | None = None
    configured_plugin_paths: list[str] = Field(default_factory=list)
    reusable_output: bool = False


class ExecutionPlanEdge(ServiceModel):
    source: str
    target: str
    kind: Literal["input", "output", "depends_on"]
    input_mode: Literal["dataset", "map"] | None = None


class ArtifactSummary(ServiceModel):
    """One session file reference.

    ``artifact_id`` is reserved for typed manifest artifacts and is absent for
    ordinary logs, reports and payload files. Those files are addressed by the
    project-relative ``file_ref`` instead.
    """

    artifact_id: str | None = None
    file_ref: str | None = None
    path: Path
    kind: Literal["result", "figure", "table", "manifest", "log", "other"] = "other"
    format: str | None = None
    job_name: str | None = None
    size: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AxisSummary(ServiceModel):
    """Read-only summary of one product axis (metadata only)."""

    name: str
    role: str
    size: int | None = None
    coordinate: str | None = None
    start: float | None = None
    step: float | None = None
    units: str = ""
    monotonic: bool = True


class VariableSummary(ServiceModel):
    """Read-only summary of one product variable (metadata only)."""

    name: str
    dtype: str
    value_domain: str
    dims: list[str] = Field(default_factory=list)
    quantity: str = ""
    units: str = ""
    constraints: dict[str, Any] = Field(default_factory=dict)


class CoordinateSummary(ServiceModel):
    """Read-only summary of one typed coordinate (metadata only)."""

    name: str
    variable: str
    dims: list[str] = Field(default_factory=list)
    role: str = "dimension"
    units: str = ""
    monotonic: bool = True


class SamplingBasisSummary(ServiceModel):
    """Read-only summary of one uncertainty sampling basis."""

    name: str
    source_axis: str | None = None
    count: int | None = None
    count_variable: str | None = None


class UncertaintySummary(ServiceModel):
    """Read-only summary of one declared uncertainty (metadata only)."""

    target: str
    kind: str
    sampling_basis: str = ""
    covariance: str | None = None
    scope: str | None = None
    data_variable: str | None = None
    confidence: float | None = None
    count: int | None = None


class BundleSummary(ServiceModel):
    """Read-only summary of the bundle descriptor of an artifact.

    Scan fields are unpacked from the descriptor's ``scan`` record when the
    owning resource package recorded one (e.g. ``sde.bundle/1``); generic
    bundles leave them ``None``.
    """

    type_id: str
    adapter_id: str
    descriptor_schema: str
    descriptor: dict[str, Any] = Field(default_factory=dict)
    product_roles: dict[str, str] = Field(default_factory=dict)
    scan_shape: list[int] | None = None
    scan_combine: bool | str | None = None
    scan_axes: dict[str, Any] | None = None
    n_traj_per_point: int | None = None


class ProductSummary(ServiceModel):
    """Read-only summary of one typed data product (metadata only).

    Building this DTO never materializes payloads and never registers
    artifact locations: every field comes from the artifact manifest
    (product schema, storage summary, descriptor) plus ``stat`` of the
    referenced payload files. ``materializable`` reports whether the
    product could be reopened in this process — its storage adapter is
    registered and every referenced payload file exists — with
    ``missing_reason`` naming the first blocker otherwise.
    """

    name: str
    kind: str
    axes: list[AxisSummary] = Field(default_factory=list)
    variables: list[VariableSummary] = Field(default_factory=list)
    coordinates: list[CoordinateSummary] = Field(default_factory=list)
    sampling_bases: list[SamplingBasisSummary] = Field(default_factory=list)
    uncertainties: list[UncertaintySummary] = Field(default_factory=list)
    backing: Literal["runtime", "artifact"] = "artifact"
    devices: list[str] = Field(default_factory=list)
    materializable: bool = True
    missing_reason: str | None = None
    nbytes: int | None = None
    physical_nbytes: int | None = None
    chunk_count: int | None = None
    schema_version: str = ""
    schema_fingerprint: str = ""
    storage_adapter: str = ""
    storage_descriptor_schema: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)


class ArtifactProductCatalog(ServiceModel):
    """Read-only catalog of the typed products of a v4 artifact directory.

    ``size`` counts only payload files referenced by the manifest; job
    logs, exports and stale chunks left by replaced writes are excluded.
    """

    artifact_id: str
    path: Path
    loader: str
    products: list[ProductSummary] = Field(default_factory=list)
    bundle: BundleSummary | None = None
    size: int = 0


class ExecutionPlan(ServiceModel):
    session_preview_id: str | None = None
    jobs: list[ExecutionPlanJob] = Field(default_factory=list)
    edges: list[ExecutionPlanEdge] = Field(default_factory=list)
    artifacts: list[ArtifactSummary] = Field(default_factory=list)
    validation_issues: list[ConfigValidationIssue] = Field(default_factory=list)


class SessionHandle(ServiceModel):
    session_id: str | None = None
    session_dir: Path | None = None
    status: str


ExecutionState = Literal[
    "queued",
    "running",
    "pause_requested",
    "paused",
    "completed",
    "partial",
    "failed",
    "cancelled",
]


class PluginActivity(ServiceModel):
    path: str
    status: Literal["configured", "active", "completed", "failed", "cancelled"]


class ExecutionJobState(ServiceModel):
    name: str
    engine: str
    status: str = "pending"
    stage: str | None = None
    fraction: float | None = None
    message: str = ""
    plugins: list[PluginActivity] = Field(default_factory=list)


class ExecutionSummary(ServiceModel):
    execution_id: str
    project_id: str
    workflow_id: str
    session_id: str | None = None
    state: ExecutionState
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    position: int | None = None
    jobs: list[ExecutionJobState] = Field(default_factory=list)
    current_job: str | None = None
    current_stage: str | None = None
    latest_message: str = ""
    submission_tags: list[str] = Field(default_factory=list)
    error: str | None = None


class ExecutionEvent(ServiceModel):
    sequence: int
    timestamp: datetime
    execution_id: str
    session_id: str | None = None
    payload: dict[str, Any]


class SessionSummary(ServiceModel):
    session_id: str
    project_id: str | None = None
    workflow_id: str | None = None
    status: str
    alias: str | None = None
    note: str | None = None
    start_time: datetime | None = None
    last_update: datetime | None = None
    jobs: dict[str, Any] = Field(default_factory=dict)


class WorkflowDocument(ServiceModel):
    doc_id: str
    workflow_id: str
    title: str
    path: Path
    writable: bool
    revision: str
    job_names: list[str] = Field(default_factory=list)
    content: str | None = None
