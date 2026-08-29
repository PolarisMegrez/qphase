"""Project-aware scheduler service facade."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..data.store import BundleDescriptor

from qphase.core.compiler import CompiledJob, CompiledWorkflow, WorkflowCompiler
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.config_loader import (
    get_config_for_job,
    merge_plugin_config_sections,
)
from qphase.core.errors import ErrorCode, QPhaseIOError
from qphase.core.execution import CancellationController
from qphase.core.persistence import ProjectStateStore
from qphase.core.project import ProjectContext
from qphase.core.registry import DiscoveryService, RegistryCenter, registry
from qphase.core.scheduler import JobResult, Scheduler
from qphase.core.system_config import SystemConfig, load_system_config
from qphase.core.tags import (
    freeze_tag_rules,
    load_tag_policy,
    validate_declared_tags,
)
from qphase.core.workflow import WorkflowCatalog
from qphase.data.errors import ArtifactCorruptError

from .models import (
    ArtifactProductCatalog,
    ArtifactSummary,
    AxisSummary,
    BundleSummary,
    ConfigValidationIssue,
    CoordinateSummary,
    ExecutionPlan,
    ExecutionPlanEdge,
    ExecutionPlanJob,
    ProductSummary,
    SamplingBasisSummary,
    SessionHandle,
    UncertaintySummary,
    VariableSummary,
)


class SchedulerService:
    """Structured API over workflow planning and synchronous execution."""

    def __init__(
        self,
        system_config: SystemConfig | None = None,
        project: ProjectContext | None = None,
        registry_center: RegistryCenter | None = None,
    ) -> None:
        self.project = project or ProjectContext.discover()
        self.system_config = system_config or load_system_config()
        self.catalog = WorkflowCatalog(self.project)
        self.state_store = ProjectStateStore(self.project)
        if registry_center is None:
            self.registry = registry.snapshot(include_local=False)
            project_discovery = DiscoveryService(self.registry)
            project_discovery.discover_plugins()
            project_discovery.discover_local_plugins(self.project)
        else:
            self.registry = registry_center
        self.last_session_handle: SessionHandle | None = None

    def list_workflows(self) -> list[str]:
        return [item.id for item in self.catalog.list()]

    def load_workflow(self, reference: str | Path) -> WorkflowSpec:
        return self.catalog.load(reference)

    def compile_workflow(self, workflow: WorkflowSpec) -> CompiledWorkflow:
        """Compile one workflow for planning, submission, or execution."""
        return WorkflowCompiler(
            project=self.project,
            system_config=self.system_config,
            registry_view=self.registry.view(),
        ).compile(workflow)

    def build_plan(self, workflow: WorkflowSpec) -> ExecutionPlan:
        issues: list[ConfigValidationIssue] = []
        compiled = None
        try:
            compiled = self.compile_workflow(workflow)
        except Exception as exc:
            issues.append(
                ConfigValidationIssue(path="jobs", message=str(exc), source="compiler")
            )
        jobs = list(compiled.logical_jobs) if compiled is not None else workflow.jobs
        return ExecutionPlan(
            jobs=[
                self._plan_job(
                    job,
                    compiled.job(job.name) if compiled is not None else None,
                )
                for job in jobs
            ],
            edges=self._build_edges(jobs),
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
        compiled_workflow: CompiledWorkflow | None = None,
        submission_tags: list[str] | None = None,
        submission_tag_policy_revision: str | None = None,
        submission_tag_rules: dict[str, dict[str, Any]] | None = None,
        execution_id: str | None = None,
    ) -> list[JobResult]:
        scheduler = Scheduler(
            system_config=self.system_config,
            project=self.project,
            on_progress=progress_callback,
            cancellation=cancellation,
            before_job=before_job,
            state_store=self.state_store,
            registry_center=self.registry,
        )
        if on_scheduler is not None:
            on_scheduler(scheduler)
        run_kwargs: dict[str, Any] = {
            "resume_from": Path(resume_from) if resume_from is not None else None
        }
        if compiled_workflow is not None:
            run_kwargs["compiled_workflow"] = compiled_workflow
        if submission_tags is not None:
            policy = load_tag_policy(self.project)
            run_kwargs["submission_tags"] = validate_declared_tags(
                list(submission_tags), "execution", policy
            )
            run_kwargs["submission_tag_policy_revision"] = (
                submission_tag_policy_revision
                if submission_tag_policy_revision is not None
                else policy.revision
                if policy is not None
                else None
            )
            if submission_tag_rules is not None:
                run_kwargs["submission_tag_rules"] = submission_tag_rules
        # Every real run owns an execution record: queued runs bring their
        # existing identity, direct runs (CLI) open one here. Plan/dry-run
        # and resume never create records.
        execution_payload: dict[str, Any] | None = None
        if execution_id is None and resume_from is None:
            execution_payload = self._open_execution(workflow, run_kwargs)
            execution_id = str(execution_payload["execution_id"])
        if execution_id is not None:
            run_kwargs["execution_id"] = execution_id
        try:
            results = scheduler.run(workflow, **run_kwargs)
        except Exception as exc:
            if execution_payload is not None:
                self._close_execution(execution_payload, scheduler, "failed", str(exc))
            raise
        statuses = {result.status for result in results}
        if execution_payload is not None:
            self._close_execution(
                execution_payload,
                scheduler,
                "failed"
                if "failed" in statuses
                else "cancelled"
                if "cancelled" in statuses
                else "partial"
                if "skipped_dependency" in statuses
                else "completed",
            )
        self.last_session_handle = SessionHandle(
            session_id=scheduler.session_id,
            session_dir=scheduler.session_dir,
            status="failed"
            if "failed" in statuses
            else "cancelled"
            if "cancelled" in statuses
            else "partial"
            if "skipped_dependency" in statuses
            else "completed",
        )
        return results

    def _open_execution(
        self, workflow: WorkflowSpec, run_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Persist the initial execution record of a direct (non-queued) run.

        Shares the record payload construction with the queued path
        (:class:`~qphase.service.execution.ExecutionManager`) so both entry
        points persist the same ``qphase.execution/1`` shape. Late import:
        the execution service builds on this module.
        """
        from datetime import datetime

        from .execution import initial_execution_payload

        policy = load_tag_policy(self.project)
        tags = run_kwargs.get("submission_tags")
        if tags is None:
            tags = validate_declared_tags([], "execution", policy)
        policy_revision = run_kwargs.get("submission_tag_policy_revision")
        if policy_revision is None:
            policy_revision = policy.revision if policy is not None else None
        rules = run_kwargs.get("submission_tag_rules")
        if rules is None:
            rules = freeze_tag_rules(policy, tags)
        compiled = run_kwargs.get("compiled_workflow")
        if compiled is None:
            compiled = self.compile_workflow(workflow)
            run_kwargs["compiled_workflow"] = compiled
        payload = initial_execution_payload(
            execution_id=uuid.uuid4().hex[:12],
            workflow=workflow,
            source_workflow=workflow.id,
            submitted_at=datetime.now().astimezone(),
            compiled_workflow=compiled,
            submission_tags=list(tags),
            tag_policy_revision=policy_revision,
            submission_tag_rules=rules,
            state="running",
        )
        payload["started_at"] = datetime.now().astimezone().isoformat()
        self.state_store.save_execution(payload)
        return payload

    def _close_execution(
        self,
        payload: dict[str, Any],
        scheduler: Scheduler,
        state: str,
        error: str | None = None,
    ) -> None:
        """Persist the terminal state of a direct run's execution record."""
        from datetime import datetime

        session_dir = (
            Path(scheduler.session_dir)
            .resolve()
            .relative_to(self.project.session_root.resolve())
            .as_posix()
            if scheduler.session_dir is not None
            else None
        )
        payload.update(
            state=state,
            finished_at=datetime.now().astimezone().isoformat(),
            session_id=scheduler.session_id,
            session_dir=session_dir,
            error=error,
        )
        self.state_store.save_execution(payload)

    def dry_run(self, workflow: WorkflowSpec) -> ExecutionPlan:
        return self.build_plan(workflow)

    def list_artifacts(self, session_dir: str | Path) -> list[ArtifactSummary]:
        """List typed artifact directories and ordinary session files.

        A v4 artifact is one manifest-backed directory, not its manifest and
        payload files individually. Files outside those directories retain a
        project-relative ``file_ref`` and have no artifact identity.
        """
        root = Path(session_dir).expanduser().resolve()
        from ..data.errors import ArtifactCorruptError
        from ..data.store import (
            ARTIFACT_SCHEMA_VERSION,
            ArtifactManifest,
            storage_referenced_files,
        )

        def job_name(path: Path) -> str | None:
            relative = path.relative_to(root)
            return relative.parts[0] if relative.parts else None

        artifact_dirs: set[Path] = set()
        artifacts: list[ArtifactSummary] = []
        for manifest_path in sorted(root.rglob("artifact_manifest.json")):
            artifact_dir = manifest_path.parent.resolve()
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArtifactCorruptError(
                    f"failed to parse artifact manifest {manifest_path}: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise ArtifactCorruptError(
                    f"artifact manifest {manifest_path} must contain an object"
                )
            version = raw.get("schema_version")
            if version != ARTIFACT_SCHEMA_VERSION:
                if isinstance(version, str) and version.startswith("qphase.artifact/"):
                    ArtifactManifest.read(artifact_dir)
                continue
            manifest = ArtifactManifest.read(artifact_dir)
            size = sum(
                (artifact_dir / file).stat().st_size
                for entry in manifest.products
                for file in storage_referenced_files(entry)
            )
            artifact_dirs.add(artifact_dir)
            artifacts.append(
                ArtifactSummary(
                    artifact_id=manifest.artifact_id,
                    path=artifact_dir,
                    kind="result",
                    format=manifest.schema_version,
                    job_name=job_name(artifact_dir),
                    size=size,
                )
            )

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if any(resolved.is_relative_to(directory) for directory in artifact_dirs):
                continue
            artifacts.append(
                ArtifactSummary(
                    file_ref=path.relative_to(root).as_posix(),
                    path=path,
                    kind=self._artifact_kind(path),
                    format=path.suffix.lstrip(".") or None,
                    job_name=job_name(path.parent),
                    size=path.stat().st_size,
                )
            )
        return artifacts

    def describe_artifact_by_id(
        self, artifact_id: str, *, session_dir: str | Path
    ) -> ArtifactProductCatalog:
        """Describe one manifest-backed artifact by its persisted identity."""
        matches = [
            item
            for item in self.list_artifacts(session_dir)
            if item.artifact_id == artifact_id
        ]
        if not matches:
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")
        if len(matches) > 1:
            raise ArtifactCorruptError(
                f"artifact identity conflict for {artifact_id!r}: "
                f"{len(matches)} manifest-backed artifacts found"
            )
        return self.describe_products(matches[0].path, session_dir=session_dir)

    def load_file_by_ref(
        self, file_ref: str, *, session_dir: str | Path
    ) -> dict[str, Any]:
        """Load an ordinary session file through its relative reference."""
        return self.load_artifact(file_ref, session_dir=session_dir)

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
        try:
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
                    content=artifact.read_text(encoding="utf-8"),
                    content_type="text/plain",
                )
            else:
                payload.update(content=None, content_type="application/octet-stream")
            return payload
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QPhaseIOError(
                f"failed to read session file: {artifact}",
                code=ErrorCode.ARTIFACT_IO,
                context={"path": str(artifact)},
            ) from exc

    def describe_products(
        self, path: str | Path, *, session_dir: str | Path
    ) -> ArtifactProductCatalog:
        """Build the metadata-only product catalog of a v4 artifact directory.

        Never materializes payloads, never opens storage adapters and never
        registers artifact locations: every field comes from the manifest
        (product schemas, storage summaries, bundle descriptor) plus
        ``stat`` of the manifest-referenced payload files, so this is safe
        for GUI listings. Error states stay typed: ``FileNotFoundError``
        when the directory or manifest does not exist,
        :class:`~qphase.data.errors.ArtifactUnsupportedError` for other
        schema versions and :class:`~qphase.data.errors.ArtifactCorruptError`
        for parse/cross-field/hash failures.
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
        from ..data.errors import ArtifactNotFoundError
        from ..data.store import (
            ArtifactManifest,
            storage_adapter_available,
            storage_referenced_files,
        )

        try:
            manifest = ArtifactManifest.read(artifact)
        except ArtifactNotFoundError as exc:
            raise FileNotFoundError(
                f"Not a qphase 2.x artifact directory: {path}"
            ) from exc

        products: list[ProductSummary] = []
        size = 0
        for entry in manifest.products:
            schema = entry.product_schema
            materializable = True
            missing_reason: str | None = None
            physical_nbytes = 0
            if not storage_adapter_available(entry.storage.adapter):
                materializable = False
                missing_reason = (
                    f"storage adapter {entry.storage.adapter!r} is not "
                    "registered in this process"
                )
            else:
                for file in storage_referenced_files(entry):
                    try:
                        physical_nbytes += (artifact / file).stat().st_size
                    except OSError:
                        materializable = False
                        missing_reason = f"payload file {file!r} is missing"
                        break
            size += physical_nbytes
            products.append(
                ProductSummary(
                    name=entry.name,
                    kind=schema.kind.value,
                    axes=[
                        AxisSummary(
                            name=axis.name,
                            role=axis.role.value,
                            size=axis.size,
                            coordinate=axis.coordinate,
                            start=axis.start,
                            step=axis.step,
                            units=axis.units,
                        )
                        for axis in schema.axes
                    ],
                    variables=[
                        VariableSummary(
                            name=variable.name,
                            dtype=variable.dtype,
                            value_domain=variable.value_domain,
                            dims=list(variable.dims),
                            quantity=variable.quantity,
                            units=variable.units,
                            constraints=variable.constraints.model_dump(mode="json"),
                        )
                        for variable in schema.variables
                    ],
                    coordinates=[
                        CoordinateSummary(
                            name=coordinate.name,
                            variable=coordinate.variable,
                            dims=list(coordinate.dims),
                            role=coordinate.role,
                            units=coordinate.units,
                            monotonic=coordinate.monotonic,
                        )
                        for coordinate in schema.coordinates
                    ],
                    sampling_bases=[
                        SamplingBasisSummary(
                            name=basis.name,
                            source_axis=basis.source_axis,
                            count=basis.count,
                            count_variable=basis.count_variable,
                        )
                        for basis in schema.sampling_bases
                    ],
                    uncertainties=[
                        UncertaintySummary(
                            target=uncertainty.target,
                            kind=uncertainty.kind,
                            sampling_basis=uncertainty.sampling_basis,
                            covariance=uncertainty.covariance,
                            scope=uncertainty.scope,
                            data_variable=uncertainty.data_variable,
                            confidence=uncertainty.confidence,
                            count=uncertainty.count,
                        )
                        for uncertainty in schema.uncertainties
                    ],
                    backing="artifact",
                    # All current storage adapters materialize to the host.
                    devices=["cpu"],
                    materializable=materializable,
                    missing_reason=missing_reason,
                    nbytes=sum(
                        variable.nbytes for variable in entry.storage.summary.values()
                    ),
                    physical_nbytes=physical_nbytes,
                    chunk_count=sum(
                        variable.chunk_count
                        for variable in entry.storage.summary.values()
                    ),
                    schema_version=schema.schema_version,
                    schema_fingerprint=schema.fingerprint(),
                    storage_adapter=entry.storage.adapter,
                    storage_descriptor_schema=entry.storage.descriptor_schema,
                    attributes=dict(schema.attributes),
                )
            )
        return ArtifactProductCatalog(
            artifact_id=manifest.artifact_id,
            path=artifact,
            loader="+".join(
                sorted({entry.storage.adapter for entry in manifest.products})
            ),
            products=products,
            bundle=_bundle_summary(manifest.bundle),
            size=size,
        )

    def load_session_manifest(self, session_dir: str | Path) -> dict[str, Any]:
        from qphase.core.persistence import ProjectStateStore

        return ProjectStateStore(self.project).load_session_manifest(Path(session_dir))

    def _plan_job(
        self, job: JobConfig, compiled: CompiledJob | None = None
    ) -> ExecutionPlanJob:
        if compiled is not None:
            return ExecutionPlanJob(
                name=compiled.name,
                engine=compiled.engine_name,
                plugins=job.plugins,
                required_plugins=list(compiled.required_plugins),
                optional_plugins=list(compiled.optional_plugins),
                explicit_plugins=list(compiled.explicit_plugins),
                inherited_project_defaults={
                    namespace: list(names)
                    for namespace, names in compiled.inherited_plugins.items()
                },
                optional_plugins_enabled=[
                    namespace
                    for namespace in compiled.optional_plugins
                    if namespace in compiled.plugin_config
                ],
                scan_summary=(
                    compiled.parameter_grid.summary()
                    if compiled.parameter_grid is not None
                    else None
                ),
                input=compiled.input_source,
                output=compiled.output,
                save=compiled.save,
                expected_job_subdir=compiled.name,
                expected_output_name=self._expected_output_name(job),
                configured_plugin_paths=[
                    f"{namespace}.{name}"
                    for namespace, entries in compiled.plugin_config.items()
                    if isinstance(entries, dict)
                    for name in entries
                ],
                reusable_output=compiled.save is not False,
            )
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

    def _engine_manifest(self, engine_name: str) -> dict[str, list[str]]:
        try:
            cls = self.registry.get_plugin_class("engine", engine_name)
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

    def _explicit_plugin_namespaces(self, job: JobConfig) -> set[str]:
        explicit = set(job.plugins)
        extra = job.model_extra or {}
        explicit.update(
            key for key in self.registry.list(namespace=None) if key in extra
        )
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
        plugins = merge_plugin_config_sections(
            merged, namespaces=set(self.registry.list(namespace=None))
        )
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


def _bundle_summary(bundle: BundleDescriptor) -> BundleSummary:
    """Unpack a manifest bundle descriptor into a read-only summary."""
    scan_shape: list[int] | None = None
    scan_combine: bool | str | None = None
    scan_axes: dict[str, Any] | None = None
    n_traj_per_point: int | None = None
    scan = bundle.descriptor.get("scan")
    if isinstance(scan, dict):
        shape = scan.get("shape")
        if isinstance(shape, list):
            scan_shape = [int(extent) for extent in shape]
        combine = scan.get("combine")
        if isinstance(combine, bool | str):
            scan_combine = combine
        axes = scan.get("axes")
        if isinstance(axes, dict):
            scan_axes = dict(axes)
        n_traj = scan.get("n_traj_per_point")
        if n_traj is not None:
            n_traj_per_point = int(n_traj)
    return BundleSummary(
        type_id=bundle.type_id,
        adapter_id=bundle.adapter_id,
        descriptor_schema=bundle.descriptor_schema,
        descriptor=dict(bundle.descriptor),
        product_roles=dict(bundle.product_roles),
        scan_shape=scan_shape,
        scan_combine=scan_combine,
        scan_axes=scan_axes,
        n_traj_per_point=n_traj_per_point,
    )
