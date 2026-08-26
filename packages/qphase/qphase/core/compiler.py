"""Compile a logical workflow before any engine or worker starts.

The compiler is deliberately limited to control-plane work: it resolves the
project defaults, engine declarations, plugin configuration and logical data
flow. It never instantiates an engine or backend and never opens a product
payload.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .config import JobConfig, WorkflowSpec
from .config_loader import (
    get_config_for_job,
    merge_plugin_config_sections,
    registered_plugin_namespaces,
)
from .errors import ErrorCode, QPhaseConfigError, QPhasePluginError
from .project import ProjectContext
from .registry import RegistryView, registry
from .scan import ParameterGrid
from .system_config import SystemConfig

__all__ = ["CompiledJob", "CompiledWorkflow", "WorkflowCompiler"]


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Freeze one compiler-owned mapping without copying payloads."""
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class CompiledJob:
    """Immutable control-plane description of one logical job."""

    job: JobConfig
    engine_name: str
    engine_config: Mapping[str, Any]
    merged_config: Mapping[str, Any]
    plugin_config: Mapping[str, Any]
    required_plugins: tuple[str, ...]
    optional_plugins: tuple[str, ...]
    explicit_plugins: tuple[str, ...]
    inherited_plugins: Mapping[str, tuple[str, ...]]
    input_source: str | None
    input_mode: str | None
    output: str | None
    save: bool | str | None
    depends_on: tuple[str, ...]
    parameter_grid: ParameterGrid | None

    @property
    def name(self) -> str:
        """Logical job name."""
        return self.job.name


@dataclass(frozen=True)
class CompiledWorkflow:
    """Resolved workflow representation shared by planning and execution."""

    workflow: WorkflowSpec
    project_id: str
    jobs: tuple[CompiledJob, ...]
    topological_order: tuple[str, ...]
    revision: str | None = None

    def job(self, name: str) -> CompiledJob:
        """Return one compiled job by logical name."""
        for item in self.jobs:
            if item.name == name:
                return item
        raise KeyError(f"unknown compiled job {name!r}")

    @property
    def logical_jobs(self) -> tuple[JobConfig, ...]:
        """Return jobs in declared workflow order for execution/reporting."""
        return tuple(item.job for item in self.jobs)


class WorkflowCompiler:
    """Compile workflows against one project, system config and registry view."""

    def __init__(
        self,
        project: ProjectContext,
        system_config: SystemConfig,
        registry_view: RegistryView | None = None,
    ) -> None:
        self.project = project
        self.system_config = system_config
        self.registry = registry_view or registry.view()

    def compile(
        self,
        workflow: WorkflowSpec,
        *,
        revision: str | None = None,
    ) -> CompiledWorkflow:
        """Resolve and validate a workflow without instantiating plugins."""
        self._validate_unique_names(workflow)
        compiled = tuple(self._compile_job(job, workflow) for job in workflow.jobs)
        order = self._topological_order(compiled)
        return CompiledWorkflow(
            workflow=workflow,
            project_id=self.project.project_id,
            jobs=compiled,
            topological_order=order,
            revision=revision,
        )

    def _compile_job(
        self, job: JobConfig, workflow: WorkflowSpec
    ) -> CompiledJob:
        engine_name = job.get_engine_name()
        if not engine_name:
            raise QPhaseConfigError(
                f"Job {job.name!r} is missing required 'engine' field",
                code=ErrorCode.CONFIG,
            )
        try:
            self.registry.get_plugin_class("engine", engine_name)
        except QPhasePluginError:
            raise
        except Exception as exc:
            raise QPhasePluginError(
                f"failed to resolve engine {engine_name!r}: {exc}",
                code=ErrorCode.PLUGIN_DISCOVERY,
                context={"engine": engine_name},
            ) from exc

        manifest = self.registry.get_plugin_manifest("engine", engine_name)
        input_plugins = set(getattr(manifest, "input_plugins", set()))
        required_plugins = set(getattr(manifest, "required_plugins", set()))
        required = (
            input_plugins
            if job.input is not None and input_plugins
            else required_plugins
        )
        optional = set(getattr(manifest, "optional_plugins", set()))
        merged = get_config_for_job(
            self.project,
            job_config_dict=self._job_override(job),
        )
        plugin_sections = merge_plugin_config_sections(merged)
        explicit = self._explicit_namespaces(job)
        selected: dict[str, Any] = {}
        for namespace, config in plugin_sections.items():
            if namespace in explicit or namespace in required:
                if namespace in explicit and isinstance(config, Mapping):
                    extra = job.model_extra or {}
                    allowed = set(job.plugins.get(namespace, {}))
                    override = extra.get(namespace)
                    if isinstance(override, Mapping):
                        allowed.update(override)
                    selected[namespace] = {
                        name: values
                        for name, values in config.items()
                        if name in allowed
                    }
                else:
                    selected[namespace] = config

        missing = sorted(
            namespace for namespace in required if not selected.get(namespace)
        )
        if missing:
            mode = " for input/analyze mode" if job.input and input_plugins else ""
            raise QPhaseConfigError(
                f"Job {job.name!r} uses engine {engine_name!r} but is missing "
                f"required plugins{mode}: {missing}",
                code=ErrorCode.CONFIG,
            )
        inherited = {
            namespace: tuple(sorted(config))
            for namespace, config in selected.items()
            if namespace in required and namespace not in explicit
            and isinstance(config, Mapping)
        }
        forbidden_explicit = sorted(
            namespace
            for namespace in explicit
            if namespace in getattr(manifest, "forbidden_plugins", set())
        )
        if forbidden_explicit:
            raise QPhaseConfigError(
                f"Job {job.name!r} configures forbidden plugins: {forbidden_explicit}",
                code=ErrorCode.CONFIG,
            )

        engine_config = self._engine_config(merged, engine_name)
        self._validate_plugin_sections(selected)
        parameter_grid = job.scan.compile() if job.scan is not None else None
        input_source = self._validate_input(job, workflow)
        return CompiledJob(
            job=job,
            engine_name=engine_name,
            engine_config=_freeze_mapping(engine_config),
            merged_config=_freeze_mapping(merged),
            plugin_config=_freeze_mapping(selected),
            required_plugins=tuple(sorted(required)),
            optional_plugins=tuple(sorted(optional)),
            explicit_plugins=tuple(sorted(explicit)),
            inherited_plugins=inherited,
            input_source=input_source,
            input_mode=job.input.mode if job.input else None,
            output=job.output,
            save=job.save,
            depends_on=tuple(job.depends_on),
            parameter_grid=parameter_grid,
        )

    @staticmethod
    def _job_override(job: JobConfig) -> dict[str, Any]:
        override: dict[str, Any] = {
            "plugins": job.plugins,
            "engine": job.engine,
            "params": job.params,
        }
        for namespace in registered_plugin_namespaces():
            if namespace in (job.model_extra or {}):
                override[namespace] = (job.model_extra or {})[namespace]
        return override

    @staticmethod
    def _explicit_namespaces(job: JobConfig) -> set[str]:
        namespaces = set(job.plugins)
        extra = job.model_extra or {}
        namespaces.update(
            key for key in registered_plugin_namespaces() if key in extra
        )
        return namespaces

    def _engine_config(
        self, merged: Mapping[str, Any], engine_name: str
    ) -> dict[str, Any]:
        engines = merged.get("engine", {})
        if not isinstance(engines, Mapping) or engine_name not in engines:
            raise QPhaseConfigError(
                f"compiled config has no engine section for {engine_name!r}",
                code=ErrorCode.CONFIG,
            )
        config = engines[engine_name]
        if not isinstance(config, Mapping):
            raise QPhaseConfigError(
                f"engine config for {engine_name!r} must be a mapping",
                code=ErrorCode.CONFIG,
            )
        schema = self.registry.get_plugin_schema("engine", engine_name)
        if schema is not None:
            schema.model_validate({"name": engine_name, **dict(config)})
        return dict(config)

    def _validate_plugin_sections(self, selected: Mapping[str, Any]) -> None:
        for namespace, config in selected.items():
            if not isinstance(config, Mapping):
                raise QPhaseConfigError(
                    f"plugin section {namespace!r} must be a mapping",
                    code=ErrorCode.CONFIG,
                )
            for name, values in config.items():
                if not isinstance(values, Mapping):
                    raise QPhaseConfigError(
                        f"plugin config for {namespace}:{name} must be a mapping",
                        code=ErrorCode.CONFIG,
                    )
                self.registry.validate_plugin_config(
                    namespace,
                    {"name": str(name), **dict(values)},
                )

    def _validate_input(self, job: JobConfig, workflow: WorkflowSpec) -> str | None:
        if job.input is None:
            return None
        source = job.input.from_
        names = {item.name for item in workflow.jobs}
        if source in names:
            if source == job.name:
                raise QPhaseConfigError(
                    f"Job {job.name!r} cannot consume its own output",
                    code=ErrorCode.INPUT,
                )
            return source
        same_engine = [
            item.name for item in workflow.jobs if item.get_engine_name() == source
        ]
        if len(same_engine) > 1:
            raise QPhaseConfigError(
                f"Job {job.name!r} input {source!r} is ambiguous: {same_engine}",
                code=ErrorCode.INPUT,
            )
        if same_engine:
            return same_engine[0]
        path = Path(source).expanduser()
        if not path.exists():
            raise QPhaseConfigError(
                f"Job {job.name!r} input {source!r} is neither a workflow job "
                "nor an existing external path",
                code=ErrorCode.INPUT,
            )
        return source

    @staticmethod
    def _validate_unique_names(workflow: WorkflowSpec) -> None:
        names = [job.name for job in workflow.jobs]
        if len(set(names)) != len(names):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise QPhaseConfigError(
                f"workflow {workflow.id!r} has duplicate job names: {duplicates}",
                code=ErrorCode.CONFIG,
            )

    @staticmethod
    def _topological_order(jobs: tuple[CompiledJob, ...]) -> tuple[str, ...]:
        names = {job.name for job in jobs}
        dependencies = {job.name: set(job.depends_on) for job in jobs}
        for job in jobs:
            if job.input_source is not None and job.input_source in names:
                dependencies[job.name].add(job.input_source)
            unknown = dependencies[job.name] - names
            if unknown:
                raise QPhaseConfigError(
                    f"Job {job.name!r} depends on unknown jobs: {sorted(unknown)}",
                    code=ErrorCode.INPUT,
                )
            if job.name in dependencies[job.name]:
                raise QPhaseConfigError(
                    f"Job {job.name!r} depends on itself",
                    code=ErrorCode.INPUT,
                )
        order: list[str] = []
        pending = {name: set(values) for name, values in dependencies.items()}
        while pending:
            ready = [name for name, values in pending.items() if not values]
            if not ready:
                raise QPhaseConfigError(
                    "workflow job dependencies contain a cycle",
                    code=ErrorCode.INPUT,
                )
            for name in ready:
                order.append(name)
                pending.pop(name)
            for values in pending.values():
                values.difference_update(ready)
        return tuple(order)
