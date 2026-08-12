"""Workflow loading, recursive discovery, and stable-ID resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import JobConfig, WorkflowSpec
from .errors import QPhaseConfigError, QPhaseIOError
from .project import ProjectContext
from .utils import deep_merge_dicts, load_yaml


@dataclass(frozen=True)
class WorkflowReference:
    """A workflow's stable identity and current project-relative location."""

    id: str
    title: str
    path: Path
    relative_path: str
    collection: str | None
    tags: tuple[str, ...]
    job_count: int


def load_workflow(path: str | Path) -> WorkflowSpec:
    """Load one strict ``qphase.workflow/2`` document."""
    workflow_path = Path(path)
    if not workflow_path.exists():
        raise QPhaseIOError(f"Workflow file not found: {workflow_path}")
    try:
        data = load_yaml(workflow_path)
    except Exception as exc:
        raise QPhaseConfigError(
            f"Failed to read workflow {workflow_path}: {exc}"
        ) from exc
    if not isinstance(data, dict) or data.get("schema") != "qphase.workflow/2":
        raise QPhaseConfigError(
            f"{workflow_path} is not a qphase.workflow/2 document. "
            "Migrate legacy job configuration files before running them."
        )
    raw_jobs = data.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise QPhaseConfigError(
            f"Workflow {workflow_path} must contain a non-empty jobs list"
        )
    payload = dict(data)
    payload["jobs"] = [_load_job(item, workflow_path) for item in raw_jobs]
    try:
        return WorkflowSpec.model_validate(payload)
    except Exception as exc:
        raise QPhaseConfigError(f"Invalid workflow {workflow_path}: {exc}") from exc


class WorkflowCatalog:
    """Recursively index workflows by stable ID within one project."""

    def __init__(self, project: ProjectContext) -> None:
        self.project = project

    def list(self) -> list[WorkflowReference]:
        root = self.project.workflow_root
        if not root.exists():
            return []
        entries: list[WorkflowReference] = []
        ids: dict[str, Path] = {}
        paths = sorted([*root.rglob("*.yaml"), *root.rglob("*.yml")])
        for path in paths:
            reference = self._inspect(path)
            previous = ids.get(reference.id)
            if previous is not None:
                raise QPhaseConfigError(
                    f"Duplicate workflow id {reference.id!r}: {previous} and {path}"
                )
            ids[reference.id] = path
            entries.append(reference)
        return sorted(entries, key=lambda item: item.id)

    def search(
        self,
        *,
        collection: str | None = None,
        tag: str | None = None,
        query: str | None = None,
    ) -> list[WorkflowReference]:
        """Filter catalog entries without changing stable identity resolution."""
        needle = query.casefold().strip() if query else None
        return [
            item
            for item in self.list()
            if (collection is None or item.collection == collection)
            and (tag is None or tag in item.tags)
            and (
                needle is None
                or needle in item.id.casefold()
                or needle in item.title.casefold()
                or needle in item.relative_path.casefold()
            )
        ]

    def resolve(self, reference: str | Path) -> WorkflowReference:
        value = str(reference)
        candidate = Path(value)
        is_path = (
            candidate.suffix.lower() in {".yaml", ".yml"}
            or "/" in value
            or "\\" in value
        )
        if is_path:
            path = (
                candidate
                if candidate.is_absolute()
                else self.project.workflow_root / candidate
            )
            path = path.resolve()
            if not path.is_relative_to(self.project.workflow_root):
                raise QPhaseConfigError("Workflow path escapes the current project")
            if not path.exists():
                raise QPhaseIOError(f"Workflow file not found: {path}")
            return self._reference(path, load_workflow(path))
        matches = [item for item in self.list() if item.id == value]
        if not matches:
            raise QPhaseIOError(f"Workflow not found in project: {value}")
        return matches[0]

    def load(self, reference: str | Path) -> WorkflowSpec:
        return load_workflow(self.resolve(reference).path)

    def _reference(self, path: Path, workflow: WorkflowSpec) -> WorkflowReference:
        return WorkflowReference(
            id=workflow.id,
            title=workflow.title,
            path=path,
            relative_path=path.relative_to(self.project.workflow_root).as_posix(),
            collection=workflow.collection,
            tags=tuple(workflow.tags),
            job_count=len(workflow.jobs),
        )

    def _inspect(self, path: Path) -> WorkflowReference:
        """Read catalog metadata without importing or validating plugins."""
        data = load_yaml(path)
        if not isinstance(data, dict) or data.get("schema") != "qphase.workflow/2":
            raise QPhaseConfigError(f"{path} is not a qphase.workflow/2 document")
        workflow_id = data.get("id")
        title = data.get("title")
        jobs = data.get("jobs")
        if not isinstance(workflow_id, str) or not workflow_id:
            raise QPhaseConfigError(f"Workflow {path} has no stable id")
        if not isinstance(title, str) or not title:
            raise QPhaseConfigError(f"Workflow {path} has no title")
        if not isinstance(jobs, list) or not jobs:
            raise QPhaseConfigError(f"Workflow {path} has no logical jobs")
        tags = data.get("tags", [])
        if not isinstance(tags, list) or not all(
            isinstance(item, str) for item in tags
        ):
            raise QPhaseConfigError(f"Workflow {path} tags must be a string list")
        collection = data.get("collection")
        if collection is not None and not isinstance(collection, str):
            raise QPhaseConfigError(f"Workflow {path} collection must be a string")
        return WorkflowReference(
            id=workflow_id,
            title=title,
            path=path,
            relative_path=path.relative_to(self.project.workflow_root).as_posix(),
            collection=collection,
            tags=tuple(tags),
            job_count=len(jobs),
        )


def _load_job(raw: object, path: Path) -> JobConfig:
    if not isinstance(raw, dict):
        raise QPhaseConfigError(f"Workflow job in {path} must be a mapping")
    job_data, plugin_data = _extract_plugin_fields(dict(raw))
    explicit = job_data.pop("plugins", {})
    if isinstance(explicit, dict):
        plugin_data = deep_merge_dicts(explicit, plugin_data)
    return JobConfig.model_validate({**job_data, "plugins": plugin_data})


def _extract_plugin_fields(
    data: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    from .registry import registry

    core_fields = {
        "name",
        "package",
        "input",
        "input_loader",
        "output",
        "system",
        "engine",
        "params",
        "tags",
        "depends_on",
        "scan",
        "save",
        "plugins",
    }
    try:
        namespaces = set(registry.list(namespace=None))
    except Exception:
        namespaces = set()
    namespaces.update(
        {
            "backend",
            "integrator",
            "model",
            "analyser",
            "analyzer",
            "observer",
            "cam_solver",
            "cam_postprocessor",
            "visualizer",
        }
    )
    jobs: dict[str, object] = {}
    plugins: dict[str, object] = {}
    for key, value in data.items():
        if key in core_fields:
            jobs[key] = value
        elif key in namespaces and isinstance(value, dict):
            plugins[key] = value
        else:
            jobs[key] = value
    return jobs, plugins
