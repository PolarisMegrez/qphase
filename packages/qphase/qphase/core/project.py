"""Project identity, portable paths, and discovery."""

from __future__ import annotations

import os
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import QPhaseConfigError

PROJECT_MANIFEST_NAME = "qphase.toml"
PROJECT_ENV_VAR = "QPHASE_PROJECT"


class ProjectPaths(BaseModel):
    """Portable paths resolved relative to a project root."""

    workflows: str = "configs/workflows"
    defaults: str = "configs/defaults.yaml"
    plugins: list[str] = Field(default_factory=lambda: ["models"])
    sessions: str = "runs"

    model_config = ConfigDict(extra="forbid")

    @field_validator("workflows", "defaults", "sessions")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        return _portable_relative_path(value)

    @field_validator("plugins")
    @classmethod
    def _validate_plugin_paths(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("at least one project plugin directory is required")
        return [_portable_relative_path(value) for value in values]


class ProjectManifest(BaseModel):
    """Contents of ``qphase.toml``."""

    schema_: Literal["qphase.project/2"] = Field(alias="schema")
    project_id: str = Field(min_length=3, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]+$")
    name: str = Field(min_length=1)
    paths: ProjectPaths = Field(default_factory=ProjectPaths)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


@dataclass(frozen=True)
class ProjectContext:
    """Resolved project manifest and its filesystem boundary."""

    root: Path
    manifest: ProjectManifest

    @classmethod
    def discover(
        cls,
        start: str | Path | None = None,
        *,
        project: str | Path | None = None,
        environ: dict[str, str] | None = None,
    ) -> ProjectContext:
        """Find a project from an explicit path, environment, or parent search."""
        environment = os.environ if environ is None else environ
        requested = project or environment.get(PROJECT_ENV_VAR)
        if requested is not None:
            candidate = Path(requested).expanduser().resolve()
            manifest_path = (
                candidate
                if candidate.name == PROJECT_MANIFEST_NAME
                else candidate / PROJECT_MANIFEST_NAME
            )
            return cls.load(manifest_path)

        current = Path(start or Path.cwd()).expanduser().resolve()
        if current.is_file():
            current = current.parent
        for candidate in (current, *current.parents):
            manifest_path = candidate / PROJECT_MANIFEST_NAME
            if manifest_path.exists():
                return cls.load(manifest_path)
        raise QPhaseConfigError(
            f"No {PROJECT_MANIFEST_NAME} found from {current}. "
            "Run 'qphase project init' or set QPHASE_PROJECT."
        )

    @classmethod
    def load(cls, manifest_path: str | Path) -> ProjectContext:
        path = Path(manifest_path).expanduser().resolve()
        if path.is_dir():
            path = path / PROJECT_MANIFEST_NAME
        if not path.exists():
            raise QPhaseConfigError(f"Project manifest not found: {path}")
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
            manifest = ProjectManifest.model_validate(payload)
        except Exception as exc:
            raise QPhaseConfigError(f"Invalid project manifest {path}: {exc}") from exc
        return cls(root=path.parent, manifest=manifest)

    @classmethod
    def create(
        cls, root: str | Path, *, name: str | None = None, force: bool = False
    ) -> ProjectContext:
        project_root = Path(root).expanduser().resolve()
        project_root.mkdir(parents=True, exist_ok=True)
        path = project_root / PROJECT_MANIFEST_NAME
        if path.exists() and not force:
            raise QPhaseConfigError(f"Project already exists: {path}")
        manifest = ProjectManifest.model_validate(
            {
                "schema": "qphase.project/2",
                "project_id": f"qp_{uuid.uuid4().hex}",
                "name": name or project_root.name,
                "paths": {},
            }
        )
        path.write_text(_dump_manifest(manifest), encoding="utf-8")
        context = cls(project_root, manifest)
        context.ensure_directories()
        return context

    @property
    def project_id(self) -> str:
        return self.manifest.project_id

    @property
    def workflow_root(self) -> Path:
        return self._resolve(self.manifest.paths.workflows)

    @property
    def defaults_path(self) -> Path:
        return self._resolve(self.manifest.paths.defaults)

    @property
    def plugin_dirs(self) -> list[Path]:
        return [self._resolve(path) for path in self.manifest.paths.plugins]

    @property
    def session_root(self) -> Path:
        return self._resolve(self.manifest.paths.sessions)

    def ensure_directories(self) -> None:
        self.workflow_root.mkdir(parents=True, exist_ok=True)
        self.defaults_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_root.mkdir(parents=True, exist_ok=True)
        for path in self.plugin_dirs:
            path.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise QPhaseConfigError(f"Project path escapes project root: {relative}")
        return path


def _portable_relative_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError("project paths must be non-empty relative paths without '..'")
    return path.as_posix()


def _dump_manifest(manifest: ProjectManifest) -> str:
    data: dict[str, Any] = manifest.model_dump(by_alias=True)
    paths = data["paths"]
    plugins = ", ".join(f'"{item}"' for item in paths["plugins"])
    return (
        f'schema = "{data["schema"]}"\n'
        f'project_id = "{data["project_id"]}"\n'
        f'name = "{data["name"].replace(chr(34), chr(92) + chr(34))}"\n\n'
        "[paths]\n"
        f'workflows = "{paths["workflows"]}"\n'
        f'defaults = "{paths["defaults"]}"\n'
        f"plugins = [{plugins}]\n"
        f'sessions = "{paths["sessions"]}"\n'
    )
