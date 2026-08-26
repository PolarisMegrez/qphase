"""qphase: Artifact Resolver Contracts
---------------------------------------------------------
An artifact ref carries identity only; turning it into a storage location is
the job of an *artifact resolver*. Resolvers are injected by the
project/session artifact store — datasets never resolve locations through a
module-level global, and an unbound ref refuses to materialize with a clear
error.

The process-default resolver holds explicit ``artifact_id -> directory``
bindings populated by ``save_products``/``load_products``.  The
:class:`ProjectArtifactResolver` resolves within one project's Session root
without relying on process-global state; the later catalog phase can replace
its direct scan with an indexed implementation.

Public API
----------
ArtifactResolverProtocol
    Location-resolution contract for artifact refs.
DirectoryArtifactResolver
    Process-local resolver over explicit bindings.
ProjectArtifactResolver
    Project-scoped resolver over manifest locations.
default_artifact_resolver
    The process-default resolver instance.
register_artifact_location
    Bind an artifact id to a directory on the default resolver.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..core.project import ProjectContext
from .artifact import ArtifactRef
from .errors import ArtifactCorruptError, ArtifactNotFoundError

__all__ = [
    "ArtifactResolverProtocol",
    "DirectoryArtifactResolver",
    "ProjectArtifactResolver",
    "default_artifact_resolver",
    "register_artifact_location",
]


@runtime_checkable
class ArtifactResolverProtocol(Protocol):
    """Location-resolution contract for artifact references."""

    def resolve(self, ref: ArtifactRef) -> Path:
        """Return the artifact directory backing ``ref``.

        Raises :class:`ArtifactNotFoundError` when the ref is not bound.
        """
        ...


class DirectoryArtifactResolver:
    """Process-local resolver over explicit artifact-id bindings."""

    def __init__(self) -> None:
        self._bindings: dict[str, Path] = {}

    def register(self, artifact_id: str, directory: Path | str) -> None:
        """Bind one artifact id to its on-disk directory."""
        if not artifact_id:
            raise ValueError("artifact_id must be non-empty")
        self._bindings[artifact_id] = Path(directory)

    def clear(self) -> None:
        """Drop all bindings (test/support use)."""
        self._bindings.clear()

    def resolve(self, ref: ArtifactRef) -> Path:
        """Return the bound directory for ``ref``'s artifact id."""
        try:
            return self._bindings[ref.artifact_id]
        except KeyError:
            raise ArtifactNotFoundError(
                f"artifact {ref.artifact_id!r} is not bound in this resolver; "
                "open the artifact directory through "
                "qphase.data.store.load_products or bind it explicitly first"
            ) from None


class ProjectArtifactResolver:
    """Resolve artifact identities within one project's session root.

    The resolver deliberately performs a direct manifest scan. Project-wide
    indexing belongs to the later catalog phase; this implementation provides
    the correct project boundary without introducing a second index.
    """

    def __init__(self, project: ProjectContext) -> None:
        self.project = project

    def resolve(self, ref: ArtifactRef) -> Path:
        """Return the unique artifact directory for ``ref`` in this project."""
        matches: list[Path] = []
        root = self.project.session_root
        if not root.exists():
            raise ArtifactNotFoundError(
                f"artifact {ref.artifact_id!r} is not present in this project"
            )
        for manifest_path in root.rglob("artifact_manifest.json"):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArtifactCorruptError(
                    f"failed to read artifact manifest {manifest_path}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ArtifactCorruptError(
                    f"artifact manifest {manifest_path} must contain an object"
                )
            if payload.get("artifact_id") == ref.artifact_id:
                matches.append(manifest_path.parent.resolve())
        if not matches:
            raise ArtifactNotFoundError(
                f"artifact {ref.artifact_id!r} is not present in this project"
            )
        if len(matches) > 1:
            raise ArtifactCorruptError(
                f"artifact identity conflict for {ref.artifact_id!r}: {matches}"
            )
        return matches[0]


_DEFAULT_RESOLVER = DirectoryArtifactResolver()


def default_artifact_resolver() -> DirectoryArtifactResolver:
    """Return the process-default artifact resolver."""
    return _DEFAULT_RESOLVER


def register_artifact_location(artifact_id: str, directory: Path | str) -> None:
    """Bind an artifact id to a directory on the default resolver."""
    _DEFAULT_RESOLVER.register(artifact_id, directory)
