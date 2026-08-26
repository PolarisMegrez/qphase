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
without relying on process-global state, backed by the project object catalog
with a direct scan as the fresh-truth fallback.

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
from .errors import (
    ArtifactAmbiguousError,
    ArtifactCorruptError,
    ArtifactNotFoundError,
)

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

    Resolution is catalog-first: the project object catalog (a rebuildable
    index of session-relative occurrence paths) answers historical lookups
    without a filesystem scan. On a catalog miss — for example an artifact
    written seconds ago by a running session — the resolver falls back to a
    direct manifest scan, which remains the fresh truth.

    An ``artifact_id`` identifies the immutable artifact; every location is a
    producing *occurrence* (session/job context). The two concepts never
    merge: an artifact ref without occurrence context resolves only when
    exactly one location exists, otherwise the lookup reports
    :class:`ArtifactAmbiguousError` listing all locations instead of picking
    one arbitrarily.
    """

    def __init__(self, project: ProjectContext) -> None:
        self.project = project

    def resolve(self, ref: ArtifactRef) -> Path:
        """Return the single directory holding ``ref``'s artifact.

        Raises :class:`ArtifactNotFoundError` when no location exists and
        :class:`ArtifactAmbiguousError` when several do.
        """
        candidates = self.locations(ref.artifact_id)
        if not candidates:
            raise ArtifactNotFoundError(
                f"artifact {ref.artifact_id!r} is not present in this project"
            )
        if len(candidates) > 1:
            raise ArtifactAmbiguousError(
                f"artifact {ref.artifact_id!r} occurs in {len(candidates)} "
                f"locations: {[str(path) for path in candidates]}; resolve a "
                "specific occurrence (session/job) instead"
            )
        return candidates[0]

    def locations(self, artifact_id: str) -> list[Path]:
        """Return every existing directory holding ``artifact_id``, sorted."""
        located = self._catalog_locations(artifact_id)
        if located:
            return located
        return self._scan_locations(artifact_id)

    def _catalog_locations(self, artifact_id: str) -> list[Path]:
        from ..core.catalog import ProjectObjectCatalog

        catalog = ProjectObjectCatalog(self.project)
        if not catalog.path.exists():
            catalog.reindex()
        candidates = []
        for relative in catalog.locate_artifact_paths(artifact_id):
            candidate = (self.project.session_root / relative).resolve()
            # Stale index entries (artifact moved or deleted) miss to the scan.
            if candidate.exists():
                candidates.append(candidate)
        return candidates

    def _scan_locations(self, artifact_id: str) -> list[Path]:
        root = self.project.session_root
        candidates: list[Path] = []
        if root.exists():
            for manifest_path in sorted(root.rglob("artifact_manifest.json")):
                if ".trash" in manifest_path.parts:
                    continue
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
                if payload.get("artifact_id") == artifact_id:
                    candidates.append(manifest_path.parent.resolve())
        return candidates


_DEFAULT_RESOLVER = DirectoryArtifactResolver()


def default_artifact_resolver() -> DirectoryArtifactResolver:
    """Return the process-default artifact resolver."""
    return _DEFAULT_RESOLVER


def register_artifact_location(artifact_id: str, directory: Path | str) -> None:
    """Bind an artifact id to a directory on the default resolver."""
    _DEFAULT_RESOLVER.register(artifact_id, directory)
