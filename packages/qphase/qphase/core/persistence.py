"""Project-scoped persistence ports for execution control and Session events.

The implementation is intentionally one small file-backed store. Session
manifests, event journals, execution records and annotation documents have
different protocols, but they share the project filesystem and do not need
separate database wrappers.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

from .annotations import (
    ARTIFACT_ANNOTATIONS_FILENAME,
    PROJECT_ANNOTATIONS_FILENAME,
    SESSION_ANNOTATIONS_FILENAME,
)
from .errors import ErrorCode, QPhaseIOError
from .project import ProjectContext

__all__ = [
    "AnnotationStoreProtocol",
    "EventStoreProtocol",
    "ExecutionStoreProtocol",
    "ProjectStateStore",
    "SessionStoreProtocol",
]


class SessionStoreProtocol(Protocol):
    """Port for the durable Session manifest."""

    def save_session_manifest(
        self, session_dir: Path, manifest: Mapping[str, Any]
    ) -> None:
        """Atomically persist one Session manifest."""
        ...

    def load_session_manifest(self, session_dir: Path) -> dict[str, Any]:
        """Load one Session manifest."""
        ...


class EventStoreProtocol(Protocol):
    """Port for append-only execution events and cursor reads."""

    def append_events(
        self, session_dir: Path, events: Iterable[Mapping[str, Any]]
    ) -> None:
        """Append JSON-safe events to a Session journal."""
        ...

    def read_events(
        self, session_dir: Path, *, after: int = 0
    ) -> list[dict[str, Any]]:
        """Read events whose sequence is greater than ``after``."""
        ...


class ExecutionStoreProtocol(Protocol):
    """Port for durable logical execution records."""

    def save_execution(self, payload: Mapping[str, Any]) -> None:
        """Persist one execution control record."""
        ...

    def load_executions(self) -> list[dict[str, Any]]:
        """Load persisted execution control records."""
        ...

    def delete_execution(self, execution_id: str) -> None:
        """Delete one retained execution record."""
        ...


class AnnotationStoreProtocol(Protocol):
    """Port for mutable annotation documents with optimistic concurrency.

    ``expected_revision`` is the revision the caller based its edit on;
    ``None`` requires that no document exists yet. A mismatch raises
    ``RuntimeError("annotation revision conflict")``. The store stamps the
    written document with the next revision and the current timestamp.
    """

    def load_session_annotations(self, session_dir: Path) -> dict[str, Any] | None:
        """Load the session annotation document; ``None`` when absent."""
        ...

    def save_session_annotations(
        self,
        session_dir: Path,
        document: Mapping[str, Any],
        *,
        expected_revision: int | None,
    ) -> dict[str, Any]:
        """Atomically persist the session annotation document."""
        ...

    def load_artifact_annotations(self, artifact_dir: Path) -> dict[str, Any] | None:
        """Load the artifact annotation document; ``None`` when absent."""
        ...

    def save_artifact_annotations(
        self,
        artifact_dir: Path,
        document: Mapping[str, Any],
        *,
        expected_revision: int | None,
    ) -> dict[str, Any]:
        """Atomically persist the artifact annotation document."""
        ...

    def load_project_annotations(self) -> dict[str, Any] | None:
        """Load the project annotation document; ``None`` when absent."""
        ...

    def save_project_annotations(
        self,
        document: Mapping[str, Any],
        *,
        expected_revision: int | None,
    ) -> dict[str, Any]:
        """Atomically persist the project annotation document."""
        ...


class ProjectStateStore(
    SessionStoreProtocol,
    EventStoreProtocol,
    ExecutionStoreProtocol,
    AnnotationStoreProtocol,
):
    """Single project-scoped file implementation of the state ports."""

    def __init__(self, project: ProjectContext) -> None:
        self.project = project

    @property
    def execution_root(self) -> Path:
        """Project-local control records, separate from scientific artifacts."""
        return self.project.root / ".qphase" / "executions"

    def save_execution(self, payload: Mapping[str, Any]) -> None:
        """Atomically persist one JSON-safe execution record."""
        execution_id = payload.get("execution_id")
        if not isinstance(execution_id, str) or not execution_id:
            raise QPhaseIOError(
                "execution record requires a non-empty execution_id",
                code=ErrorCode.ARTIFACT_IO,
            )
        target = self.execution_root / f"{execution_id}.json"
        temporary = target.with_suffix(".tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(dict(payload), indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)
        except (OSError, TypeError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            raise QPhaseIOError(
                f"failed to save execution record: {target}",
                code=ErrorCode.ARTIFACT_IO,
                context={"path": str(target)},
            ) from exc

    def load_executions(self) -> list[dict[str, Any]]:
        """Load all project execution records in submission order."""
        if not self.execution_root.exists():
            return []
        result: list[dict[str, Any]] = []
        for path in sorted(self.execution_root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise QPhaseIOError(
                    f"failed to load execution record: {path}",
                    code=ErrorCode.ARTIFACT_IO,
                    context={"path": str(path)},
                ) from exc
            if not isinstance(payload, dict):
                raise QPhaseIOError(
                    f"execution record must contain an object: {path}",
                    code=ErrorCode.ARTIFACT_IO,
                    context={"path": str(path)},
                )
            result.append(dict(payload))
        result.sort(key=lambda item: str(item.get("submitted_at", "")))
        return result

    def delete_execution(self, execution_id: str) -> None:
        """Delete one project execution record if it exists."""
        try:
            (self.execution_root / f"{execution_id}.json").unlink(missing_ok=True)
        except OSError as exc:
            raise QPhaseIOError(
                f"failed to delete execution record: {execution_id}",
                code=ErrorCode.ARTIFACT_IO,
                context={"execution_id": execution_id},
            ) from exc

    def save_session_manifest(
        self, session_dir: Path, manifest: Mapping[str, Any]
    ) -> None:
        """Atomically write a strict JSON Session manifest."""
        target = self._session_file(session_dir, "session_manifest.json")
        temporary = target.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(dict(manifest), indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)
        except (OSError, TypeError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            raise QPhaseIOError(
                f"failed to save session manifest: {target}",
                code=ErrorCode.ARTIFACT_IO,
                context={"path": str(target)},
            ) from exc

    def load_session_manifest(self, session_dir: Path) -> dict[str, Any]:
        """Load and validate the JSON object stored as a Session manifest."""
        target = self._session_file(session_dir, "session_manifest.json")
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QPhaseIOError(
                f"failed to load session manifest: {target}",
                code=ErrorCode.ARTIFACT_IO,
                context={"path": str(target)},
            ) from exc
        if not isinstance(payload, dict):
            raise QPhaseIOError(
                f"session manifest must contain an object: {target}",
                code=ErrorCode.ARTIFACT_IO,
                context={"path": str(target)},
            )
        return dict(payload)

    def append_events(
        self, session_dir: Path, events: Iterable[Mapping[str, Any]]
    ) -> None:
        """Append strict JSON event records to the Session journal."""
        target = self._session_file(session_dir, "events.jsonl")
        records = list(events)
        if not records:
            return
        try:
            with target.open("a", encoding="utf-8") as handle:
                for event in records:
                    handle.write(
                        json.dumps(dict(event), allow_nan=False) + "\n"
                    )
        except (OSError, TypeError, ValueError) as exc:
            raise QPhaseIOError(
                f"failed to append session events: {target}",
                code=ErrorCode.ARTIFACT_IO,
                context={"path": str(target)},
            ) from exc

    def read_events(
        self, session_dir: Path, *, after: int = 0
    ) -> list[dict[str, Any]]:
        """Read valid event records after a sequence cursor."""
        target = self._session_file(session_dir, "events.jsonl")
        if not target.exists():
            return []
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise QPhaseIOError(
                f"failed to read session events: {target}",
                code=ErrorCode.ARTIFACT_IO,
                context={"path": str(target)},
            ) from exc
        result: list[dict[str, Any]] = []
        try:
            for line in lines:
                payload = json.loads(line)
                if (
                    isinstance(payload, dict)
                    and int(payload.get("sequence", 0)) > after
                ):
                    result.append(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise QPhaseIOError(
                f"invalid session event record: {target}",
                code=ErrorCode.ARTIFACT_IO,
                context={"path": str(target)},
            ) from exc
        return result

    def load_session_annotations(self, session_dir: Path) -> dict[str, Any] | None:
        return self._load_annotation_document(
            self._session_file(session_dir, SESSION_ANNOTATIONS_FILENAME)
        )

    def save_session_annotations(
        self,
        session_dir: Path,
        document: Mapping[str, Any],
        *,
        expected_revision: int | None,
    ) -> dict[str, Any]:
        stored = self._save_annotation_document(
            self._session_file(session_dir, SESSION_ANNOTATIONS_FILENAME),
            document,
            expected_revision=expected_revision,
        )
        self._append_annotation_event(session_dir, stored)
        return stored

    def load_artifact_annotations(self, artifact_dir: Path) -> dict[str, Any] | None:
        return self._load_annotation_document(self._artifact_file(artifact_dir))

    def save_artifact_annotations(
        self,
        artifact_dir: Path,
        document: Mapping[str, Any],
        *,
        expected_revision: int | None,
    ) -> dict[str, Any]:
        stored = self._save_annotation_document(
            self._artifact_file(artifact_dir),
            document,
            expected_revision=expected_revision,
        )
        # The artifact directory is one job directory below the session.
        self._append_annotation_event(Path(artifact_dir).parent, stored)
        return stored

    def load_project_annotations(self) -> dict[str, Any] | None:
        return self._load_annotation_document(
            self._project_file(PROJECT_ANNOTATIONS_FILENAME)
        )

    def save_project_annotations(
        self,
        document: Mapping[str, Any],
        *,
        expected_revision: int | None,
    ) -> dict[str, Any]:
        # The project document is not session-scoped truth, so saving it
        # journals no session annotation event.
        return self._save_annotation_document(
            self._project_file(PROJECT_ANNOTATIONS_FILENAME),
            document,
            expected_revision=expected_revision,
        )

    def _load_annotation_document(self, target: Path) -> dict[str, Any] | None:
        if not target.exists():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QPhaseIOError(
                f"failed to load annotation document: {target}",
                code=ErrorCode.ARTIFACT_IO,
                context={"path": str(target)},
            ) from exc
        if not isinstance(payload, dict):
            raise QPhaseIOError(
                f"annotation document must contain an object: {target}",
                code=ErrorCode.ARTIFACT_IO,
                context={"path": str(target)},
            )
        return dict(payload)

    def _save_annotation_document(
        self,
        target: Path,
        document: Mapping[str, Any],
        *,
        expected_revision: int | None,
    ) -> dict[str, Any]:
        # The lock serializes read/check/write across processes so a
        # concurrent writer gets a stable revision conflict instead of a
        # lost update or an I/O error. The OS releases it if the holder dies.
        with _annotation_lock(target):
            current = self._load_annotation_document(target)
            current_revision = int(current["revision"]) if current is not None else None
            if current_revision != expected_revision:
                raise RuntimeError("annotation revision conflict")
            payload = dict(document)
            payload["revision"] = (
                0 if current_revision is None else current_revision + 1
            )
            payload["updated_at"] = datetime.now(UTC).isoformat()
            temporary = target.with_suffix(".tmp")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(
                    json.dumps(payload, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(target)
            except (OSError, TypeError, ValueError) as exc:
                temporary.unlink(missing_ok=True)
                raise QPhaseIOError(
                    f"failed to save annotation document: {target}",
                    code=ErrorCode.ARTIFACT_IO,
                    context={"path": str(target)},
                ) from exc
            return payload

    def _append_annotation_event(
        self, session_dir: Path, document: Mapping[str, Any]
    ) -> None:
        """Journal one ``annotations_updated`` event for the owning session."""
        existing = self.read_events(session_dir)
        sequence = (
            max((int(event.get("sequence", 0)) for event in existing), default=0) + 1
        )
        self.append_events(
            session_dir,
            [
                {
                    "sequence": sequence,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "execution_id": "",
                    "session_id": document.get("session_id"),
                    "payload": {
                        "kind": "annotations_updated",
                        "schema": document.get("schema"),
                        "artifact_id": document.get("artifact_id"),
                        "revision": document.get("revision"),
                    },
                }
            ],
        )

    def _artifact_file(self, artifact_dir: Path) -> Path:
        root = self.project.session_root.resolve()
        directory = Path(artifact_dir).expanduser().resolve()
        if not directory.is_relative_to(root):
            raise QPhaseIOError(
                f"artifact path escapes the current project: {directory}",
                code=ErrorCode.ARTIFACT_IO,
                context={"path": str(directory)},
            )
        return directory / ARTIFACT_ANNOTATIONS_FILENAME

    def _session_file(self, session_dir: Path, name: str) -> Path:
        root = self.project.session_root.resolve()
        directory = Path(session_dir).expanduser().resolve()
        if not directory.is_relative_to(root):
            raise QPhaseIOError(
                f"session path escapes the current project: {directory}",
                code=ErrorCode.ARTIFACT_IO,
                context={"path": str(directory)},
            )
        return directory / name

    def _project_file(self, name: str) -> Path:
        root = self.project.root.resolve()
        target = (root / ".qphase" / name).resolve()
        if not target.is_relative_to(root):
            raise QPhaseIOError(
                f"project file escapes the current project: {target}",
                code=ErrorCode.ARTIFACT_IO,
                context={"path": str(target)},
            )
        return target


@contextmanager
def _annotation_lock(target: Path) -> Iterator[None]:
    """Cross-process mutex covering one annotation document's read/check/write.

    The lock lives in a ``<name>.lock`` sibling file. Blocking acquisition is
    safe against holder crashes: the OS releases the lock when the process
    dies.
    """
    lock_path = target.with_name(target.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if sys.platform == "win32":
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if sys.platform == "win32":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
