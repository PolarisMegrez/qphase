"""Project-scoped persistence ports for sessions and execution events.

The implementation is intentionally one small file-backed store. Session
manifests and event journals have different protocols, but they share the
project filesystem and do not need separate database wrappers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from .errors import ErrorCode, QPhaseIOError
from .project import ProjectContext

__all__ = [
    "EventStoreProtocol",
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


class ProjectStateStore(SessionStoreProtocol, EventStoreProtocol):
    """Single project-scoped file implementation of Session/Event ports."""

    def __init__(self, project: ProjectContext) -> None:
        self.project = project

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
