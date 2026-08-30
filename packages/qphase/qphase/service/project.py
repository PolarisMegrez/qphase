"""Project-scoped Session history and Workflow-document services."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

from qphase.core.annotations import SessionAnnotationDocument
from qphase.core.errors import ErrorCode, QPhaseIOError
from qphase.core.persistence import ProjectStateStore
from qphase.core.project import ProjectContext
from qphase.core.workflow import WorkflowCatalog, load_workflow

from .models import SessionSummary, WorkflowDocument

_UNSET = object()


class ProjectService:
    """Manage immutable sessions and editable workflow documents for one project."""

    def __init__(self, project: ProjectContext) -> None:
        self.project = project
        self.catalog = WorkflowCatalog(project)
        self.state_store = ProjectStateStore(project)

    @property
    def output_root(self) -> Path:
        return self.project.session_root

    def list_sessions(self) -> list[SessionSummary]:
        manifests = (
            list(self.output_root.rglob("session_manifest.json"))
            if self.output_root.exists()
            else []
        )
        sessions = [
            self._session_summary(path.parent)
            for path in manifests
            if ".trash" not in path.parts
        ]
        return sorted(
            sessions, key=lambda item: item.start_time or datetime.min, reverse=True
        )

    def get_session(self, session_id: str) -> SessionSummary:
        return self._session_summary(self.session_dir(session_id))

    def session_events(self, session_id: str, *, after: int = 0) -> list[dict]:
        """Read persisted execution events for a completed or active Session."""
        return self.state_store.read_events(self.session_dir(session_id), after=after)

    def update_session(
        self,
        session_id: str,
        *,
        alias: str | None | object = _UNSET,
        note: str | None | object = _UNSET,
    ) -> SessionSummary:
        root = self.session_dir(session_id)
        current = self.state_store.load_session_annotations(root)
        if current is not None:
            document = SessionAnnotationDocument.model_validate(current)
            expected_revision: int | None = document.revision
        else:
            document = self.new_session_annotations(session_id)
            expected_revision = None
        if alias is not _UNSET:
            document.alias = cast("str | None", alias)
        if note is not _UNSET:
            document.note = cast("str | None", note)
        self.state_store.save_session_annotations(
            root,
            document.model_dump(mode="json", by_alias=True),
            expected_revision=expected_revision,
        )
        return self.get_session(session_id)

    def new_session_annotations(self, session_id: str) -> SessionAnnotationDocument:
        """Create the initial (empty) annotation document for one session."""
        return SessionAnnotationDocument(
            project_id=self.project.project_id,
            session_id=session_id,
        )

    def trash_session(self, session_id: str) -> None:
        root = self.session_dir(session_id)
        if self.get_session(session_id).status == "running":
            raise ValueError("a running session cannot be deleted")
        trash = self.output_root / ".trash"
        trash.mkdir(parents=True, exist_ok=True)
        target = trash / session_id
        if target.exists():
            target = trash / f"{session_id}-{datetime.now():%Y%m%d%H%M%S}"
        root.replace(target)

    def purge_trash(self) -> int:
        root = self.output_root / ".trash"
        if not root.exists():
            return 0
        entries = list(root.iterdir())
        for path in entries:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        return len(entries)

    def list_workflow_documents(self) -> list[WorkflowDocument]:
        return [
            self._workflow_document(item.path, item.relative_path, content=False)
            for item in self.catalog.list()
        ]

    def get_workflow_document(self, doc_id: str) -> WorkflowDocument:
        path = self._workflow_path(doc_id, must_exist=True)
        return self._workflow_document(path, doc_id, content=True)

    def put_workflow_document(
        self, doc_id: str, content: str, *, revision: str | None
    ) -> WorkflowDocument:
        path = self._workflow_path(doc_id, must_exist=False)
        if path.exists() and revision != self._revision(path.read_bytes()):
            raise RuntimeError("workflow revision conflict")
        temporary = path.with_suffix(path.suffix + ".validate")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(content, encoding="utf-8")
        try:
            load_workflow(temporary)
            existing = {
                item.id
                for item in self.catalog.list()
                if item.path.resolve() != path.resolve()
            }
            if load_workflow(temporary).id in existing:
                raise ValueError("workflow id must be unique within the project")
        finally:
            temporary.unlink(missing_ok=True)
        self._atomic_text(path, content)
        return self._workflow_document(path, doc_id, content=True)

    def delete_workflow_document(self, doc_id: str, *, revision: str) -> None:
        path = self._workflow_path(doc_id, must_exist=True)
        if revision != self._revision(path.read_bytes()):
            raise RuntimeError("workflow revision conflict")
        path.unlink()

    def session_dir(self, session_id: str) -> Path:
        matches = [
            path.parent
            for path in self.output_root.rglob("session_manifest.json")
            if path.parent.name == session_id and ".trash" not in path.parts
        ]
        if len(matches) != 1:
            raise FileNotFoundError(f"session not found or ambiguous: {session_id}")
        return matches[0].resolve()

    def _session_summary(self, root: Path) -> SessionSummary:
        manifest = self._read_json(root / "session_manifest.json")
        # Annotation documents are the only store for alias/note; once one
        # exists it is authoritative.
        annotations = self.state_store.load_session_annotations(root)
        if annotations is not None:
            alias = annotations.get("alias")
            note = annotations.get("note")
        else:
            alias = note = None
        status = str(manifest.get("status", "unknown"))
        if status == "running" and not self._owner_alive(root / "session.lock"):
            status = "interrupted"
        files = [path for path in root.rglob("*") if path.is_file()]
        modified = (
            datetime.fromtimestamp(max(path.stat().st_mtime for path in files))
            if files
            else None
        )
        started = manifest.get("start_time")
        jobs = manifest.get("jobs", {})
        if not isinstance(jobs, dict):
            raise QPhaseIOError(
                f"session manifest jobs must be an object: {root}",
                code=ErrorCode.ARTIFACT_IO,
                context={"path": str(root / "session_manifest.json")},
            )
        return SessionSummary(
            session_id=str(manifest.get("session_id") or root.name),
            project_id=str(manifest["project_id"])
            if manifest.get("project_id")
            else None,
            workflow_id=str(manifest["workflow_id"])
            if manifest.get("workflow_id")
            else None,
            status=status,
            alias=alias if isinstance(alias, str) else None,
            note=note if isinstance(note, str) else None,
            start_time=datetime.fromisoformat(str(started)) if started else None,
            last_update=modified,
            jobs={str(key): value for key, value in jobs.items()},
        )

    def _workflow_path(self, doc_id: str, *, must_exist: bool) -> Path:
        if Path(doc_id).suffix.lower() not in {".yaml", ".yml"}:
            raise PermissionError("workflow document must be a YAML file")
        path = (self.project.workflow_root / doc_id).resolve()
        if not path.is_relative_to(self.project.workflow_root):
            raise PermissionError("workflow path escapes the project workflow root")
        if must_exist and not path.exists():
            raise FileNotFoundError(f"workflow document not found: {doc_id}")
        return path

    def _workflow_document(
        self, path: Path, doc_id: str, *, content: bool
    ) -> WorkflowDocument:
        raw = path.read_bytes()
        workflow = load_workflow(path)
        return WorkflowDocument(
            doc_id=doc_id,
            workflow_id=workflow.id,
            title=workflow.title,
            path=path,
            writable=os.access(path, os.W_OK),
            revision=self._revision(raw),
            job_names=[job.name for job in workflow.jobs],
            content=raw.decode("utf-8") if content else None,
        )

    @staticmethod
    def _revision(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QPhaseIOError(
                f"failed to read project JSON: {path}",
                code=ErrorCode.ARTIFACT_IO,
                context={"path": str(path)},
            ) from exc
        if not isinstance(payload, dict):
            raise QPhaseIOError(
                f"project JSON must contain an object: {path}",
                code=ErrorCode.ARTIFACT_IO,
                context={"path": str(path)},
            )
        return {str(key): value for key, value in payload.items()}

    @staticmethod
    def _owner_alive(path: Path) -> bool:
        payload = ProjectService._read_json(path)
        pid, heartbeat = payload.get("pid"), payload.get("heartbeat")
        if not isinstance(pid, int) or not isinstance(heartbeat, str):
            return False
        try:
            last_seen = datetime.fromisoformat(heartbeat)
            if datetime.now(tz=last_seen.tzinfo) - last_seen > timedelta(seconds=30):
                return False
        except ValueError:
            return False
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            kernel32.CloseHandle(handle)
            return True
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
