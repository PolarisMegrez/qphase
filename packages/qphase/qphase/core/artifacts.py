"""Artifact persistence for one logical job result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset import DatasetResultProtocol, DatasetSaveReport, estimate_result_nbytes
from .errors import QPhaseRuntimeError
from .protocols import ResultProtocol


class ArtifactStore:
    """Save logical results and describe their physical representation."""

    def __init__(self, run_dir: Path, config: Any) -> None:
        self.run_dir = run_dir
        self.config = config

    def save_result(self, result: ResultProtocol, name: str) -> Path:
        before = set(self.run_dir.rglob("*"))
        layout = self._layout(result)
        base = self.run_dir / name
        if isinstance(result, DatasetResultProtocol):
            report = result.save_dataset(
                base,
                layout=layout,
                shard_target_bytes=int(self.config.shard_target_mib * (1 << 20)),
            )
        else:
            result.save(base)
            files = tuple(
                sorted(
                    path
                    for path in self.run_dir.rglob("*")
                    if path.is_file() and path not in before
                )
            )
            report = DatasetSaveReport("single", files)
        additional = tuple(
            sorted(
                path
                for path in self.run_dir.rglob("*")
                if path.is_file()
                and path.name not in {"artifact_manifest.json", "config_snapshot.json"}
                and path not in report.files
            )
        )
        if additional:
            report = DatasetSaveReport(
                report.layout,
                report.files + additional,
                loader=report.loader,
                schema_version=report.schema_version,
            )
        manifest_path = self.run_dir / "artifact_manifest.json"
        payload = {
            "schema_version": "1.0",
            "result_type": f"{type(result).__module__}:{type(result).__qualname__}",
            "result_schema": getattr(result, "schema_version", "1.0"),
            "axes": _jsonable(getattr(result, "axes", {})),
            "shape": list(getattr(result, "shape", ())),
            "layout": report.layout,
            "files": [
                str(path.relative_to(self.run_dir))
                if path.is_relative_to(self.run_dir)
                else str(path)
                for path in report.files
            ],
            "loader": report.loader,
        }
        try:
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            raise QPhaseRuntimeError(
                f"failed to write artifact manifest: {exc}"
            ) from exc
        return manifest_path

    def _layout(self, result: ResultProtocol) -> str:
        requested = self.config.storage_layout
        if requested != "auto":
            return requested
        size = estimate_result_nbytes(result)
        threshold = int(self.config.auto_shard_threshold_mib * (1 << 20))
        return "sharded" if size is not None and size > threshold else "single"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
