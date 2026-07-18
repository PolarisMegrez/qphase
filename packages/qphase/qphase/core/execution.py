"""Execution context and workstation runtime services passed to engines."""

from __future__ import annotations

import hashlib
import json
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import QPhaseConfigError, QPhaseRuntimeError
from .scan import ParameterGrid


class CancellationToken:
    """Thread-safe cooperative cancellation token."""

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise QPhaseRuntimeError("execution cancelled")


class ProgressReporter:
    """Stable progress API for resource engines."""

    def __init__(self, callback: Any | None = None) -> None:
        self._callback = callback

    def report(
        self,
        percent: float | None,
        *,
        total_duration_estimate: float | None = None,
        message: str = "",
        stage: str | None = None,
    ) -> None:
        if self._callback is not None:
            self._callback(percent, total_duration_estimate, message, stage)

    def __call__(
        self,
        percent: float | None,
        total_duration_estimate: float | None,
        message: str,
        stage: str | None,
    ) -> None:
        self.report(
            percent,
            total_duration_estimate=total_duration_estimate,
            message=message,
            stage=stage,
        )


@dataclass(frozen=True)
class ResourceSnapshot:
    """Core-collected workstation hints; engines decide how to use them."""

    cpu_worker_limit: int | None
    memory_limit_mib: int | None
    gpu_device: int | str | None
    gpu_memory_fraction: float | None

    @classmethod
    def from_system_config(cls, config: Any) -> ResourceSnapshot:
        resources = config.scan_runtime.resources
        return cls(
            resources.cpu_worker_limit,
            resources.memory_limit_mib,
            resources.gpu_device,
            resources.gpu_memory_fraction,
        )


class CheckpointStore:
    """Chunk-level checkpoint storage scoped to one logical job."""

    def __init__(self, root: Path, config: Any, fingerprint: dict[str, Any]) -> None:
        self.root = root / ".checkpoints"
        self.enabled = bool(config.enabled)
        self.interval_chunks = int(config.interval_chunks)
        self.keep_on_success = bool(config.keep_on_success)
        self.fingerprint = fingerprint
        if self.enabled:
            self._prepare()

    def _prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / "checkpoint_manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("fingerprint") != self.fingerprint:
                raise QPhaseConfigError(
                    "checkpoint is incompatible with the current config, plugins, "
                    "backend, or dtype"
                )
            return
        manifest_path.write_text(
            json.dumps(
                {"schema_version": "1.0", "fingerprint": self.fingerprint},
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_chunk(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self.root / f"{key}.pkl"
        if not path.exists():
            return None
        with path.open("rb") as handle:
            return pickle.load(handle)

    def save_chunk(self, key: str, payload: Any) -> None:
        if not self.enabled:
            return
        path = self.root / f"{key}.pkl"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)

    def complete(self) -> None:
        if not self.enabled or self.keep_on_success or not self.root.exists():
            return
        for path in self.root.iterdir():
            if path.is_file():
                path.unlink()
        self.root.rmdir()


@dataclass
class ExecutionContext:
    """Runtime services supplied by the scheduler to one logical engine job."""

    parameter_grid: ParameterGrid | None
    resources: ResourceSnapshot
    progress: ProgressReporter
    cancellation: CancellationToken
    artifacts: Any
    checkpoints: CheckpointStore
    run_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)


def execution_fingerprint(
    job_config: dict[str, Any],
    *,
    plugins: dict[str, str],
    backend: str | None,
    dtype: str | None,
) -> dict[str, Any]:
    """Build the compatibility fingerprint stored beside checkpoints."""
    encoded = json.dumps(job_config, sort_keys=True, default=str).encode("utf-8")
    return {
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "plugins": plugins,
        "backend": backend,
        "dtype": dtype,
    }
