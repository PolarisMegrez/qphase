"""Execution context and workstation runtime services passed to engines."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
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
        self._pending: dict[str, Any] = {}
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
        if key in self._pending:
            return self._pending[key]
        path = self.root / f"{key}.pkl"
        if not path.exists():
            return None
        with path.open("rb") as handle:
            return pickle.load(handle)

    def save_chunk(self, key: str, payload: Any) -> None:
        if not self.enabled:
            return
        self._pending[key] = payload
        if len(self._pending) >= self.interval_chunks:
            self.flush()

    def flush(self) -> None:
        """Atomically persist all completed chunks waiting for the next interval."""
        for key, payload in self._pending.items():
            path = self.root / f"{key}.pkl"
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)
        self._pending.clear()

    def complete(self) -> None:
        if not self.enabled:
            return
        if self.keep_on_success:
            self.flush()
            return
        self._pending.clear()
        if not self.root.exists():
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
    plugins: dict[str, Any],
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


def plugin_fingerprint(instance: Any) -> dict[str, str | None]:
    """Describe an installed or workspace plugin for checkpoint validation."""
    plugin_type = type(instance)
    module_name = plugin_type.__module__
    distribution_version = None
    top_level = module_name.partition(".")[0]
    for distribution in importlib.metadata.packages_distributions().get(top_level, []):
        try:
            distribution_version = importlib.metadata.version(distribution)
            break
        except importlib.metadata.PackageNotFoundError:
            continue

    source_sha256 = None
    try:
        source_path = inspect.getsourcefile(plugin_type)
        if source_path is not None:
            source_sha256 = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    except (OSError, TypeError):
        pass
    return {
        "class": f"{module_name}:{plugin_type.__qualname__}",
        "distribution_version": distribution_version,
        "source_sha256": source_sha256,
    }
