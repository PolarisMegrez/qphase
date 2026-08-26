"""Execution context and workstation runtime services passed to engines."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pickle
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ErrorCode, QPhaseConfigError, QPhaseRuntimeError
from .progress import ProgressEvent, ProgressReporter
from .scan import ParameterGrid
from .utils import canonical_json

__all__ = [
    "CancellationController",
    "CancellationToken",
    "CheckpointStore",
    "BackendRuntimeSnapshot",
    "ExecutionContext",
    "HardwareSnapshot",
    "ProgressEvent",
    "ProgressReporter",
    "ResourceSnapshot",
    "execution_fingerprint",
    "plugin_fingerprint",
]


class CancellationToken:
    """Thread-safe cooperative cancellation token."""

    def __init__(self, parent: CancellationToken | None = None) -> None:
        self._event = threading.Event()
        self._parent = parent

    @property
    def cancelled(self) -> bool:
        return self._event.is_set() or (
            self._parent is not None and self._parent.cancelled
        )

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise QPhaseRuntimeError(
                "execution cancelled",
                code=ErrorCode.CANCELLATION,
                hint="The job was cancelled by the user or the scheduler.",
            )


class CancellationController:
    """Own execution- and job-scoped cooperative cancellation tokens."""

    def __init__(self) -> None:
        self.execution = CancellationToken()
        self._jobs: dict[str, CancellationToken] = {}
        self._lock = threading.Lock()

    def token_for(self, job_name: str) -> CancellationToken:
        with self._lock:
            return self._jobs.setdefault(
                job_name, CancellationToken(parent=self.execution)
            )

    def cancel_execution(self) -> None:
        self.execution.cancel()

    def cancel_job(self, job_name: str) -> None:
        self.token_for(job_name).cancel()


@dataclass(frozen=True)
class HardwareSnapshot:
    """Dynamic host facts sampled near the start of a logical job."""

    logical_cpu_count: int | None
    total_memory_mib: int | None
    available_memory_mib: int | None

    @classmethod
    def collect(cls) -> HardwareSnapshot:
        total, available = _host_memory_bytes()
        return cls(
            logical_cpu_count=os.cpu_count(),
            total_memory_mib=_bytes_to_mib(total),
            available_memory_mib=_bytes_to_mib(available),
        )


@dataclass(frozen=True)
class BackendRuntimeSnapshot:
    """Optional device facts supplied by the selected backend instance."""

    name: str
    device: str | None
    total_memory_mib: int | None = None
    available_memory_mib: int | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def collect(cls, backend: Any | None) -> BackendRuntimeSnapshot | None:
        if backend is None:
            return None
        name = _safe_backend_call(backend, "backend_name") or type(backend).__name__
        device = _safe_backend_call(backend, "device")
        payload: dict[str, Any] = {}
        provider = getattr(backend, "runtime_resources", None)
        if callable(provider):
            try:
                payload = dict(provider() or {})
            except Exception:
                payload = {}
        capabilities = _safe_backend_call(backend, "capabilities")
        return cls(
            name=str(name),
            device=None if device is None else str(device),
            total_memory_mib=_memory_value_mib(payload, "total"),
            available_memory_mib=_memory_value_mib(payload, "available"),
            capabilities=(dict(capabilities) if isinstance(capabilities, dict) else {}),
        )


@dataclass(frozen=True)
class ResourceSnapshot:
    """Core-collected workstation hints; engines decide how to use them."""

    cpu_worker_limit: int | None
    memory_limit_mib: int | None
    gpu_device: int | str | None
    gpu_memory_fraction: float | None
    hardware: HardwareSnapshot = field(default_factory=HardwareSnapshot.collect)
    backend: BackendRuntimeSnapshot | None = None

    @classmethod
    def from_system_config(
        cls, config: Any, *, backend: Any | None = None
    ) -> ResourceSnapshot:
        resources = config.scan_runtime.resources
        return cls(
            resources.cpu_worker_limit,
            resources.memory_limit_mib,
            resources.gpu_device,
            resources.gpu_memory_fraction,
            HardwareSnapshot.collect(),
            BackendRuntimeSnapshot.collect(backend),
        )


def _safe_backend_call(backend: Any, name: str) -> Any | None:
    value = getattr(backend, name, None)
    if not callable(value):
        return None
    try:
        return value()
    except Exception:
        return None


def _memory_value_mib(payload: dict[str, Any], prefix: str) -> int | None:
    mib = payload.get(f"{prefix}_memory_mib")
    if mib is not None:
        return int(mib)
    byte_count = payload.get(f"{prefix}_memory_bytes")
    return _bytes_to_mib(byte_count)


def _bytes_to_mib(value: Any | None) -> int | None:
    if value is None:
        return None
    return max(0, int(value) // (1024**2))


def _host_memory_bytes() -> tuple[int | None, int | None]:
    """Collect total and currently available physical memory without psutil."""
    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical), int(status.available_physical)
        except Exception:
            return None, None
    if sys.platform.startswith("linux"):
        try:
            values: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                name, raw = line.split(":", 1)
                values[name] = int(raw.strip().split()[0]) * 1024
            return values.get("MemTotal"), values.get("MemAvailable")
        except Exception:
            pass
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return None, None
    try:
        page_size = int(sysconf("SC_PAGE_SIZE"))
        total = page_size * int(sysconf("SC_PHYS_PAGES"))
        available = page_size * int(sysconf("SC_AVPHYS_PAGES"))
        return total, available
    except (AttributeError, OSError, ValueError):
        return None, None


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
                    "backend, or dtype",
                    code=ErrorCode.CHECKPOINT,
                    hint="Delete the .checkpoints directory or restore the "
                    "matching configuration before resuming.",
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
    job_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)


def execution_fingerprint(
    job_config: dict[str, Any],
    *,
    plugins: dict[str, Any],
    backend: str | None,
    dtype: str | None,
) -> dict[str, Any]:
    """Build the compatibility fingerprint stored beside checkpoints."""
    encoded = canonical_json(job_config).encode("utf-8")
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

    return {
        "class": f"{module_name}:{plugin_type.__qualname__}",
        "distribution_version": distribution_version,
    }
