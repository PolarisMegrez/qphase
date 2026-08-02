from qphase.core.execution import (
    BackendRuntimeSnapshot,
    HardwareSnapshot,
    ResourceSnapshot,
)
from qphase.core.system_config import SystemConfig


class FakeBackend:
    def backend_name(self):
        return "accelerator"

    def device(self):
        return "device:2"

    def capabilities(self):
        return {"fft": True}

    def runtime_resources(self):
        return {
            "total_memory_bytes": 8 * 1024**3,
            "available_memory_bytes": 6 * 1024**3,
        }


def test_backend_runtime_snapshot_uses_optional_provider():
    snapshot = BackendRuntimeSnapshot.collect(FakeBackend())

    assert snapshot is not None
    assert snapshot.name == "accelerator"
    assert snapshot.device == "device:2"
    assert snapshot.total_memory_mib == 8192
    assert snapshot.available_memory_mib == 6144
    assert snapshot.capabilities == {"fft": True}


def test_resource_snapshot_combines_policy_and_runtime_facts(monkeypatch):
    hardware = HardwareSnapshot(12, 32768, 24576)
    monkeypatch.setattr(HardwareSnapshot, "collect", classmethod(lambda cls: hardware))
    config = SystemConfig.model_validate(
        {
            "scan_runtime": {
                "resources": {
                    "cpu_worker_limit": 6,
                    "memory_limit_mib": 16000,
                    "gpu_device": 2,
                    "gpu_memory_fraction": 0.8,
                }
            }
        }
    )

    snapshot = ResourceSnapshot.from_system_config(config, backend=FakeBackend())

    assert snapshot.cpu_worker_limit == 6
    assert snapshot.memory_limit_mib == 16000
    assert snapshot.hardware is hardware
    assert snapshot.backend is not None
    assert snapshot.backend.available_memory_mib == 6144
