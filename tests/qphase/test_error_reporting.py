from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.error_report import build_error_report, save_error_report
from qphase.core.errors import ErrorCode, QPhaseRuntimeError
from qphase.core.protocols import EngineManifest
from qphase.core.registry import registry
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig

pytestmark = pytest.mark.integration


class FailingEngineConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


class FailingEngine:
    name: ClassVar[str] = "failing"
    config_schema: ClassVar[type[FailingEngineConfig]] = FailingEngineConfig
    manifest: ClassVar[EngineManifest] = EngineManifest(required_plugins=set())

    def __init__(self, config: Any = None, plugins: Any = None, **kwargs: Any) -> None:
        del config, plugins, kwargs

    def run(self, data: Any = None, **kwargs: Any) -> Any:
        del data, kwargs
        try:
            raise ValueError("inner numerical failure")
        except ValueError as exc:
            raise QPhaseRuntimeError(
                "fixed-point solve failed",
                code=ErrorCode.ENGINE_RUNTIME,
                hint="Check the initial guesses.",
                context={"stage": "solve"},
            ) from exc


def test_error_report_preserves_cause_chain_and_traceback(tmp_path: Path) -> None:
    try:
        try:
            raise ValueError("inner")
        except ValueError as exc:
            raise QPhaseRuntimeError("outer", hint="retry") from exc
    except QPhaseRuntimeError as exc:
        report = build_error_report(exc, job_name="job", engine="cam")

    path = save_error_report(report, tmp_path)
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [item["type"] for item in payload["cause_chain"]] == [
        "QPhaseRuntimeError",
        "ValueError",
    ]
    assert "ValueError: inner" in payload["traceback"]
    assert payload["hint"] == "retry"


def test_scheduler_uses_one_error_id_across_result_manifest_and_report(
    tmp_path: Path,
    temp_project,
) -> None:
    registry.register("engine", "failing", FailingEngine, overwrite=True)
    try:
        config = SystemConfig()
        scheduler = Scheduler(system_config=config, project=temp_project)

        results = scheduler.run(
            WorkflowSpec(
                schema="qphase.workflow/2",
                id="test-workflow",
                title="Test Workflow",
                jobs=[JobConfig(name="broken", engine={"failing": {}}, save=False)],
            )
        )
    finally:
        registry._tables.get("engine", {}).pop("failing", None)

    assert len(results) == 1
    result = results[0]
    assert result.status == "failed"
    assert result.error_id is not None
    assert result.error_report_path is not None
    report = json.loads(Path(result.error_report_path).read_text(encoding="utf-8"))
    assert report["error_id"] == result.error_id
    assert report["code"] == ErrorCode.ENGINE_RUNTIME

    assert scheduler.session_dir is not None
    manifest = json.loads(
        (scheduler.session_dir / "session_manifest.json").read_text(encoding="utf-8")
    )
    entry = manifest["jobs"]["broken"]
    assert entry["error_id"] == result.error_id
    assert entry["error_report"] == result.error_report_path
    assert manifest["status"] == "failed"
    assert manifest["workflow_hash"]
    assert (scheduler.session_dir / "workflow_snapshot.yaml").exists()
    assert (scheduler.session_dir / "qphase.log").exists()
