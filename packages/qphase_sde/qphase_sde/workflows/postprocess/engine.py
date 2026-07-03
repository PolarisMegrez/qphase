"""Scheduler engine for SDE postprocessing workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from qphase.core.protocols import EngineManifest
from qphase_sde.postprocess import export_postprocess_bundle, postprocess_run
from qphase_sde.result import SDEResult


class SDEPostprocessEngineConfig(BaseModel):
    """Configuration for the SDE postprocess workflow engine."""

    model_config = ConfigDict(extra="allow")

    run_dir: str | None = Field(
        default=None,
        description="Run directory or .npz result file to postprocess.",
    )
    scan_param: str = Field(..., description="Parameter name under meta.params.")
    psd_key: str = "psd"
    mode: int = 0
    fit_window: float | None = None
    export_dist: bool = False
    overwrite: bool = False
    output_dir: str | None = None
    pattern: str = "*.npz"


class SDEPostprocessEngine:
    """Package-level workflow engine for postprocessing saved SDE results."""

    config_schema = SDEPostprocessEngineConfig
    manifest: ClassVar[EngineManifest] = EngineManifest(required_plugins=set())

    def __init__(self, config: SDEPostprocessEngineConfig, **_: Any):
        self.config = config

    def run(self, data: Any | None = None, **_: Any) -> SDEResult:
        source = self.config.run_dir or data
        if source is None:
            raise ValueError("sde_postprocess requires run_dir or input data")
        if not isinstance(source, str | Path):
            raise TypeError("sde_postprocess currently expects a saved run path")

        output_dir = self.config.output_dir or str(Path(source).parent)
        bundle = postprocess_run(
            source,
            scan_param=self.config.scan_param,
            psd_key=self.config.psd_key,
            mode=self.config.mode,
            fit_window=self.config.fit_window,
            export_dist=self.config.export_dist,
            pattern=self.config.pattern,
        )
        written = export_postprocess_bundle(
            bundle,
            output_dir,
            scan_param=self.config.scan_param,
            overwrite=self.config.overwrite,
            export_dist=self.config.export_dist,
        )
        return SDEResult(
            analysis={
                "postprocess": {
                    "fit_rows": bundle.fit_rows,
                    "psd_columns": list(bundle.psd_columns.keys()),
                    "artifacts": {key: str(path) for key, path in written.items()},
                }
            },
            meta={
                "engine": "sde_postprocess",
                "params": {"scan_param": self.config.scan_param},
            },
        )
