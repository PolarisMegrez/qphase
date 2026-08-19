"""Shared normalization of saved or in-memory SDE analyser inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qphase.core.aggregation import (
    AggregateResult,
    DirectoryInputResult,
    iter_result_files,
)
from qphase.core.dataset import DatasetResultProtocol
from qphase.core.errors import QPhaseError

from ..result import SDEResult

__all__ = ["LoadedResult", "load_sde_results"]


@dataclass(frozen=True)
class LoadedResult:
    """Normalized view of one saved SDE job result."""

    path: Path
    job_name: str
    result: SDEResult

    @property
    def meta(self) -> dict[str, Any]:
        return self.result.meta

    @property
    def analysis(self) -> dict[str, Any]:
        return self.result.analysis


def load_sde_results(data: Any, pattern: str) -> list[LoadedResult]:
    """Normalize analyser input into point-level SDE results."""
    if isinstance(data, DatasetResultProtocol):
        dataset_loaded = []
        for flat_index, index in enumerate(np.ndindex(data.shape)):
            result = data.point_view(index)
            if isinstance(result, SDEResult):
                dataset_loaded.append(
                    LoadedResult(
                        path=Path("."),
                        job_name=result.meta.get(
                            "job_name", f"point_{flat_index:06d}"
                        ),
                        result=result,
                    )
                )
        if dataset_loaded:
            return dataset_loaded

    if isinstance(data, DirectoryInputResult):
        data = data.path

    if isinstance(data, str | Path):
        return [
            LoadedResult(
                path=path,
                job_name=path.parent.name or path.stem,
                result=SDEResult.load(path),
            )
            for path in iter_result_files(data, pattern)
        ]

    if isinstance(data, AggregateResult):
        data = data.results

    if isinstance(data, dict):
        loaded: list[LoadedResult] = []
        for name, result in data.items():
            if isinstance(result, SDEResult):
                loaded.append(
                    LoadedResult(path=Path("."), job_name=name, result=result)
                )
                continue
            result_data = getattr(result, "data", None)
            if isinstance(result_data, SDEResult):
                loaded.append(
                    LoadedResult(path=Path("."), job_name=name, result=result_data)
                )
        if not loaded:
            raise QPhaseError(
                "analyser received a dict but no SDEResult values were found"
            )
        return loaded

    if isinstance(data, SDEResult):
        return [LoadedResult(path=Path("."), job_name="single", result=data)]

    raise QPhaseError(
        f"analyser received unsupported input type: {type(data).__name__}"
    )
