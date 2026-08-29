"""Shared normalization of saved or in-memory SDE analyser inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qphase.core.aggregation import AggregateResult, DirectoryInputResult
from qphase.core.dataset import DatasetResultProtocol
from qphase.core.errors import QPhaseError
from qphase.data import Dataset, load_bundle

from ..result import SDEDataBundle, SDEResult, legacy_view_from_products

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


def _is_products_mapping(data: Any) -> bool:
    """Check whether ``data`` is a mapping of product labels to datasets."""
    return (
        isinstance(data, dict)
        and bool(data)
        and all(isinstance(value, Dataset) for value in data.values())
    )


def _expand_dataset(data: Any, *, path: Path) -> list[LoadedResult]:
    """Expand one logical dataset into per-point legacy result views."""
    shape = tuple(data.shape)
    indices = list(np.ndindex(shape)) if shape else [()]
    loaded = []
    for flat_index, index in enumerate(indices):
        view = data.point_view(index)
        if isinstance(view, SDEDataBundle):
            view = view.legacy_result()
        if isinstance(view, SDEResult):
            loaded.append(
                LoadedResult(
                    path=path,
                    job_name=view.meta.get("job_name", f"point_{flat_index:06d}"),
                    result=view,
                )
            )
    return loaded


def _load_artifact_directory(root: Path) -> list[LoadedResult]:
    """Load SDE artifact directories under ``root`` as per-point results."""
    if root.is_file():
        raise QPhaseError(f"analyser input must be an artifact directory: {root}")
    if not root.exists():
        raise QPhaseError(f"Result directory does not exist: {root}")
    job_dirs = sorted(
        manifest.parent for manifest in root.glob("*/artifact_manifest.json")
    )
    if not job_dirs and (root / "artifact_manifest.json").exists():
        job_dirs = [root]
    if not job_dirs:
        raise QPhaseError(f"no artifact manifests found under {root}")
    loaded: list[LoadedResult] = []
    for job_dir in job_dirs:
        bundle = load_bundle(job_dir)
        if not isinstance(bundle, SDEDataBundle):
            raise QPhaseError(f"artifact at {job_dir} is not an SDE data bundle")
        loaded.extend(_expand_dataset(bundle, path=job_dir))
    return loaded


def load_sde_results(data: Any) -> list[LoadedResult]:
    """Normalize analyser input into point-level SDE results."""
    if isinstance(data, SDEDataBundle) or isinstance(data, DatasetResultProtocol):
        dataset_loaded = _expand_dataset(data, path=Path("."))
        if dataset_loaded:
            return dataset_loaded

    if isinstance(data, DirectoryInputResult):
        data = data.path

    if isinstance(data, str | Path):
        return _load_artifact_directory(Path(data))

    if isinstance(data, AggregateResult):
        data = data.results

    if _is_products_mapping(data):
        # One current artifact restored through the manifest loader: the mapping
        # itself is a single logical result with one product per analyser.
        result = legacy_view_from_products(data)
        return [
            LoadedResult(
                path=Path("."),
                job_name=result.meta.get("job_name", "artifact"),
                result=result,
            )
        ]

    if isinstance(data, dict):
        loaded: list[LoadedResult] = []
        for name, result in data.items():
            if isinstance(result, SDEDataBundle):
                loaded.append(
                    LoadedResult(
                        path=Path("."),
                        job_name=name,
                        result=result.legacy_result(),
                    )
                )
                continue
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
