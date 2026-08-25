"""Artifact persistence for one logical job result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .dataset import DatasetResultProtocol, DatasetSaveReport, estimate_result_nbytes
from .errors import QPhaseRuntimeError
from .protocols import ResultProtocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..data.datasets import Dataset
    from ..data.store import ArtifactManifestV3


class ArtifactStore:
    """Save logical results and describe their physical representation."""

    def __init__(self, job_dir: Path, config: Any) -> None:
        self.job_dir = job_dir
        self.config = config

    def save_products(
        self,
        products: Mapping[str, Dataset],
        *,
        provenance: Mapping[str, Any] | None = None,
        parents: Sequence[str] = (),
        artifact_id: str | None = None,
    ) -> ArtifactManifestV3:
        """Persist typed data products and write the v3 artifact manifest.

        This is the typed counterpart of :meth:`save_result`: products are
        stored through the manifest v3 pipeline (native dtypes, per-chunk
        checksums, lazy restore). The storage layout honors the system
        configuration exactly like :meth:`save_result` does: ``single``
        disables external sharding, ``auto`` compares the estimated payload
        size against ``auto_shard_threshold_mib`` and the legacy
        ``per_point`` maps to byte-targeted sharding.
        """
        # Imported lazily: core must not depend on the data layer at module
        # import time (the data layer already depends on core.utils).
        from ..data.store import save_products

        return save_products(
            self.job_dir,
            products,
            provenance=provenance,
            parents=parents,
            artifact_id=artifact_id,
            shard_target_bytes=int(self.config.shard_target_mib * (1 << 20)),
            layout=self._products_layout(products),
        )

    def _products_layout(self, products: Mapping[str, Dataset]) -> str:
        """Resolve the configured layout for a typed product mapping."""
        requested = getattr(self.config, "storage_layout", "auto")
        if requested == "per_point":
            # Legacy export-only layout; the v3 pipeline stores per-point
            # chunks through byte-targeted sharding instead.
            return "sharded"
        if requested != "auto":
            return requested
        total = 0
        size: int | None = total
        for dataset in products.values():
            if dataset.nbytes is None:
                size = None
                break
            total += dataset.nbytes
        else:
            size = total
        threshold = int(
            getattr(self.config, "auto_shard_threshold_mib", 512) * (1 << 20)
        )
        return "sharded" if size is not None and size > threshold else "single"

    def load_products(self) -> dict[str, Dataset]:
        """Reopen this job's artifact directory as lazily-backed datasets."""
        from ..data.store import load_products

        return load_products(self.job_dir)

    def save_result(self, result: ResultProtocol, name: str) -> Path:
        from collections.abc import Mapping as _Mapping

        from ..data.datasets import Dataset as _Dataset

        products = getattr(result, "products", None)
        if isinstance(products, _Mapping) and all(
            isinstance(product, _Dataset) for product in products.values()
        ):
            # 2.0 typed bundles persist through the v3 manifest pipeline;
            # no legacy manifest is written for them.
            provenance = getattr(result, "provenance", None)
            if provenance is not None and hasattr(provenance, "model_dump"):
                provenance = provenance.model_dump(mode="json")
            self.save_products(
                products,
                provenance={"job_name": name, **(provenance or {})},
            )
            return self.job_dir / "artifact_manifest.json"

        before = set(self.job_dir.rglob("*"))
        layout = self._layout(result)
        base = self.job_dir / name
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
                    for path in self.job_dir.rglob("*")
                    if path.is_file() and path not in before
                )
            )
            report = DatasetSaveReport("single", files)
        additional = tuple(
            sorted(
                path
                for path in self.job_dir.rglob("*")
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
        manifest_path = self.job_dir / "artifact_manifest.json"
        payload = {
            "schema_version": "2.0",
            "result_type": f"{type(result).__module__}:{type(result).__qualname__}",
            "result_schema": getattr(result, "schema_version", "1.0"),
            "axes": _jsonable(getattr(result, "axes", {})),
            "shape": list(getattr(result, "shape", ())),
            "layout": report.layout,
            "files": [
                str(path.relative_to(self.job_dir))
                if path.is_relative_to(self.job_dir)
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
