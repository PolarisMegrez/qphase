"""Serializable variable-length CAM bifurcation candidate results."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qphase.core.dataset import DatasetSaveReport
from qphase.core.errors import QPhaseIOError


@dataclass
class CAMBifurcationResult:
    """Candidate states and diagnostics from one adaptive parameter search."""

    states: np.ndarray
    state_vectors: np.ndarray
    control_values: np.ndarray
    control_names: tuple[str, ...]
    full_residual_norm: np.ndarray
    search_residual_norm: np.ndarray
    success: np.ndarray
    status: np.ndarray
    method: np.ndarray
    verification_digits: np.ndarray
    verification_status: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    result_kind = "bifurcation_candidates"

    @property
    def data(self) -> np.ndarray:
        return self.states

    @property
    def metadata(self) -> dict[str, Any]:
        return self.meta

    @property
    def label(self) -> Any:
        return self.meta.get("label")

    @label.setter
    def label(self, value: Any) -> None:
        self.meta["label"] = value

    @property
    def shape(self) -> tuple[int, ...]:
        return (int(len(self.states)),)

    @property
    def axes(self) -> dict[str, np.ndarray]:
        return {"candidate": np.arange(len(self.states), dtype=int)}

    @property
    def nbytes(self) -> int:
        arrays = (
            self.states,
            self.state_vectors,
            self.control_values,
            self.full_residual_norm,
            self.search_residual_norm,
            self.success,
            self.verification_digits,
            *self.diagnostics.values(),
        )
        return sum(np.asarray(value).nbytes for value in arrays)

    def point_view(self, index: tuple[int, ...]) -> CAMBifurcationResult:
        if len(index) != 1:
            raise IndexError("bifurcation results require one candidate index")
        position = index[0]
        return CAMBifurcationResult(
            states=self.states[position : position + 1],
            state_vectors=self.state_vectors[position : position + 1],
            control_values=self.control_values[position : position + 1],
            control_names=self.control_names,
            full_residual_norm=self.full_residual_norm[position : position + 1],
            search_residual_norm=self.search_residual_norm[position : position + 1],
            success=self.success[position : position + 1],
            status=self.status[position : position + 1],
            method=self.method[position : position + 1],
            verification_digits=self.verification_digits[
                position : position + 1
            ],
            verification_status=self.verification_status[
                position : position + 1
            ],
            diagnostics={
                name: self._candidate_slice(value, position)
                for name, value in self.diagnostics.items()
            },
            meta={**self.meta, "candidate_index": int(position)},
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(
                target,
                states=self.states,
                state_vectors=self.state_vectors,
                control_values=self.control_values,
                control_names=np.asarray(self.control_names),
                full_residual_norm=self.full_residual_norm,
                search_residual_norm=self.search_residual_norm,
                success=self.success,
                status=self.status,
                method=self.method,
                verification_digits=self.verification_digits,
                verification_status=self.verification_status,
                diagnostics=np.asarray(self.diagnostics, dtype=object),
                meta=np.asarray(self.meta, dtype=object),
            )
            self._save_csv(target.with_suffix(".csv"))
        except Exception as exc:
            raise QPhaseIOError(
                f"failed to save CAM bifurcation result to {target}: {exc}"
            ) from exc

    def save_dataset(
        self,
        path: str | Path,
        *,
        layout: str,
        shard_target_bytes: int,
    ) -> DatasetSaveReport:
        del layout, shard_target_bytes
        target = Path(path)
        self.save(target)
        files = tuple(
            item
            for item in (target.with_suffix(".npz"), target.with_suffix(".csv"))
            if item.exists()
        )
        return DatasetSaveReport(
            layout="single",
            files=files,
            loader=(
                "qphase_cam.bifurcation_result:CAMBifurcationResult.load"
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> CAMBifurcationResult:
        with np.load(Path(path), allow_pickle=True) as data:
            return cls(
                states=data["states"],
                state_vectors=data["state_vectors"],
                control_values=data["control_values"],
                control_names=tuple(str(value) for value in data["control_names"]),
                full_residual_norm=data["full_residual_norm"],
                search_residual_norm=data["search_residual_norm"],
                success=data["success"],
                status=data["status"],
                method=data["method"],
                verification_digits=data["verification_digits"],
                verification_status=data["verification_status"],
                diagnostics=data["diagnostics"].item(),
                meta=data["meta"].item(),
            )

    def _save_csv(self, path: Path) -> None:
        fields = [
            "candidate",
            *self.control_names,
            "full_residual_norm",
            "search_residual_norm",
            "success",
            "status",
            "method",
            "verification_digits",
            "verification_status",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in range(len(self.states)):
                row: dict[str, Any] = {
                    "candidate": index,
                    "full_residual_norm": self.full_residual_norm[index],
                    "search_residual_norm": self.search_residual_norm[index],
                    "success": bool(self.success[index]),
                    "status": self.status[index],
                    "method": self.method[index],
                    "verification_digits": self.verification_digits[index],
                    "verification_status": self.verification_status[index],
                }
                row.update(
                    zip(
                        self.control_names,
                        self.control_values[index],
                        strict=True,
                    )
                )
                writer.writerow(row)

    @staticmethod
    def _candidate_slice(value: Any, position: int) -> Any:
        array = np.asarray(value)
        if array.ndim > 0 and len(array) > position:
            return array[position : position + 1]
        return value
