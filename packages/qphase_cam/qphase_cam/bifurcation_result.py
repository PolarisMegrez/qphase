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
class CAMBifurcationBranchTable:
    """Local response branches linked to candidates by integer index."""

    candidate_index: np.ndarray
    local_branch_index: np.ndarray
    signature_index: np.ndarray
    state_order: np.ndarray
    perturbation_order: np.ndarray
    coupling_state_order: np.ndarray
    exponent_numerator: np.ndarray
    exponent_denominator: np.ndarray
    epsilon_side: np.ndarray
    amplitude: np.ndarray
    real_branch: np.ndarray
    sublinear: np.ndarray
    leading_state_coefficient: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.candidate_index))

    def view(self, index: int) -> dict[str, Any]:
        return {name: getattr(self, name)[index] for name in self.__dataclass_fields__}

    def to_table(self) -> list[dict[str, Any]]:
        return [self.view(index) for index in range(self.size)]


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
    branches: CAMBifurcationBranchTable | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    result_kind = "bifurcation_candidates"
    schema_version = "2.0"

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
        branch_arrays = (
            ()
            if self.branches is None
            else tuple(
                getattr(self.branches, name)
                for name in self.branches.__dataclass_fields__
            )
        )
        return sum(np.asarray(value).nbytes for value in (*arrays, *branch_arrays))

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
            verification_digits=self.verification_digits[position : position + 1],
            verification_status=self.verification_status[position : position + 1],
            branches=self._branch_subset(position),
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
                bifurcation_schema_version=np.asarray(self.schema_version),
                **self._branch_arrays(),
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
            loader=("qphase_cam.bifurcation_result:CAMBifurcationResult.load"),
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
                branches=cls._load_branches(data),
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
            "is_physical",
            "is_stable",
            "classification_status",
            "classification_accepted",
            "signature_count",
            "minimum_exponent",
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
                    "is_physical": self._diagnostic(index, "is_physical", False),
                    "is_stable": self._diagnostic(index, "is_stable", False),
                    "classification_status": self._diagnostic(
                        index, "classification_status", "not_run"
                    ),
                    "classification_accepted": self._diagnostic(
                        index, "classification_accepted", False
                    ),
                    "signature_count": self._signature_count(index),
                    "minimum_exponent": self._minimum_exponent(index),
                }
                row.update(
                    zip(
                        self.control_names,
                        self.control_values[index],
                        strict=True,
                    )
                )
                writer.writerow(row)

    def to_candidate_table(self) -> list[dict[str, Any]]:
        output = []
        for index in range(len(self.states)):
            row = {
                "candidate": index,
                **dict(
                    zip(
                        self.control_names,
                        self.control_values[index],
                        strict=True,
                    )
                ),
                "full_residual_norm": self.full_residual_norm[index],
                "search_residual_norm": self.search_residual_norm[index],
                "success": bool(self.success[index]),
                "status": self.status[index],
                "method": self.method[index],
                "is_physical": self._diagnostic(index, "is_physical", False),
                "is_stable": self._diagnostic(index, "is_stable", False),
                "classification_status": self._diagnostic(
                    index, "classification_status", "not_run"
                ),
                "classification_accepted": self._diagnostic(
                    index, "classification_accepted", False
                ),
                "signature_count": self._signature_count(index),
                "minimum_exponent": self._minimum_exponent(index),
            }
            for state_index, state_id in enumerate(self.meta.get("state_ids", ())):
                row[str(state_id)] = self.state_vectors[index, state_index]
            output.append(row)
        return output

    def to_branch_table(self) -> list[dict[str, Any]]:
        return [] if self.branches is None else self.branches.to_table()

    def branch_view(self, index: int) -> dict[str, Any]:
        if self.branches is None:
            raise IndexError("bifurcation result has no response branches")
        return self.branches.view(index)

    def _branch_arrays(self) -> dict[str, np.ndarray]:
        if self.branches is None:
            return {}
        return {
            f"branch_{name}": np.asarray(getattr(self.branches, name))
            for name in self.branches.__dataclass_fields__
        }

    @classmethod
    def _load_branches(cls, data: Any) -> CAMBifurcationBranchTable | None:
        names = tuple(CAMBifurcationBranchTable.__dataclass_fields__)
        if not all(f"branch_{name}" in data.files for name in names):
            return None
        return CAMBifurcationBranchTable(
            **{name: data[f"branch_{name}"] for name in names}
        )

    def _branch_subset(self, position: int) -> CAMBifurcationBranchTable | None:
        if self.branches is None:
            return None
        mask = self.branches.candidate_index == position
        if not np.any(mask):
            return None
        values = {
            name: np.asarray(getattr(self.branches, name))[mask]
            for name in self.branches.__dataclass_fields__
        }
        values["candidate_index"] = np.zeros(np.count_nonzero(mask), dtype=int)
        return CAMBifurcationBranchTable(**values)

    def _diagnostic(self, index: int, name: str, default: Any) -> Any:
        value = self.diagnostics.get(name)
        if value is None:
            return default
        array = np.asarray(value)
        return array[index] if array.ndim else array.item()

    def _signature_count(self, index: int) -> int:
        return len(self._signatures(index))

    def _minimum_exponent(self, index: int) -> float:
        values = [float(item["exponent"]) for item in self._signatures(index)]
        return min(values, default=np.nan)

    def _signatures(self, index: int) -> tuple[dict[str, Any], ...]:
        signatures = self._diagnostic(index, "scaling_signatures", ())
        if signatures is None:
            return ()
        if isinstance(signatures, dict):
            return (signatures,)
        if isinstance(signatures, np.ndarray):
            signatures = signatures.tolist()
        return tuple(signatures)

    @staticmethod
    def _candidate_slice(value: Any, position: int) -> Any:
        array = np.asarray(value)
        if array.ndim > 0 and len(array) > position:
            return array[position : position + 1]
        return value
