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
    postprocess: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    result_kind = "bifurcation_candidates"
    schema_version = "3.0"

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
        arrays_nbytes = sum(
            np.asarray(value).nbytes for value in (*arrays, *branch_arrays)
        )
        return arrays_nbytes + sum(
            self._nested_nbytes(value) for value in self.postprocess.values()
        )

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
            postprocess=self._subset_postprocess(
                self.postprocess, position, position + 1
            ),
            meta={**self.meta, "candidate_index": int(position)},
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(target, **self._npz_payload())
            self._save_csv(target.with_suffix(".csv"))
            self._save_branch_csv(target.with_name(f"{target.stem}_branches.csv"))
            self._save_response_csv(target.with_name(f"{target.stem}_responses.csv"))
            self._save_postprocess_csv(
                target.with_name(f"{target.stem}_response_summary.csv"),
                "local_response_summary",
            )
            self._save_postprocess_csv(
                target.with_name(f"{target.stem}_stochastic_validity.csv"),
                "stochastic_validity",
            )
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
            for item in (
                target.with_suffix(".npz"),
                target.with_suffix(".csv"),
                target.with_name(f"{target.stem}_branches.csv"),
                target.with_name(f"{target.stem}_responses.csv"),
                target.with_name(f"{target.stem}_response_summary.csv"),
                target.with_name(f"{target.stem}_stochastic_validity.csv"),
            )
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
            return cls._from_npz(data)

    def _save_csv(self, path: Path) -> None:
        rows = self.to_candidate_table()
        fields = self._candidate_fields()
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _save_branch_csv(self, path: Path) -> None:
        rows = self.to_branch_table()
        if not rows:
            return
        fields = [name for name in rows[0] if name != "leading_state_coefficient"] + [
            "coefficient_frobenius_norm"
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                output = {name: row[name] for name in fields[:-1]}
                output["coefficient_frobenius_norm"] = float(
                    np.linalg.norm(row["leading_state_coefficient"])
                )
                writer.writerow(output)

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
                "verification_digits": int(self.verification_digits[index]),
                "verification_status": self.verification_status[index],
                "is_physical": self._diagnostic(index, "is_physical", False),
                "is_stable": self._diagnostic(index, "is_stable", False),
                "minimum_physical_eigenvalue": self._diagnostic(
                    index, "minimum_physical_eigenvalue", np.nan
                ),
                "maximum_jacobian_real_part": self._diagnostic(
                    index, "maximum_jacobian_real_part", np.nan
                ),
                "multiplicity_residual_norm": self._diagnostic(
                    index, "multiplicity_residual_norm", np.nan
                ),
                "verified_full_residual_norm": self._diagnostic(
                    index, "verified_full_residual_norm", np.nan
                ),
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

    def subset(self, start: int, stop: int) -> CAMBifurcationResult:
        if start < 0 or stop < start or stop > len(self.states):
            raise IndexError("invalid bifurcation candidate slice")
        return CAMBifurcationResult(
            states=self.states[start:stop],
            state_vectors=self.state_vectors[start:stop],
            control_values=self.control_values[start:stop],
            control_names=self.control_names,
            full_residual_norm=self.full_residual_norm[start:stop],
            search_residual_norm=self.search_residual_norm[start:stop],
            success=self.success[start:stop],
            status=self.status[start:stop],
            method=self.method[start:stop],
            verification_digits=self.verification_digits[start:stop],
            verification_status=self.verification_status[start:stop],
            branches=self._branch_range(start, stop),
            diagnostics={
                name: np.asarray(value)[start:stop]
                for name, value in self.diagnostics.items()
            },
            postprocess=self._subset_postprocess(self.postprocess, start, stop),
            meta=dict(self.meta),
        )

    @classmethod
    def concatenate(cls, results: list[CAMBifurcationResult]) -> CAMBifurcationResult:
        if not results:
            raise ValueError("cannot concatenate an empty bifurcation result list")
        control_names = results[0].control_names
        if any(result.control_names != control_names for result in results):
            raise ValueError("bifurcation scan cases produced different controls")
        diagnostics = cls._concatenate_diagnostics(results)
        branches = cls._concatenate_branches(results)
        return cls(
            states=np.concatenate([result.states for result in results], axis=0),
            state_vectors=np.concatenate(
                [result.state_vectors for result in results], axis=0
            ),
            control_values=np.concatenate(
                [result.control_values for result in results], axis=0
            ),
            control_names=control_names,
            full_residual_norm=np.concatenate(
                [result.full_residual_norm for result in results]
            ),
            search_residual_norm=np.concatenate(
                [result.search_residual_norm for result in results]
            ),
            success=np.concatenate([result.success for result in results]),
            status=np.concatenate([result.status for result in results]),
            method=np.concatenate([result.method for result in results]),
            verification_digits=np.concatenate(
                [result.verification_digits for result in results]
            ),
            verification_status=np.concatenate(
                [result.verification_status for result in results]
            ),
            branches=branches,
            diagnostics=diagnostics,
            postprocess={},
            meta=dict(results[0].meta),
        )

    def _candidate_fields(self) -> list[str]:
        return [
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
            "minimum_physical_eigenvalue",
            "maximum_jacobian_real_part",
            "multiplicity_residual_norm",
            "verified_full_residual_norm",
            "classification_status",
            "classification_accepted",
            "signature_count",
            "minimum_exponent",
            *(str(value) for value in self.meta.get("state_ids", ())),
        ]

    def _npz_payload(self, prefix: str = "") -> dict[str, Any]:
        return {
            f"{prefix}states": self.states,
            f"{prefix}state_vectors": self.state_vectors,
            f"{prefix}control_values": self.control_values,
            f"{prefix}control_names": np.asarray(self.control_names),
            f"{prefix}full_residual_norm": self.full_residual_norm,
            f"{prefix}search_residual_norm": self.search_residual_norm,
            f"{prefix}success": self.success,
            f"{prefix}status": self.status,
            f"{prefix}method": self.method,
            f"{prefix}verification_digits": self.verification_digits,
            f"{prefix}verification_status": self.verification_status,
            f"{prefix}bifurcation_schema_version": np.asarray(self.schema_version),
            **{
                f"{prefix}{name}": value
                for name, value in self._branch_arrays().items()
            },
            f"{prefix}diagnostics": np.asarray(self.diagnostics, dtype=object),
            f"{prefix}postprocess": np.asarray(self.postprocess, dtype=object),
            f"{prefix}meta": np.asarray(self.meta, dtype=object),
        }

    @classmethod
    def _from_npz(cls, data: Any, prefix: str = "") -> CAMBifurcationResult:
        return cls(
            states=data[f"{prefix}states"],
            state_vectors=data[f"{prefix}state_vectors"],
            control_values=data[f"{prefix}control_values"],
            control_names=tuple(str(value) for value in data[f"{prefix}control_names"]),
            full_residual_norm=data[f"{prefix}full_residual_norm"],
            search_residual_norm=data[f"{prefix}search_residual_norm"],
            success=data[f"{prefix}success"],
            status=data[f"{prefix}status"],
            method=data[f"{prefix}method"],
            verification_digits=data[f"{prefix}verification_digits"],
            verification_status=data[f"{prefix}verification_status"],
            branches=cls._load_branches(data, prefix=prefix),
            diagnostics=data[f"{prefix}diagnostics"].item(),
            postprocess=(
                data[f"{prefix}postprocess"].item()
                if f"{prefix}postprocess" in data.files
                else {}
            ),
            meta=data[f"{prefix}meta"].item(),
        )

    def _save_response_csv(self, path: Path) -> None:
        self._save_postprocess_csv(path, "local_response_validation")

    def _save_postprocess_csv(self, path: Path, name: str) -> None:
        table = self.postprocess.get(name)
        if not table or "candidate_index" not in table:
            return
        # Empty tables still write their header so downstream consumers can
        # rely on a stable column schema.
        fields = tuple(table)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            size = len(table["candidate_index"])
            for index in range(size):
                writer.writerow(
                    {name: np.asarray(values)[index] for name, values in table.items()}
                )

    @staticmethod
    def _nested_nbytes(value: Any) -> int:
        if isinstance(value, dict):
            return sum(
                CAMBifurcationResult._nested_nbytes(item) for item in value.values()
            )
        return int(np.asarray(value).nbytes)

    @staticmethod
    def _subset_postprocess(
        postprocess: dict[str, Any], start: int, stop: int
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for name, value in postprocess.items():
            if not isinstance(value, dict) or "candidate_index" not in value:
                output[name] = value
                continue
            candidate_index = np.asarray(value["candidate_index"])
            mask = (candidate_index >= start) & (candidate_index < stop)
            subset = {
                field: np.asarray(values)[mask] for field, values in value.items()
            }
            subset["candidate_index"] = subset["candidate_index"] - start
            output[name] = subset
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
    def _load_branches(
        cls, data: Any, *, prefix: str = ""
    ) -> CAMBifurcationBranchTable | None:
        names = tuple(CAMBifurcationBranchTable.__dataclass_fields__)
        if not all(f"{prefix}branch_{name}" in data.files for name in names):
            return None
        return CAMBifurcationBranchTable(
            **{name: data[f"{prefix}branch_{name}"] for name in names}
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

    def _branch_range(self, start: int, stop: int) -> CAMBifurcationBranchTable | None:
        if self.branches is None:
            return None
        mask = (self.branches.candidate_index >= start) & (
            self.branches.candidate_index < stop
        )
        if not np.any(mask):
            return None
        values = {
            name: np.asarray(getattr(self.branches, name))[mask]
            for name in self.branches.__dataclass_fields__
        }
        values["candidate_index"] = values["candidate_index"] - start
        return CAMBifurcationBranchTable(**values)

    @staticmethod
    def _concatenate_diagnostics(
        results: list[CAMBifurcationResult],
    ) -> dict[str, np.ndarray]:
        names = sorted(set().union(*(result.diagnostics for result in results)))
        output: dict[str, np.ndarray] = {}
        for name in names:
            exemplar = next(
                np.asarray(result.diagnostics[name])
                for result in results
                if name in result.diagnostics
            )
            parts = []
            for result in results:
                if name in result.diagnostics:
                    parts.append(np.asarray(result.diagnostics[name]))
                    continue
                shape = (len(result.states), *exemplar.shape[1:])
                parts.append(np.full(shape, np.nan, dtype=object))
            try:
                output[name] = np.concatenate(parts, axis=0)
            except ValueError:
                packed = np.empty(
                    sum(len(result.states) for result in results), dtype=object
                )
                offset = 0
                for result in results:
                    size = len(result.states)
                    if name not in result.diagnostics:
                        packed[offset : offset + size] = np.nan
                    else:
                        values = np.asarray(result.diagnostics[name], dtype=object)
                        packed[offset : offset + size] = [
                            values[index] for index in range(size)
                        ]
                    offset += size
                output[name] = packed
        return output

    @staticmethod
    def _concatenate_branches(
        results: list[CAMBifurcationResult],
    ) -> CAMBifurcationBranchTable | None:
        rows: dict[str, list[np.ndarray]] = {
            name: [] for name in CAMBifurcationBranchTable.__dataclass_fields__
        }
        offset = 0
        for result in results:
            if result.branches is not None:
                for name in rows:
                    values = np.asarray(getattr(result.branches, name))
                    if name == "candidate_index":
                        values = values + offset
                    rows[name].append(values)
            offset += len(result.states)
        if not rows["candidate_index"]:
            return None
        return CAMBifurcationBranchTable(
            **{name: np.concatenate(parts, axis=0) for name, parts in rows.items()}
        )

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


@dataclass
class CAMBifurcationScanResult:
    """Ragged bifurcation candidates indexed by an outer parameter scan."""

    case_axes: dict[str, np.ndarray]
    case_shape: tuple[int, ...]
    case_params: dict[str, np.ndarray]
    candidate_offsets: np.ndarray
    candidates: CAMBifurcationResult
    case_metadata: tuple[dict[str, Any], ...]
    postprocess: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    result_kind = "bifurcation_scan"
    schema_version = "2.0"

    @property
    def data(self) -> np.ndarray:
        return np.diff(self.candidate_offsets).reshape(self.case_shape)

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
    def axes(self) -> dict[str, np.ndarray]:
        return self.case_axes

    @property
    def shape(self) -> tuple[int, ...]:
        return self.case_shape

    @property
    def nbytes(self) -> int:
        arrays = (
            *self.case_axes.values(),
            *self.case_params.values(),
            self.candidate_offsets,
        )
        return (
            self.candidates.nbytes
            + sum(np.asarray(value).nbytes for value in arrays)
            + sum(
                CAMBifurcationResult._nested_nbytes(value)
                for value in self.postprocess.values()
            )
        )

    def params_at(self, index: tuple[int, ...]) -> dict[str, Any]:
        flat = int(np.ravel_multi_index(index, self.case_shape))
        return {
            name: np.asarray(values).reshape(-1)[flat]
            for name, values in self.case_params.items()
        }

    def point_view(self, index: tuple[int, ...]) -> CAMBifurcationResult:
        flat = int(np.ravel_multi_index(index, self.case_shape))
        start = int(self.candidate_offsets[flat])
        stop = int(self.candidate_offsets[flat + 1])
        result = self.candidates.subset(start, stop)
        result.postprocess = CAMBifurcationResult._subset_postprocess(
            self.postprocess, start, stop
        )
        result.meta.update(
            {
                "case_index": flat,
                "case_grid_index": tuple(int(value) for value in index),
                "case_params": self.params_at(index),
                "case_search": self.case_metadata[flat],
            }
        )
        return result

    def save(self, path: str | Path) -> None:
        target = Path(path).with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bifurcation_scan_schema_version": np.asarray(self.schema_version),
            "case_axes": np.asarray(self.case_axes, dtype=object),
            "case_shape": np.asarray(self.case_shape, dtype=int),
            "case_params": np.asarray(self.case_params, dtype=object),
            "candidate_offsets": self.candidate_offsets,
            "case_metadata": np.asarray(self.case_metadata, dtype=object),
            "postprocess": np.asarray(self.postprocess, dtype=object),
            "meta": np.asarray(self.meta, dtype=object),
            **self.candidates._npz_payload(prefix="candidate_"),
        }
        try:
            np.savez_compressed(target, **payload)
            self._save_case_csv(target.with_name(f"{target.stem}_cases.csv"))
            self._save_candidate_csv(target.with_name(f"{target.stem}_candidates.csv"))
            self._save_branch_csv(target.with_name(f"{target.stem}_branches.csv"))
            self._save_response_csv(target.with_name(f"{target.stem}_responses.csv"))
            self._save_postprocess_csv(
                target.with_name(f"{target.stem}_response_summary.csv"),
                "local_response_summary",
            )
            self._save_postprocess_csv(
                target.with_name(f"{target.stem}_stochastic_validity.csv"),
                "stochastic_validity",
            )
        except Exception as exc:
            raise QPhaseIOError(
                f"failed to save CAM bifurcation scan to {target}: {exc}"
            ) from exc

    @classmethod
    def load(cls, path: str | Path) -> CAMBifurcationScanResult:
        with np.load(Path(path).with_suffix(".npz"), allow_pickle=True) as data:
            return cls(
                case_axes=data["case_axes"].item(),
                case_shape=tuple(int(value) for value in data["case_shape"]),
                case_params=data["case_params"].item(),
                candidate_offsets=data["candidate_offsets"],
                candidates=CAMBifurcationResult._from_npz(data, prefix="candidate_"),
                case_metadata=tuple(data["case_metadata"].tolist()),
                postprocess=(
                    data["postprocess"].item() if "postprocess" in data.files else {}
                ),
                meta=data["meta"].item(),
            )

    def save_dataset(
        self,
        path: str | Path,
        *,
        layout: str,
        shard_target_bytes: int,
    ) -> DatasetSaveReport:
        del layout, shard_target_bytes
        target = Path(path).with_suffix(".npz")
        self.save(target)
        files = tuple(
            item
            for item in (
                target,
                target.with_name(f"{target.stem}_cases.csv"),
                target.with_name(f"{target.stem}_candidates.csv"),
                target.with_name(f"{target.stem}_branches.csv"),
                target.with_name(f"{target.stem}_responses.csv"),
                target.with_name(f"{target.stem}_response_summary.csv"),
                target.with_name(f"{target.stem}_stochastic_validity.csv"),
            )
            if item.exists()
        )
        return DatasetSaveReport(
            layout="single",
            files=files,
            loader=("qphase_cam.bifurcation_result:CAMBifurcationScanResult.load"),
            schema_version=self.schema_version,
        )

    def _save_response_csv(self, path: Path) -> None:
        self._save_postprocess_csv(path, "local_response_validation")

    def _save_postprocess_csv(self, path: Path, name: str) -> None:
        table = self.postprocess.get(name)
        if not table or "candidate_index" not in table:
            return
        # Empty tables still write their header so downstream consumers can
        # rely on a stable column schema.
        fields = ("case", *self.case_axes, *table)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, candidate in enumerate(table["candidate_index"]):
                case = int(
                    np.searchsorted(
                        self.candidate_offsets, int(candidate), side="right"
                    )
                    - 1
                )
                row = self._case_values(case)
                row.pop("candidate_count", None)
                row.update(
                    {name: np.asarray(values)[index] for name, values in table.items()}
                )
                writer.writerow(row)

    def _case_values(self, flat: int) -> dict[str, Any]:
        index = tuple(int(value) for value in np.unravel_index(flat, self.case_shape))
        values = {"case": flat, "candidate_count": int(self.data[index])}
        axis_names = tuple(self.case_axes)
        for position, name in enumerate(axis_names):
            axis = np.asarray(self.case_axes[name])
            values[name] = axis[
                index[0] if len(self.case_shape) == 1 else index[position]
            ]
        return values

    def _save_case_csv(self, path: Path) -> None:
        fields = [
            "case",
            *self.case_axes,
            "candidate_count",
            "structural_coverage",
            "numerical_coverage",
            "singular_coverage",
            "generated_candidate_count",
            "prefilter_pass_count",
            "prefilter_rejected_count",
            "refinement_start_count",
            "refinement_duplicate_count",
            "accepted_count",
            "rejected_count",
            "control_point_count",
            "polynomial_root_count",
            "brent_interval_count",
            "fixed_point_guess_count",
            "upstream_seed_count",
            "physical_status",
            "top_rejection_reasons",
            "near_miss_saved",
            "near_miss_dropped",
            "truncation_reasons",
            "consumer_error_count",
            "consumer_errors",
            "materialization_failure_count",
            "materialization_failures",
            "result_note",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for flat, metadata in enumerate(self.case_metadata):
                row = self._case_values(flat)
                row["structural_coverage"] = metadata.get(
                    "structural_coverage", "unknown"
                )
                row["numerical_coverage"] = metadata.get(
                    "numerical_coverage", "unknown"
                )
                row["singular_coverage"] = metadata.get("singular_coverage", "unknown")
                row.update(self._case_audit_values(metadata))
                writer.writerow(row)

    @staticmethod
    def _case_audit_values(metadata: dict[str, Any]) -> dict[str, Any]:
        """Flat audit summary columns for one bifurcation scan case.

        Count columns use the ``candidate start`` unit from
        ``audit["totals"]``; workload columns (``control_point_count``,
        ``polynomial_root_count``, ``brent_interval_count``,
        ``fixed_point_guess_count``, ``upstream_seed_count``) each keep
        their own unit and are blank when the path never used that unit.
        ``truncation_reasons`` lists every reduction-search truncation
        reason plus ``seed_max_starts`` when seed generation was truncated.
        """
        audit = metadata.get("audit", {})
        totals = audit.get("totals", {})
        workload = totals.get("workload", {}) or {}
        search = metadata.get("reduction_search") or {}
        reasons = sorted(
            (totals.get("rejected_by_reason") or {}).items(),
            key=lambda item: (-item[1], item[0]),
        )
        truncation = [str(reason) for reason in search.get("truncation_reasons", ())]
        if totals.get("seed_truncated"):
            truncation.append("seed_max_starts")
        consumer_errors = tuple(search.get("consumer_errors", ()))
        materialization_failures = tuple(search.get("materialization_failures", ()))
        values: dict[str, Any] = {
            "generated_candidate_count": totals.get("generated_candidate_count", ""),
            "prefilter_pass_count": totals.get("prefilter_pass_count", ""),
            "prefilter_rejected_count": totals.get("prefilter_rejected_count", ""),
            "refinement_start_count": totals.get("refinement_start_count", ""),
            "refinement_duplicate_count": totals.get("refinement_duplicate_count", ""),
            "accepted_count": totals.get("accepted_count", ""),
            "rejected_count": totals.get(
                "rejected_count", metadata.get("rejected_count", "")
            ),
            "physical_status": audit.get(
                "physical_status", totals.get("physical_status", "")
            ),
            "top_rejection_reasons": ";".join(
                f"{reason}:{count}" for reason, count in reasons[:3]
            ),
            "near_miss_saved": totals.get("near_miss_saved", ""),
            "near_miss_dropped": totals.get("near_miss_dropped", ""),
            "truncation_reasons": ";".join(truncation),
            "consumer_error_count": search.get("consumer_error_count", ""),
            "consumer_errors": ";".join(str(error) for error in consumer_errors),
            "materialization_failure_count": search.get(
                "materialization_failure_count", ""
            ),
            "materialization_failures": ";".join(
                f"{failure.get('chart_id', 'unknown')}: {failure.get('error', '')}"
                for failure in materialization_failures
            ),
            "result_note": audit.get("result_note", ""),
        }
        for unit in (
            "control_point",
            "polynomial_root",
            "brent_interval",
            "fixed_point_guess",
            "upstream_seed",
        ):
            values[f"{unit}_count"] = workload.get(f"{unit}_count", "")
        return values

    def _save_candidate_csv(self, path: Path) -> None:
        fields = ["case", *self.case_axes, *self.candidates._candidate_fields()]
        rows = self.candidates.to_candidate_table()
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for case in range(len(self.case_metadata)):
                start = int(self.candidate_offsets[case])
                stop = int(self.candidate_offsets[case + 1])
                case_values = self._case_values(case)
                for candidate in range(start, stop):
                    row = {
                        name: case_values[name] for name in ("case", *self.case_axes)
                    }
                    row.update(rows[candidate])
                    writer.writerow(row)

    def _save_branch_csv(self, path: Path) -> None:
        rows = self.candidates.to_branch_table()
        if not rows:
            return
        fields = [
            "case",
            *self.case_axes,
            *(name for name in rows[0] if name != "leading_state_coefficient"),
            "coefficient_frobenius_norm",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                candidate = int(row["candidate_index"])
                case = int(
                    np.searchsorted(self.candidate_offsets, candidate, side="right") - 1
                )
                case_values = self._case_values(case)
                output = {name: case_values[name] for name in ("case", *self.case_axes)}
                output.update(
                    {
                        name: row[name]
                        for name in row
                        if name != "leading_state_coefficient"
                    }
                )
                output["coefficient_frobenius_norm"] = float(
                    np.linalg.norm(row["leading_state_coefficient"])
                )
                writer.writerow(output)
