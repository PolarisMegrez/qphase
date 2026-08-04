"""Fixed-capacity CAM result implementing QPhase's result protocol."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qphase.backend.xputil import convert_to_numpy
from qphase.core.dataset import DatasetSaveReport
from qphase.core.errors import QPhaseIOError


@dataclass
class CAMResult:
    """Serializable coherent-amplitude matrix result."""

    states: Any
    residuals: Any
    success: Any
    valid_mask: Any
    solution_count: Any
    params: dict[str, Any]
    axes: dict[str, Any] = field(default_factory=dict)
    postprocess: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    result_kind = "fixed_points"

    @property
    def data(self) -> Any:
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
    def grid_shape(self) -> tuple[int, ...]:
        """Shape of the parameter grid, excluding the solution axis."""
        return tuple(np.asarray(convert_to_numpy(self.valid_mask)).shape[:-1])

    @property
    def shape(self) -> tuple[int, ...]:
        """Dataset scan shape, excluding solution and matrix dimensions."""
        return self.grid_shape

    @property
    def nbytes(self) -> int:
        arrays = (
            self.states,
            self.residuals,
            self.success,
            self.valid_mask,
            self.solution_count,
            *self.postprocess.values(),
        )
        return sum(int(np.asarray(convert_to_numpy(value)).nbytes) for value in arrays)

    def point_view(self, index: tuple[int, ...]) -> CAMResult:
        """Return one scan point without materializing other point results."""
        if len(index) != len(self.grid_shape):
            raise IndexError(f"expected {len(self.grid_shape)} indices, got {index}")
        return CAMResult(
            states=np.asarray(convert_to_numpy(self.states))[index],
            residuals=np.asarray(convert_to_numpy(self.residuals))[index],
            success=np.asarray(convert_to_numpy(self.success))[index],
            valid_mask=np.asarray(convert_to_numpy(self.valid_mask))[index],
            solution_count=np.asarray(convert_to_numpy(self.solution_count))[index],
            params=self.params_at(index),
            axes={},
            postprocess={
                name: self._point_postprocess(value, index)
                for name, value in self.postprocess.items()
            },
            meta={**self.meta, "grid_index": index},
        )

    def params_at(self, index: tuple[int, ...]) -> dict[str, Any]:
        """Resolve model parameters for one grid point or solution index."""
        grid_index = tuple(index)
        if len(grid_index) == len(self.grid_shape) + 1:
            grid_index = grid_index[:-1]
        if len(grid_index) != len(self.grid_shape):
            raise IndexError(
                f"expected {len(self.grid_shape)} grid indices; got {grid_index}"
            )
        resolved = {
            name: self._value_at(value, grid_index)
            for name, value in self.params.items()
        }
        for name in self.axes:
            if name in resolved:
                resolved[name] = self._axis_at(name, grid_index)
        return resolved

    def _value_at(self, value: Any, grid_index: tuple[int, ...]) -> Any:
        array = np.asarray(convert_to_numpy(value))
        if self.grid_shape and array.shape[: len(self.grid_shape)] == self.grid_shape:
            selected = array[grid_index]
            return selected.item() if np.asarray(selected).ndim == 0 else selected
        return value

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(
                path,
                states=convert_to_numpy(self.states),
                residuals=convert_to_numpy(self.residuals),
                success=convert_to_numpy(self.success),
                valid_mask=convert_to_numpy(self.valid_mask),
                solution_count=convert_to_numpy(self.solution_count),
                params=np.array(self.params, dtype=object),
                axes=np.array(self.axes, dtype=object),
                postprocess=np.array(
                    {
                        key: convert_to_numpy(value)
                        for key, value in self.postprocess.items()
                    },
                    dtype=object,
                ),
                meta=np.array(self.meta, dtype=object),
            )
            self._save_csv(path.with_suffix(".csv"))
        except Exception as exc:
            raise QPhaseIOError(f"failed to save CAM result to {path}: {exc}") from exc

    def save_dataset(
        self,
        path: str | Path,
        *,
        layout: str,
        shard_target_bytes: int,
    ) -> DatasetSaveReport:
        """Save one logical CAM dataset using the selected physical layout."""
        base = Path(path)
        if layout == "single" or not self.grid_shape:
            self.save(base)
            single_files = tuple(
                item
                for item in (base.with_suffix(".npz"), base.with_suffix(".csv"))
                if item.exists()
            )
            return DatasetSaveReport(
                "single", single_files, loader="qphase_cam.result:CAMResult.load"
            )

        root = base
        root.mkdir(parents=True, exist_ok=True)
        point_count = int(np.prod(self.grid_shape))
        if layout == "per_point":
            chunks = [(index, index + 1) for index in range(point_count)]
        else:
            bytes_per_point = max(self.nbytes // max(point_count, 1), 1)
            points_per_shard = max(shard_target_bytes // bytes_per_point, 1)
            chunks = [
                (start, min(start + points_per_shard, point_count))
                for start in range(0, point_count, points_per_shard)
            ]
        files: list[Path] = []
        for shard_index, (start, stop) in enumerate(chunks):
            shard = self._flat_slice(start, stop)
            shard_path = root / f"shard_{shard_index:06d}"
            shard.save(shard_path)
            files.extend(
                item
                for item in (
                    shard_path.with_suffix(".npz"),
                    shard_path.with_suffix(".csv"),
                )
                if item.exists()
            )
        return DatasetSaveReport(
            layout,
            tuple(files),
            loader="qphase_cam.result:CAMResult.load_dataset",
        )

    def _flat_slice(self, start: int, stop: int) -> CAMResult:
        count = stop - start
        states = np.asarray(convert_to_numpy(self.states)).reshape(
            (-1,) + np.asarray(self.states).shape[-3:]
        )[start:stop]
        capacity = np.asarray(self.valid_mask).shape[-1]
        valid = np.asarray(convert_to_numpy(self.valid_mask)).reshape(-1, capacity)[
            start:stop
        ]
        residuals = np.asarray(convert_to_numpy(self.residuals)).reshape(-1, capacity)[
            start:stop
        ]
        success = np.asarray(convert_to_numpy(self.success)).reshape(-1, capacity)[
            start:stop
        ]
        counts = np.asarray(convert_to_numpy(self.solution_count)).reshape(-1)[
            start:stop
        ]
        indices = [
            tuple(int(value) for value in np.unravel_index(i, self.grid_shape))
            for i in range(start, stop)
        ]
        params = {
            name: np.asarray([self.params_at(index)[name] for index in indices])
            for name in self.params
        }
        axes = {
            name: np.asarray([self._axis_at(name, index) for index in indices])
            for name in self.axes
        }
        postprocess = {
            name: self._flat_postprocess(value, start, stop)
            for name, value in self.postprocess.items()
        }
        dataset_postprocess = tuple(
            name
            for name, value in self.postprocess.items()
            if np.asarray(convert_to_numpy(value)).shape[: len(self.grid_shape)]
            == self.grid_shape
        )
        return CAMResult(
            states.reshape((count,) + states.shape[1:]),
            residuals,
            success,
            valid,
            counts,
            params,
            axes=axes,
            postprocess=postprocess,
            meta={
                **self.meta,
                "source_grid_shape": self.grid_shape,
                "dataset_postprocess": dataset_postprocess,
            },
        )

    def _axis_at(self, name: str, index: tuple[int, ...]) -> Any:
        values = np.asarray(convert_to_numpy(self.axes[name]))
        position = list(self.axes).index(name)
        if values.shape == self.grid_shape:
            return values[index]
        if values.ndim == 1 and len(self.grid_shape) == 1:
            return values[index[0]]
        if values.ndim == 1 and values.size == self.grid_shape[position]:
            return values[index[position]]
        return values.reshape(self.grid_shape)[index]

    def _point_postprocess(self, value: Any, index: tuple[int, ...]) -> Any:
        array = np.asarray(convert_to_numpy(value))
        if array.shape[: len(self.grid_shape)] == self.grid_shape:
            return array[index]
        return value

    def _flat_postprocess(self, value: Any, start: int, stop: int) -> Any:
        array = np.asarray(convert_to_numpy(value))
        if array.shape[: len(self.grid_shape)] != self.grid_shape:
            return value
        trailing = array.shape[len(self.grid_shape) :]
        return array.reshape((-1,) + trailing)[start:stop]

    def _save_csv(self, path: Path) -> None:
        states = np.asarray(convert_to_numpy(self.states))
        residuals = np.asarray(convert_to_numpy(self.residuals))
        success = np.asarray(convert_to_numpy(self.success))
        valid = np.asarray(convert_to_numpy(self.valid_mask))
        grid_shape = valid.shape[:-1]
        parameter_names = sorted(self.params)
        state_columns = self._state_columns(states.shape[-1])
        postprocess_columns = self._postprocess_columns(valid.shape)
        fieldnames = [
            "grid_index",
            "solution_slot",
            "solution_count",
            "residual",
            "success",
            *(f"param.{name}" for name in parameter_names),
            *state_columns,
            *postprocess_columns,
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for grid_index in np.ndindex(grid_shape or ()):
                point_params = self.params_at(grid_index)
                for slot in range(valid.shape[-1]):
                    index = grid_index + (slot,)
                    if not valid[index]:
                        continue
                    row: dict[str, Any] = {
                        "grid_index": repr(grid_index),
                        "solution_slot": slot,
                        "solution_count": np.asarray(self.solution_count)[grid_index],
                        "residual": residuals[index],
                        "success": success[index],
                    }
                    row.update(
                        {
                            f"param.{name}": point_params[name]
                            for name in parameter_names
                        }
                    )
                    row.update(self._flatten_state(states[index]))
                    row.update(self._flatten_postprocess(index, valid.shape))
                    writer.writerow(row)

    @staticmethod
    def _state_columns(n_modes: int) -> list[str]:
        columns = [f"R[{i},{i}]" for i in range(n_modes)]
        for i in range(n_modes):
            for j in range(i + 1, n_modes):
                columns.extend((f"R[{i},{j}].real", f"R[{i},{j}].imag"))
        return columns

    @staticmethod
    def _flatten_state(state: np.ndarray) -> dict[str, Any]:
        n_modes = state.shape[-1]
        output: dict[str, Any] = {
            f"R[{i},{i}]": float(np.real(state[i, i])) for i in range(n_modes)
        }
        for i in range(n_modes):
            for j in range(i + 1, n_modes):
                output[f"R[{i},{j}].real"] = float(np.real(state[i, j]))
                output[f"R[{i},{j}].imag"] = float(np.imag(state[i, j]))
        return output

    def _postprocess_columns(self, result_shape: tuple[int, ...]) -> list[str]:
        columns: list[str] = []
        for name in sorted(self.postprocess):
            value = np.asarray(convert_to_numpy(self.postprocess[name]))
            if value.shape[: len(result_shape)] != result_shape:
                continue
            columns.extend(
                self._value_columns(name, value.shape[len(result_shape) :], value)
            )
        return columns

    @staticmethod
    def _value_columns(
        name: str, trailing: tuple[int, ...], value: np.ndarray
    ) -> list[str]:
        indices = list(np.ndindex(trailing or ()))
        complex_value = np.issubdtype(value.dtype, np.complexfloating)
        columns: list[str] = []
        for index in indices:
            suffix = "" if not index else "[" + ",".join(map(str, index)) + "]"
            if complex_value:
                columns.extend((f"{name}{suffix}.real", f"{name}{suffix}.imag"))
            else:
                columns.append(f"{name}{suffix}")
        return columns

    def _flatten_postprocess(
        self, index: tuple[int, ...], result_shape: tuple[int, ...]
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for name in sorted(self.postprocess):
            array = np.asarray(convert_to_numpy(self.postprocess[name]))
            if array.shape[: len(result_shape)] != result_shape:
                continue
            selected = np.asarray(array[index])
            for trailing in np.ndindex(selected.shape or ()):
                value = selected[trailing] if trailing else selected.item()
                suffix = (
                    "" if not trailing else "[" + ",".join(map(str, trailing)) + "]"
                )
                if np.iscomplexobj(value):
                    output[f"{name}{suffix}.real"] = float(np.real(value))
                    output[f"{name}{suffix}.imag"] = float(np.imag(value))
                else:
                    output[f"{name}{suffix}"] = value
        return output

    @classmethod
    def load(cls, path: str | Path) -> CAMResult:
        with np.load(path, allow_pickle=True) as data:
            return cls(
                states=data["states"],
                residuals=data["residuals"],
                success=data["success"],
                valid_mask=data["valid_mask"],
                solution_count=data["solution_count"],
                params=data["params"].item(),
                axes=data["axes"].item(),
                postprocess=data["postprocess"].item(),
                meta=data["meta"].item(),
            )

    @classmethod
    def load_dataset(cls, path: str | Path) -> CAMResult:
        """Load and reassemble a sharded logical CAM dataset."""
        root = Path(path)
        shard_paths = sorted(root.glob("shard_*.npz"))
        if not shard_paths:
            raise QPhaseIOError(f"no CAM dataset shards found in {root}")

        shards = [cls.load(shard_path) for shard_path in shard_paths]
        source_shape = tuple(shards[0].meta.get("source_grid_shape", ()))
        if not source_shape:
            raise QPhaseIOError(
                f"CAM dataset shards in {root} do not record source_grid_shape"
            )

        def combine(name: str) -> np.ndarray:
            arrays = [
                np.asarray(convert_to_numpy(getattr(shard, name))) for shard in shards
            ]
            merged = np.concatenate(arrays, axis=0)
            return merged.reshape(source_shape + merged.shape[1:])

        params = {
            name: cls._combine_shard_values(
                [shard.params[name] for shard in shards], source_shape
            )
            for name in shards[0].params
        }
        axes = {
            name: cls._combine_shard_values(
                [shard.axes[name] for shard in shards], source_shape
            )
            for name in shards[0].axes
        }
        dataset_postprocess = set(shards[0].meta.get("dataset_postprocess", ()))
        postprocess = {
            name: (
                cls._combine_shard_values(
                    [shard.postprocess[name] for shard in shards], source_shape
                )
                if name in dataset_postprocess
                else shards[0].postprocess[name]
            )
            for name in shards[0].postprocess
        }
        meta = dict(shards[0].meta)
        meta.pop("source_grid_shape", None)
        meta.pop("dataset_postprocess", None)
        return cls(
            states=combine("states"),
            residuals=combine("residuals"),
            success=combine("success"),
            valid_mask=combine("valid_mask"),
            solution_count=combine("solution_count"),
            params=params,
            axes=axes,
            postprocess=postprocess,
            meta=meta,
        )

    @staticmethod
    def _combine_shard_values(
        values: list[Any], source_shape: tuple[int, ...]
    ) -> np.ndarray:
        arrays = [np.asarray(convert_to_numpy(value)) for value in values]
        merged = np.concatenate(arrays, axis=0)
        return merged.reshape(source_shape + merged.shape[1:])
