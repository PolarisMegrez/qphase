"""Fixed-capacity CAM result implementing QPhase's result protocol."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qphase.backend.xputil import convert_to_numpy
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
        for name, values in self.axes.items():
            if name in resolved:
                resolved[name] = self._value_at(values, grid_index)
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
