"""Thin adapter from QPhase ParameterGrid to the existing fused SDE path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qphase.backend.xputil import convert_to_numpy
from qphase.core.dataset import DatasetSaveReport
from qphase.core.scan import ParameterGrid

from qphase_sde.batch import SDEResultSplitter
from qphase_sde.result import SDEResult
from qphase_sde.state import TrajectorySet


class SDEParameterGridAdapter:
    """Temporarily compile a logical grid into the established fused layout."""

    def __init__(self, engine: Any, grid: ParameterGrid) -> None:
        self.engine = engine
        self.grid = grid
        self.model = engine.plugins.get("model")
        if self.model is None or isinstance(self.model, dict):
            raise RuntimeError("SDE scan requires exactly one model plugin")
        self.base_n_traj = int(engine.config.n_traj)
        self.base_params = dict(self.model.params)
        self._had_scan_count = hasattr(engine.config, "_batch_scan_count")
        self._old_scan_count = getattr(engine.config, "_batch_scan_count", None)

    def __enter__(self) -> SDEParameterGridAdapter:
        """Apply the fused scan representation."""
        expected_prefix = f"model.{self.model.name}."
        scanned: dict[str, Any] = {}
        for target, values in self.grid.target_arrays(flatten=True).items():
            if not target.startswith(expected_prefix):
                raise ValueError(
                    f"SDE scan target {target!r} must target "
                    f"{expected_prefix}<parameter>"
                )
            parameter = target.removeprefix(expected_prefix)
            if "." in parameter or parameter not in self.base_params:
                raise ValueError(f"unknown SDE model scan target {target!r}")
            scanned[parameter] = np.repeat(values, self.base_n_traj)
        self._replace_model_params({**self.base_params, **scanned})
        self.engine.config.n_traj = self.grid.size * self.base_n_traj
        object.__setattr__(self.engine.config, "_batch_scan_count", self.grid.size)
        object.__setattr__(
            self.engine.config,
            "_batch_scan_params",
            list(scanned),
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Restore the original engine and model configuration."""
        del exc_type, exc, traceback
        self.engine.config.n_traj = self.base_n_traj
        self._replace_model_params(self.base_params)
        if self._had_scan_count:
            object.__setattr__(
                self.engine.config, "_batch_scan_count", self._old_scan_count
            )
        else:
            self.engine.config.__dict__.pop("_batch_scan_count", None)
        self.engine.config.__dict__.pop("_batch_scan_params", None)

    def _replace_model_params(self, params: dict[str, Any]) -> None:
        if hasattr(self.model, "_params"):
            self.model._params = params
            return
        current = getattr(self.model, "params", None)
        if isinstance(current, dict):
            current.clear()
            current.update(params)
            return
        raise TypeError(f"model {self.model.name!r} does not expose mutable parameters")


@dataclass
class SDEScanResult:
    """One fused SDE result exposed as a named logical scan dataset."""

    combined: SDEResult
    grid: ParameterGrid
    base_params: dict[str, Any]
    n_traj_per_point: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def data(self) -> Any:
        return self.combined.data

    @property
    def metadata(self) -> dict[str, Any]:
        return {**self.combined.metadata, **self.meta}

    @property
    def label(self) -> Any:
        return self.meta.get("label")

    @property
    def axes(self) -> dict[str, Any]:
        return dict(self.grid.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.grid.shape

    @property
    def nbytes(self) -> int:
        total = 0
        trajectory = self.combined.trajectory
        if trajectory is not None:
            total += int(convert_to_numpy(trajectory.data).nbytes)
        total += _nested_nbytes(self.combined.analysis)
        return total

    def params_at(self, index: tuple[int, ...]) -> dict[str, Any]:
        params = dict(self.base_params)
        point = self.grid.point(index)
        for target, value in point.targets.items():
            params[target.rsplit(".", 1)[-1]] = value
        return params

    def point_view(self, index: tuple[int, ...]) -> SDEResult:
        point = self.grid.point(index)
        return SDEResultSplitter.point_view(
            self.combined,
            index=point.flat_index,
            scan_count=self.grid.size,
            params=self.params_at(index),
            job_name=_point_name(point.values),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        metadata = dict(self.combined.meta)
        metadata["scan"] = {
            "combine": self.grid.combine,
            "shape": self.grid.shape,
            "axes": self.axes,
            "targets": dict(self.grid.targets),
            "base_params": self.base_params,
            "n_traj_per_point": self.n_traj_per_point,
            "dataset_meta": self.meta,
        }
        original = self.combined.meta
        self.combined.meta = metadata
        try:
            self.combined.save(path)
        finally:
            self.combined.meta = original

    def save_dataset(
        self,
        path: str | Path,
        *,
        layout: str,
        shard_target_bytes: int,
    ) -> DatasetSaveReport:
        base = Path(path)
        if layout == "single":
            self.save(base)
            output = base.with_suffix(".npz")
            return DatasetSaveReport(
                "single",
                (output,) if output.exists() else (),
                loader="qphase_sde.scan:SDEScanResult.load",
            )
        root = base
        root.mkdir(parents=True, exist_ok=True)
        if layout == "per_point":
            points_per_shard = 1
        else:
            bytes_per_point = max(self.nbytes // max(self.grid.size, 1), 1)
            points_per_shard = max(shard_target_bytes // bytes_per_point, 1)
        files: list[Path] = []
        for shard_index, start in enumerate(
            range(0, self.grid.size, points_per_shard)
        ):
            stop = min(start + points_per_shard, self.grid.size)
            shard = self._flat_slice(start, stop)
            prefix = "point" if layout == "per_point" else "shard"
            output = root / f"{prefix}_{shard_index:06d}"
            shard.save(output)
            files.append(output.with_suffix(".npz"))
        return DatasetSaveReport(
            layout,
            tuple(path for path in files if path.exists()),
            loader="qphase_sde.scan:SDEScanResult.load_dataset",
        )

    def _flat_slice(self, start: int, stop: int) -> SDEScanResult:
        trajectory = self.combined.trajectory
        sliced_trajectory = None
        if trajectory is not None:
            trajectory_start = start * self.n_traj_per_point
            trajectory_stop = stop * self.n_traj_per_point
            sliced_trajectory = TrajectorySet(
                trajectory.data[trajectory_start:trajectory_stop],
                t0=trajectory.t0,
                dt=trajectory.dt,
                meta=dict(trajectory.meta),
            )
        analysis = {
            name: value[start:stop]
            if isinstance(value, list) and len(value) == self.grid.size
            else value
            for name, value in self.combined.analysis.items()
        }
        arrays = self.grid.parameter_arrays(flatten=True)
        subgrid = ParameterGrid(
            "zipped",
            {name: values[start:stop] for name, values in arrays.items()},
            dict(self.grid.targets),
            (stop - start,),
        )
        return SDEScanResult(
            SDEResult(
                sliced_trajectory,
                analysis=analysis,
                meta=dict(self.combined.meta),
            ),
            subgrid,
            dict(self.base_params),
            self.n_traj_per_point,
            meta={"source_scan_shape": self.grid.shape, "flat_range": (start, stop)},
        )

    @classmethod
    def load(cls, path: str | Path) -> SDEScanResult:
        combined = SDEResult.load(path)
        scan = combined.meta.pop("scan")
        grid = ParameterGrid(
            scan["combine"],
            {name: np.asarray(values) for name, values in scan["axes"].items()},
            scan["targets"],
            tuple(scan["shape"]),
        )
        return cls(
            combined,
            grid,
            scan["base_params"],
            int(scan["n_traj_per_point"]),
            meta=scan.get("dataset_meta", {}),
        )

    @classmethod
    def load_dataset(cls, path: str | Path) -> SDEScanResult:
        """Load and assemble a sharded logical SDE dataset."""
        root = Path(path)
        shards = [cls.load(item) for item in sorted(root.glob("*.npz"))]
        if not shards:
            raise FileNotFoundError(f"no SDE dataset shards found under {root}")
        first = shards[0]
        trajectory_parts = [
            shard.combined.trajectory.data
            for shard in shards
            if shard.combined.trajectory is not None
        ]
        trajectory = None
        if trajectory_parts:
            source = first.combined.trajectory
            trajectory = TrajectorySet(
                np.concatenate(trajectory_parts, axis=0),
                t0=source.t0,
                dt=source.dt,
                meta=dict(source.meta),
            )
        analysis: dict[str, Any] = {}
        for name in first.combined.analysis:
            values = [shard.combined.analysis[name] for shard in shards]
            if all(isinstance(value, list) for value in values):
                analysis[name] = [item for value in values for item in value]
            else:
                analysis[name] = values[0]
        point_count = sum(shard.grid.size for shard in shards)
        source_shape = tuple(first.meta.get("source_scan_shape", (point_count,)))
        if int(np.prod(source_shape)) != point_count:
            source_shape = (point_count,)
        axes = {
            name: np.concatenate(
                [shard.grid.parameter_arrays(flatten=True)[name] for shard in shards]
            ).reshape(source_shape)
            for name in first.grid.axes
        }
        grid = ParameterGrid(
            "zipped",
            axes,
            dict(first.grid.targets),
            source_shape,
        )
        return cls(
            SDEResult(
                trajectory,
                analysis=analysis,
                meta=dict(first.combined.meta),
            ),
            grid,
            dict(first.base_params),
            first.n_traj_per_point,
        )


def _nested_nbytes(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_nested_nbytes(item) for item in value.values())
    if isinstance(value, list | tuple):
        return sum(_nested_nbytes(item) for item in value)
    try:
        return int(np.asarray(convert_to_numpy(value)).nbytes)
    except Exception:
        return 0


def _point_name(values: Any) -> str:
    return ",".join(f"{name}={value}" for name, value in values.items())
