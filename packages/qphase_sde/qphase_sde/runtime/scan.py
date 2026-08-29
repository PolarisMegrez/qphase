"""Thin adapter from QPhase ParameterGrid to the existing fused SDE path."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qphase.backend.xputil import convert_to_numpy
from qphase.core.scan import ParameterGrid

from qphase_sde.result import SDEResult
from qphase_sde.runtime.batch import SDEResultSplitter


class SDEParameterGridAdapter:
    """Expose a logical parameter grid as one or more fused SDE tiles."""

    def __init__(self, engine: Any, grid: ParameterGrid) -> None:
        self.engine = engine
        self.grid = grid
        self.model = engine.plugins.get("model")
        if self.model is None or isinstance(self.model, dict):
            raise RuntimeError("SDE scan requires exactly one model plugin")
        self.base_n_traj = int(engine.config.n_traj)
        self.base_params = dict(self.model.params)
        self._scanned_values = self._validate_targets()
        self._had_scan_count = hasattr(engine.config, "_batch_scan_count")
        self._old_scan_count = getattr(engine.config, "_batch_scan_count", None)
        self._had_scan_offset = hasattr(engine.config, "_batch_scan_offset")
        self._old_scan_offset = getattr(engine.config, "_batch_scan_offset", None)
        configured_seed = engine.config.seed
        if configured_seed is None:
            configured_seed = int(
                np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0]
            )
        self.master_seed = int(configured_seed) % (1 << 64)

    def __enter__(self) -> SDEParameterGridAdapter:
        """Apply the legacy full-grid fused representation."""
        self._apply_range(0, self.grid.size)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Restore the original engine and model configuration."""
        del exc_type, exc, traceback
        self._restore()

    @contextmanager
    def tile(self, start: int, stop: int):
        """Temporarily expose the flat scan interval ``[start, stop)``."""
        if start < 0 or stop > self.grid.size or start >= stop:
            raise ValueError(f"invalid SDE scan tile [{start}, {stop})")
        self._apply_range(start, stop)
        try:
            yield self
        finally:
            self._restore()

    def point_seeds(self, start: int, stop: int) -> tuple[int, ...]:
        """Return deterministic per-point seeds independent of tile boundaries."""
        seeds: list[int] = []
        for flat_index in range(start, stop):
            sequence = np.random.SeedSequence([self.master_seed, flat_index])
            seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
            seeds.append(seed)
        return tuple(seeds)

    def _validate_targets(self) -> dict[str, np.ndarray]:
        expected_prefix = f"model.{self.model.name}."
        scanned: dict[str, np.ndarray] = {}
        for target, values in self.grid.target_arrays(flatten=True).items():
            if not target.startswith(expected_prefix):
                raise ValueError(
                    f"SDE scan target {target!r} must target "
                    f"{expected_prefix}<parameter>"
                )
            parameter = target.removeprefix(expected_prefix)
            if "." in parameter or parameter not in self.base_params:
                raise ValueError(f"unknown SDE model scan target {target!r}")
            scanned[parameter] = np.asarray(values)
        return scanned

    def _apply_range(self, start: int, stop: int) -> None:
        scanned = {
            name: np.repeat(values[start:stop], self.base_n_traj)
            for name, values in self._scanned_values.items()
        }
        self._replace_model_params({**self.base_params, **scanned})
        self.engine.config.n_traj = (stop - start) * self.base_n_traj
        object.__setattr__(self.engine.config, "_batch_scan_count", stop - start)
        object.__setattr__(self.engine.config, "_batch_scan_offset", start)
        object.__setattr__(
            self.engine.config,
            "_batch_scan_params",
            list(scanned),
        )

    def _restore(self) -> None:
        self.engine.config.n_traj = self.base_n_traj
        self._replace_model_params(self.base_params)
        if self._had_scan_count:
            object.__setattr__(
                self.engine.config, "_batch_scan_count", self._old_scan_count
            )
        else:
            self.engine.config.__dict__.pop("_batch_scan_count", None)
        if self._had_scan_offset:
            object.__setattr__(
                self.engine.config, "_batch_scan_offset", self._old_scan_offset
            )
        else:
            self.engine.config.__dict__.pop("_batch_scan_offset", None)
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
