"""Thin adapter from QPhase ParameterGrid to the existing fused SDE path."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
from qphase.core.scan import ParameterGrid


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

    @contextmanager
    def trajectory_tile(
        self,
        scan_start: int,
        scan_stop: int,
        trajectory_start: int,
        trajectory_stop: int,
    ):
        """Expose one trajectory batch for every point in a scan tile.

        The fused array is point-major: each scan point contributes the same
        trajectory slice. This is the layout consumed by ``trajectory_product``
        and keeps the existing model kernels unchanged.
        """
        if not (0 <= scan_start < scan_stop <= self.grid.size):
            raise ValueError(f"invalid SDE scan tile [{scan_start}, {scan_stop})")
        if not (0 <= trajectory_start < trajectory_stop <= self.base_n_traj):
            raise ValueError(
                f"invalid trajectory batch [{trajectory_start}, {trajectory_stop})"
            )
        point_count = scan_stop - scan_start
        batch_count = trajectory_stop - trajectory_start
        original_n_traj = int(self.engine.config.n_traj)
        original_params = dict(self.model.params)
        original_attrs = {
            name: getattr(self.engine.config, name, None)
            for name in (
                "_batch_scan_count",
                "_batch_scan_offset",
                "_batch_scan_params",
            )
        }
        scanned = {
            name: np.repeat(values[scan_start:scan_stop], batch_count)
            for name, values in self._scanned_values.items()
        }
        self._replace_model_params({**self.base_params, **scanned})
        self.engine.config.n_traj = point_count * batch_count
        object.__setattr__(self.engine.config, "_batch_scan_count", point_count)
        object.__setattr__(self.engine.config, "_batch_scan_offset", scan_start)
        object.__setattr__(
            self.engine.config,
            "_batch_scan_params",
            list(scanned),
        )
        try:
            yield self
        finally:
            self.engine.config.n_traj = original_n_traj
            self._replace_model_params(original_params)
            for name, value in original_attrs.items():
                if value is None and name in self.engine.config.__dict__:
                    self.engine.config.__dict__.pop(name, None)
                else:
                    object.__setattr__(self.engine.config, name, value)

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
