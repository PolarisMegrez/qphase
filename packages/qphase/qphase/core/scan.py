"""Explicit parameter-scan configuration and runtime grid utilities."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import QPhaseConfigError


class LinspaceSpec(BaseModel):
    """Arguments for a linearly spaced scan axis."""

    start: float
    stop: float
    num: int = Field(ge=1)
    endpoint: bool = True
    model_config = ConfigDict(extra="forbid")


class LogspaceSpec(BaseModel):
    """Arguments for a logarithmically spaced scan axis."""

    start: float
    stop: float
    num: int = Field(ge=1)
    endpoint: bool = True
    base: float = Field(default=10.0, gt=0.0)
    model_config = ConfigDict(extra="forbid")


class ScanAxisSpec(BaseModel):
    """One named scan axis and the plugin field that it targets."""

    target: str = Field(min_length=1)
    values: list[Any] | None = None
    linspace: LinspaceSpec | None = None
    logspace: LogspaceSpec | None = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_generator(self) -> ScanAxisSpec:
        configured = sum(
            value is not None
            for value in (self.values, self.linspace, self.logspace)
        )
        if configured != 1:
            raise ValueError(
                "a scan axis must define exactly one of values, linspace, or logspace"
            )
        if self.values is not None and not self.values:
            raise ValueError("scan axis values cannot be empty")
        return self

    def generate(self) -> np.ndarray:
        """Materialize this axis as a one-dimensional NumPy array."""
        if self.values is not None:
            return np.asarray(self.values)
        if self.linspace is not None:
            linear = self.linspace
            return np.linspace(
                linear.start,
                linear.stop,
                linear.num,
                endpoint=linear.endpoint,
            )
        assert self.logspace is not None
        logarithmic = self.logspace
        return np.logspace(
            logarithmic.start,
            logarithmic.stop,
            logarithmic.num,
            endpoint=logarithmic.endpoint,
            base=logarithmic.base,
        )


class ScanSpec(BaseModel):
    """Explicit scan definition attached to one logical job."""

    combine: Literal["cartesian", "zipped"] = "cartesian"
    axes: dict[str, ScanAxisSpec]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_axes(self) -> ScanSpec:
        if not self.axes:
            raise ValueError("scan.axes cannot be empty")
        invalid = [name for name in self.axes if not name or "." in name]
        if invalid:
            raise ValueError(
                "scan axis names must be non-empty identifiers without dots: "
                + ", ".join(repr(name) for name in invalid)
            )
        targets = [axis.target for axis in self.axes.values()]
        if len(set(targets)) != len(targets):
            raise ValueError("scan axis targets must be unique")
        return self

    def compile(self) -> ParameterGrid:
        """Compile this specification into an immutable runtime grid."""
        return ParameterGrid.from_spec(self)


@dataclass(frozen=True)
class ParameterPoint:
    """One flattened point in a parameter grid."""

    flat_index: int
    index: tuple[int, ...]
    values: Mapping[str, Any]
    targets: Mapping[str, Any]


@dataclass(frozen=True)
class ParameterGrid:
    """Materialized parameter grid shared with a resource engine."""

    combine: Literal["cartesian", "zipped"]
    axes: Mapping[str, np.ndarray]
    targets: Mapping[str, str]
    shape: tuple[int, ...]

    @classmethod
    def from_spec(cls, spec: ScanSpec) -> ParameterGrid:
        axes: OrderedDict[str, np.ndarray] = OrderedDict(
            (name, np.asarray(axis.generate())) for name, axis in spec.axes.items()
        )
        targets = OrderedDict(
            (name, axis.target) for name, axis in spec.axes.items()
        )
        if spec.combine == "zipped":
            lengths = {int(values.size) for values in axes.values()}
            if len(lengths) != 1:
                detail = ", ".join(
                    f"{name}={values.size}" for name, values in axes.items()
                )
                raise QPhaseConfigError(
                    f"zipped scan axes must have equal lengths; got {detail}"
                )
            shape: tuple[int, ...] = (lengths.pop(),)
        else:
            shape = tuple(int(values.size) for values in axes.values())
        return cls(spec.combine, axes, targets, shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(self.axes)

    def parameter_arrays(self, *, flatten: bool = False) -> dict[str, np.ndarray]:
        """Return named arrays with grid shape or flattened point shape."""
        if self.combine == "zipped":
            arrays = {name: np.asarray(values) for name, values in self.axes.items()}
        else:
            meshes = np.meshgrid(*self.axes.values(), indexing="ij")
            arrays = dict(zip(self.axes, meshes, strict=True))
        if flatten:
            return {name: values.reshape(-1) for name, values in arrays.items()}
        return arrays

    def target_arrays(self, *, flatten: bool = False) -> dict[str, np.ndarray]:
        arrays = self.parameter_arrays(flatten=flatten)
        return {self.targets[name]: values for name, values in arrays.items()}

    def point(self, index: tuple[int, ...]) -> ParameterPoint:
        if len(index) != len(self.shape):
            raise IndexError(f"expected {len(self.shape)} scan indices, got {index}")
        arrays = self.parameter_arrays()
        named = {name: _scalar(values[index]) for name, values in arrays.items()}
        targeted = {self.targets[name]: value for name, value in named.items()}
        flat_index = int(np.ravel_multi_index(index, self.shape))
        return ParameterPoint(flat_index, index, named, targeted)

    def iter_points(self) -> Iterator[ParameterPoint]:
        for index in np.ndindex(self.shape):
            yield self.point(index)

    def summary(self) -> dict[str, Any]:
        return {
            "combine": self.combine,
            "shape": list(self.shape),
            "size": self.size,
            "axes": [
                {
                    "name": name,
                    "target": self.targets[name],
                    "size": int(values.size),
                }
                for name, values in self.axes.items()
            ],
        }


def execute_pointwise(
    grid: ParameterGrid,
    function: Callable[[ParameterPoint], Any],
    *,
    context: Any | None = None,
    chunk_size: int = 1,
) -> list[Any]:
    """Run a pointwise scan with optional chunk-level checkpointing."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    output: list[Any] = [None] * grid.size
    points = list(grid.iter_points())
    total_chunks = (grid.size + chunk_size - 1) // chunk_size
    for chunk_index, start in enumerate(range(0, grid.size, chunk_size)):
        stop = min(start + chunk_size, grid.size)
        key = f"scan-chunk-{chunk_index:08d}"
        cached = None
        if context is not None and context.checkpoints.enabled:
            cached = context.checkpoints.load_chunk(key)
        if cached is None:
            cached = [function(point) for point in points[start:stop]]
            if context is not None and context.checkpoints.enabled:
                context.checkpoints.save_chunk(key, cached)
        output[start:stop] = cached
        if context is not None:
            context.progress.report(
                (chunk_index + 1) / total_chunks,
                message=f"scan chunk {chunk_index + 1}/{total_chunks}",
                stage="scan",
            )
            context.cancellation.raise_if_cancelled()
    return output


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
