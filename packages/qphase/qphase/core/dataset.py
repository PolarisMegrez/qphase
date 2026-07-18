"""Dataset result contracts and lazy point/group views."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .aggregation import AggregateResult
from .protocols import ResultProtocol


@dataclass(frozen=True)
class DatasetSaveReport:
    """Physical files written for one logical dataset."""

    layout: str
    files: tuple[Path, ...]
    loader: str | None = None
    schema_version: str = "1.0"


@runtime_checkable
class DatasetResultProtocol(ResultProtocol, Protocol):
    """Logical N-dimensional result with named scan axes."""

    @property
    def axes(self) -> Mapping[str, Any]: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    def point_view(self, index: tuple[int, ...]) -> ResultProtocol: ...

    def save_dataset(
        self,
        path: str | Path,
        *,
        layout: str,
        shard_target_bytes: int,
    ) -> DatasetSaveReport: ...


@dataclass
class MappedDatasetResult:
    """Results produced by mapping a downstream engine over dataset views."""

    results: OrderedDict[str, ResultProtocol]
    source_axes: dict[str, Any]
    view_shape: tuple[int, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def data(self) -> dict[str, ResultProtocol]:
        return dict(self.results)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.meta

    @property
    def label(self) -> Any:
        return self.meta.get("label")

    @property
    def axes(self) -> Mapping[str, Any]:
        return self.source_axes

    @property
    def shape(self) -> tuple[int, ...]:
        return self.view_shape

    def point_view(self, index: tuple[int, ...]) -> ResultProtocol:
        flat = int(np.ravel_multi_index(index, self.view_shape))
        return list(self.results.values())[flat]

    def save(self, path: str | Path) -> None:
        self.save_dataset(path, layout="single", shard_target_bytes=128 << 20)

    def save_dataset(
        self,
        path: str | Path,
        *,
        layout: str,
        shard_target_bytes: int,
    ) -> DatasetSaveReport:
        del shard_target_bytes
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        for index, (name, result) in enumerate(self.results.items()):
            result.save(root / f"{index:06d}_{_safe_name(name)}")
        files = tuple(sorted(item for item in root.rglob("*") if item.is_file()))
        return DatasetSaveReport(
            layout="per_point" if layout == "per_point" else "sharded",
            files=files,
            loader="qphase.core.dataset:MappedDatasetResult",
        )


def iter_dataset_views(
    dataset: DatasetResultProtocol,
    *,
    select: Mapping[str, Any] | None = None,
    group_by: tuple[str, ...] = (),
) -> Iterator[tuple[str, ResultProtocol]]:
    """Yield lazy point views or groups while preserving one logical job."""
    axes = list(dataset.axes)
    unknown = set(group_by) - set(axes)
    if unknown:
        raise ValueError(f"unknown group_by axes: {sorted(unknown)}")
    selected = [
        index for index in np.ndindex(dataset.shape) if _matches(dataset, index, select)
    ]
    if not group_by:
        for index in selected:
            yield _index_label(dataset, index), dataset.point_view(index)
        return

    grouped_positions = {axes.index(name) for name in group_by}
    groups: OrderedDict[tuple[Any, ...], OrderedDict[str, ResultProtocol]] = (
        OrderedDict()
    )
    for index in selected:
        key = tuple(
            _axis_value(dataset, axis_name, index, axis_position)
            for axis_position, axis_name in enumerate(axes)
            if axis_position not in grouped_positions
        )
        groups.setdefault(key, OrderedDict())[_index_label(dataset, index)] = (
            dataset.point_view(index)
        )
    for number, (key, results) in enumerate(groups.items()):
        yield (
            f"group-{number:06d}",
            AggregateResult(
                dict(results),
                meta={"group_key": key, "group_by": list(group_by)},
            ),
        )


def estimate_result_nbytes(result: ResultProtocol) -> int | None:
    """Best-effort in-memory size estimate for automatic storage layout."""
    explicit = getattr(result, "nbytes", None)
    if isinstance(explicit, int):
        return explicit
    data = getattr(result, "data", None)
    if data is not None and hasattr(data, "nbytes"):
        return int(data.nbytes)
    return None


def _matches(
    dataset: DatasetResultProtocol,
    index: tuple[int, ...],
    select: Mapping[str, Any] | None,
) -> bool:
    if not select:
        return True
    axes = list(dataset.axes)
    for name, expected in select.items():
        if name not in dataset.axes:
            raise ValueError(f"unknown selected axis {name!r}")
        position = axes.index(name)
        value = _axis_value(dataset, name, index, position)
        allowed = expected if isinstance(expected, (list, tuple, set)) else [expected]
        if not any(_equal(value, item) for item in allowed):
            return False
    return True


def _axis_value(
    dataset: DatasetResultProtocol,
    name: str,
    index: tuple[int, ...],
    position: int,
) -> Any:
    values = np.asarray(dataset.axes[name])
    if values.shape == dataset.shape:
        value = values[index]
    elif values.ndim == 1 and values.size == dataset.shape[position]:
        value = values[index[position]]
    elif values.ndim == 1 and values.size == int(np.prod(dataset.shape)):
        value = values.reshape(dataset.shape)[index]
    else:
        raise ValueError(f"axis {name!r} shape {values.shape} does not match dataset")
    return value.item() if np.asarray(value).ndim == 0 else value


def _index_label(dataset: DatasetResultProtocol, index: tuple[int, ...]) -> str:
    parts = [
        f"{name}={_axis_value(dataset, name, index, position)}"
        for position, name in enumerate(dataset.axes)
    ]
    return ",".join(parts) if parts else "point"


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return bool(np.isclose(left, right))
        except TypeError:
            pass
    return bool(left == right)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
