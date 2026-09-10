"""Typed input views for cross-result SDE analysers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qphase.data import Dataset

__all__ = ["TypedScanInput", "typed_scan_input"]


def _assign_nested(target: dict[str, Any], name: str, value: Any) -> None:
    parts = name.split(".")
    current = target
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise TypeError(f"typed product has conflicting field path {name!r}")
        current = child
    current[parts[-1]] = value


def _payload(product: Dataset, index: int | None) -> dict[str, Any]:
    point = product
    if index is not None and "scan" in {axis.name for axis in product.axes}:
        point = product.point_view(scan=index)
    payload: dict[str, Any] = {}
    for variable in point.variables:
        _assign_nested(
            payload,
            variable.name,
            np.asarray(point.handle(variable.name).materialize()),
        )
    metadata = point.attributes.get("payload_meta", {})
    if isinstance(metadata, Mapping):
        for name, value in metadata.items():
            if index is not None and isinstance(value, list) and len(value) > index:
                value = value[index]
            payload[name] = value
    if "orientation" not in payload:
        orientation = point.attributes.get("orientation")
        if orientation is not None:
            payload["orientation"] = orientation
    if "modes" not in payload:
        channel_names = point.attributes.get("channel_names")
        if isinstance(channel_names, list):
            modes = [
                int(name.removeprefix("mode_"))
                for name in channel_names
                if str(name).startswith("mode_")
            ]
            if len(modes) == len(channel_names):
                payload["modes"] = modes
    return payload


@dataclass(frozen=True)
class TypedScanPoint:
    scan_value: float
    job_name: str
    path: Path
    meta: dict[str, Any]
    analysis: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class TypedScanInput:
    """Point collection built directly from one or more typed products."""

    points: tuple[TypedScanPoint, ...]


def typed_scan_input(
    products: Mapping[str, Dataset],
    *,
    scan_param: str,
    metadata: Mapping[str, Any] | None = None,
) -> TypedScanInput:
    if not products or not all(
        isinstance(value, Dataset) for value in products.values()
    ):
        raise TypeError("typed scan input requires a mapping of Dataset products")
    primary = next(
        (
            value
            for value in products.values()
            if "scan" in {axis.name for axis in value.axes}
        ),
        next(iter(products.values())),
    )
    has_scan = "scan" in {axis.name for axis in primary.axes}
    scan_size = int(primary.axis("scan").size) if has_scan else 1
    if scan_size < 1:
        raise ValueError("typed scan input must contain at least one scan point")
    coordinate_name = f"parameter.{scan_param}"
    points: list[TypedScanPoint] = []
    for index in range(scan_size):
        point_products = {
            name: value
            for name, value in products.items()
            if "scan" not in {axis.name for axis in value.axes}
            or value.axis("scan").size == scan_size
        }
        if not point_products:
            raise ValueError("typed scan products do not share a scan axis")
        coordinate_product = next(
            (
                value
                for value in point_products.values()
                if coordinate_name in {variable.name for variable in value.variables}
            ),
            next(iter(point_products.values())),
        )
        assert coordinate_product is not None
        point = (
            coordinate_product.point_view(scan=index)
            if has_scan and "scan" in {axis.name for axis in coordinate_product.axes}
            else coordinate_product
        )
        if coordinate_name in {variable.name for variable in point.variables}:
            scan_value = float(
                np.asarray(point.handle(coordinate_name).materialize()).reshape(-1)[0]
            )
        else:
            params = metadata.get("params", {}) if metadata is not None else {}
            if scan_size != 1 or not isinstance(params, Mapping):
                raise ValueError(f"typed scan product is missing {coordinate_name!r}")
            value = params.get(scan_param)
            if value is None:
                raise ValueError(f"typed scan product is missing {coordinate_name!r}")
            scan_value = float(value)
        analysis = {
            name: _payload(
                value,
                index if "scan" in {axis.name for axis in value.axes} else None,
            )
            for name, value in point_products.items()
        }
        points.append(
            TypedScanPoint(
                scan_value=scan_value,
                job_name=f"point_{index:06d}",
                path=Path("."),
                meta={"params": {scan_param: scan_value}},
                analysis=analysis,
            )
        )
    return TypedScanInput(tuple(points))
