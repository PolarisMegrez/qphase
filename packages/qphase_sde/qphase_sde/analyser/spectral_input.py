"""Typed spectral scan views shared by spectral postprocessors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from qphase.data import Dataset

__all__ = ["SpectralPoint", "SpectralScanInput", "spectral_scan_input"]


@dataclass(frozen=True)
class SpectralPoint:
    scan_value: float
    job_name: str
    axis: np.ndarray
    psd: np.ndarray
    psd_sem: np.ndarray | None
    modes: tuple[int, ...]
    attributes: Mapping[str, Any]
    products: Mapping[str, Dataset]

    def readout(self, readout: int | str) -> tuple[np.ndarray, np.ndarray | None, str]:
        matrix = self.psd
        sem_matrix = self.psd_sem
        if matrix.ndim == 1:
            return matrix, sem_matrix, "direct_psd_sem"
        if matrix.ndim != 2:
            raise ValueError("PSD product must be one- or two-dimensional per point")
        if readout == "trace":
            spectrum = np.sum(matrix, axis=1)
            sem = None if sem_matrix is None else np.sum(np.abs(sem_matrix), axis=1)
            return spectrum, sem, "mode_sem_upper_bound"
        mode = int(readout)
        if mode in self.modes:
            column = self.modes.index(mode)
        elif 0 <= mode < matrix.shape[1]:
            column = mode
        else:
            raise ValueError(f"mode {mode} is absent from the PSD product")
        sem = None if sem_matrix is None else sem_matrix[:, column]
        return matrix[:, column], sem, "direct_psd_sem"


@dataclass(frozen=True)
class SpectralScanInput:
    points: tuple[SpectralPoint, ...]


def _point_product(product: Dataset, index: int, scan_size: int) -> Dataset:
    if "scan" not in {axis.name for axis in product.axes}:
        return product
    if product.axis("scan").size != scan_size:
        raise ValueError("typed input products have different scan cardinality")
    return product.point_view(scan=index)


def spectral_scan_input(
    products: Mapping[str, Dataset],
    *,
    psd_key: str,
    scan_param: str,
    metadata: Mapping[str, Any] | None = None,
) -> SpectralScanInput:
    source = products.get(psd_key)
    if source is None:
        raise ValueError(f"typed input has no spectral product {psd_key!r}")
    has_scan = "scan" in {axis.name for axis in source.axes}
    raw_size = source.axis("scan").size if has_scan else 1
    if raw_size is None:
        raise ValueError("spectral product has an open scan axis")
    scan_size = int(raw_size)
    coordinate_name = f"parameter.{scan_param}"
    source_job = str((metadata or {}).get("job_name", ""))
    points: list[SpectralPoint] = []
    for index in range(scan_size):
        point_products = {
            name: _point_product(product, index, scan_size)
            for name, product in products.items()
        }
        point = point_products[psd_key]
        variables = {variable.name for variable in point.variables}
        if coordinate_name in variables:
            scan_value = float(
                np.asarray(point.handle(coordinate_name).materialize()).reshape(-1)[0]
            )
        else:
            params = (metadata or {}).get("params", {})
            if scan_size != 1 or not isinstance(params, Mapping):
                raise ValueError(f"spectral product is missing {coordinate_name!r}")
            value = params.get(scan_param)
            if value is None:
                raise ValueError(f"spectral product is missing {coordinate_name!r}")
            scan_value = float(value)
        modes = (
            tuple(
                int(value) for value in np.asarray(point.handle("modes").materialize())
            )
            if "modes" in variables
            else ()
        )
        psd_sem = (
            np.asarray(point.handle("psd_sem").materialize(), dtype=float)
            if "psd_sem" in variables
            else None
        )
        points.append(
            SpectralPoint(
                scan_value=scan_value,
                job_name=source_job,
                axis=np.asarray(
                    point.handle("axis").materialize(), dtype=float
                ).reshape(-1),
                psd=np.asarray(point.handle("psd").materialize(), dtype=float),
                psd_sem=psd_sem,
                modes=modes,
                attributes=point.attributes,
                products=point_products,
            )
        )
    return SpectralScanInput(tuple(points))
