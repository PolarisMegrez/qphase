"""Typed bundle selection and materialization helpers for plotters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from qphase.core.dataset import DatasetResultProtocol
from qphase.data import DataKind, Dataset


class ProductNotFoundError(ValueError):
    """Raised when no typed product satisfies a plotter requirement."""


def bundle_products(data: Any) -> tuple[DatasetResultProtocol, dict[str, Dataset]]:
    """Validate the visualization input and return its typed products."""
    if not isinstance(data, DatasetResultProtocol):
        raise TypeError("visualizers require a typed DatasetResultProtocol input")
    products = getattr(data, "products", None)
    if not isinstance(products, Mapping) or not all(
        isinstance(product, Dataset) for product in products.values()
    ):
        raise TypeError("visualization input must expose named typed Dataset products")
    return data, dict(products)


def select_product(
    data: Any,
    *,
    role: str | None = None,
    kind: DataKind | None = None,
    quantity: str | None = None,
    fields: tuple[str, ...] = (),
) -> Dataset:
    """Select one product by stable role, then by typed schema facets."""
    bundle, products = bundle_products(data)
    descriptor = getattr(bundle, "bundle_descriptor", None)
    roles = getattr(descriptor, "product_roles", {})
    if role is not None and isinstance(roles, Mapping):
        name = roles.get(role)
        if isinstance(name, str):
            product = products.get(name)
            if product is None:
                raise ValueError(
                    f"bundle role {role!r} references missing product {name!r}"
                )
            if _matches(product, kind=kind, quantity=quantity, fields=fields):
                return product
            raise TypeError(f"bundle role {role!r} does not satisfy plot requirements")

    matches = [
        product
        for product in products.values()
        if _matches(product, kind=kind, quantity=quantity, fields=fields)
    ]
    if not matches:
        raise ProductNotFoundError(
            "no typed product matches the requested role or schema among "
            f"{sorted(products)}"
        )
    if len(matches) != 1:
        raise ValueError(
            "typed product selection requires exactly one match; "
            f"found {len(matches)} among {sorted(products)}"
        )
    return matches[0]


def trajectory_array(product: Dataset) -> np.ndarray:
    """Materialize trajectory amplitudes as ``(scan, trajectory, time, channel)``."""
    values = np.asarray(product.handle("alpha").materialize())
    if values.ndim == 3:
        values = values[None, ...]
    if values.ndim != 4:
        raise ValueError("trajectory alpha must have rank 3 or 4")
    return values


def time_values(product: Dataset) -> np.ndarray:
    """Return the declared regular time coordinate."""
    axis = product.axis("time")
    size = axis.size
    if size is None:
        raise ValueError("trajectory time axis must be closed")
    start = 0.0 if axis.start is None else float(axis.start)
    step = 1.0 if axis.step is None else float(axis.step)
    return start + step * np.arange(size)


def scan_values(data: Any, product: Dataset, parameter: str) -> np.ndarray:
    """Resolve one explicit scan-parameter coordinate."""
    for name in (parameter, f"parameter.{parameter}"):
        try:
            return np.asarray(product.coordinate(name), dtype=float).reshape(-1)
        except KeyError:
            pass
    variable = f"parameter.{parameter}"
    if variable in {item.name for item in product.variables}:
        values = product.handle(variable).materialize()
        return np.asarray(values, dtype=float).reshape(-1)
    bundle, _ = bundle_products(data)
    if parameter in bundle.axes:
        return np.asarray(bundle.axes[parameter], dtype=float).reshape(-1)
    raise ValueError(f"bundle has no scan coordinate {parameter!r}")


def _matches(
    product: Dataset,
    *,
    kind: DataKind | None,
    quantity: str | None,
    fields: tuple[str, ...],
) -> bool:
    if kind is not None and product.kind is not kind:
        return False
    variables = {variable.name: variable for variable in product.variables}
    if not set(fields) <= set(variables):
        return False
    return quantity is None or any(
        variable.quantity == quantity for variable in variables.values()
    )
