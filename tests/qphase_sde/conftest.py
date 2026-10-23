"""Shared helpers for the qphase_sde analyser test suite."""

from __future__ import annotations

from typing import Any

import numpy as np


def product_payload(product: Any) -> dict[str, Any]:
    """Flatten one typed analyser product into its nested payload mapping.

    The product's ``payload_meta`` leaves and every materialized variable are
    merged, then dotted names are re-nested into dicts. Digit-only path
    segments become integer keys, so per-mode tables land under their mode
    index and scalar leaves arrive as Python scalars.
    """
    payload: dict[str, Any] = {}
    values = list(product.attributes.get("payload_meta", {}).items()) + [
        (variable.name, product.handle(variable.name).materialize())
        for variable in product.variables
    ]
    for name, value in values:
        if np.asarray(value).ndim == 0:
            value = np.asarray(value).item()
        target = payload
        parts = name.split(".")
        for part in parts[:-1]:
            key = int(part) if part.isdigit() else part
            target = target.setdefault(key, {})
        target[parts[-1]] = value
    return payload
