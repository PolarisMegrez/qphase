"""Shared frequency-axis orientation conventions for SDE analysers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, cast

import numpy as np
from pydantic import BeforeValidator

FrequencyOrientation = Literal["phase_decreasing", "phase_increasing"]
OrientationInput = Annotated[
    FrequencyOrientation,
    BeforeValidator(lambda value: normalize_orientation_alias(value)),
]

DEFAULT_FREQUENCY_ORIENTATION: FrequencyOrientation = "phase_decreasing"
LEGACY_FREQUENCY_ORIENTATION: FrequencyOrientation = "phase_increasing"
ORIENTATION_ALIASES: dict[str, str] = {
    "fft": "phase_increasing",
    "physical": "phase_decreasing",
}


def normalize_orientation_alias(value: Any) -> Any:
    """Normalize config-only aliases while leaving non-string validation intact."""
    if not isinstance(value, str):
        return value
    normalized = value.strip().lower()
    return ORIENTATION_ALIASES.get(normalized, normalized)


def orientation_schema_extra() -> dict[str, Any]:
    """Return machine-readable aliases for config schema consumers."""
    return {"value_aliases": dict(ORIENTATION_ALIASES)}


def resolve_frequency_orientation(
    metadata: Mapping[str, Any] | None,
) -> FrequencyOrientation:
    """Resolve result metadata, treating pre-contract results as legacy output."""
    if metadata is None or "orientation" not in metadata:
        return LEGACY_FREQUENCY_ORIENTATION
    orientation = metadata["orientation"]
    if orientation not in {"phase_decreasing", "phase_increasing"}:
        raise ValueError(f"Unsupported frequency orientation: {orientation!r}")
    return cast(FrequencyOrientation, orientation)


def orientation_sign(orientation: FrequencyOrientation) -> float:
    """Return the sign converting phase velocity to reported frequency."""
    return -1.0 if orientation == "phase_decreasing" else 1.0


def orient_spectrum(
    axis: np.ndarray,
    *spectra: np.ndarray,
    orientation: FrequencyOrientation,
) -> tuple[np.ndarray, ...]:
    """Orient a raw forward-DFT spectrum and keep its axis increasing."""
    axis_array = np.asarray(axis)
    if orientation == "phase_increasing":
        return (axis_array, *(np.asarray(values) for values in spectra))

    oriented_axis = -axis_array
    order = np.argsort(oriented_axis, kind="stable")
    return (
        oriented_axis[order],
        *(np.take(np.asarray(values), order, axis=-1) for values in spectra),
    )


def orientation_metadata(orientation: FrequencyOrientation) -> dict[str, str]:
    """Describe the positive-frequency time dependence without domain assumptions."""
    if orientation == "phase_decreasing":
        return {
            "orientation": orientation,
            "positive_frequency_time_dependence": "exp(-i * omega * t)",
            "spectrum_kernel": "exp(+i * omega * t)",
            "phase_frequency_definition": "omega = -d arg(alpha) / dt",
        }
    return {
        "orientation": orientation,
        "positive_frequency_time_dependence": "exp(+i * omega * t)",
        "spectrum_kernel": "exp(-i * omega * t)",
        "phase_frequency_definition": "omega = +d arg(alpha) / dt",
    }


__all__ = [
    "DEFAULT_FREQUENCY_ORIENTATION",
    "FrequencyOrientation",
    "LEGACY_FREQUENCY_ORIENTATION",
    "ORIENTATION_ALIASES",
    "OrientationInput",
    "orientation_metadata",
    "orientation_schema_extra",
    "orientation_sign",
    "orient_spectrum",
    "resolve_frequency_orientation",
]
