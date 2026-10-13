"""Shared frequency-axis orientation conventions for SDE analysers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, cast

import numpy as np
from pydantic import BeforeValidator

from ..contracts.quantities import (
    DEFAULT_FREQUENCY_ORIENTATION,
    FrequencyOrientation,
    SpectralProductOrientation,
)

OrientationInput = Annotated[
    FrequencyOrientation,
    BeforeValidator(lambda value: normalize_orientation_alias(value)),
]

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
    """Resolve the orientation recorded in upstream product metadata.

    Every typed spectral product records its orientation at production time,
    so a missing key is a producer bug, not a legacy result to tolerate.

    This resolver is phase-specific: one-sided spectra of real-valued signals
    (``unsigned``) carry no phase sign and are rejected here. Spectral
    consumers that also accept such products use
    :func:`resolve_spectral_orientation` instead.
    """
    if metadata is None or "orientation" not in metadata:
        raise ValueError(
            "upstream product metadata does not record a frequency orientation"
        )
    orientation = metadata["orientation"]
    if orientation not in {"phase_decreasing", "phase_increasing"}:
        raise ValueError(f"Unsupported frequency orientation: {orientation!r}")
    return cast(FrequencyOrientation, orientation)


def resolve_spectral_orientation(
    metadata: Mapping[str, Any] | None,
) -> SpectralProductOrientation:
    """Resolve a spectral product's orientation, accepting ``unsigned``.

    ``unsigned`` marks the one-sided spectrum of a real-valued signal whose
    frequency axis is non-negative by construction. It is accepted here — at
    the generic spectral-product boundary — but never by the phase-specific
    :func:`resolve_frequency_orientation`, :func:`orientation_sign`,
    :func:`orient_spectrum` or :func:`orientation_metadata` helpers.
    """
    if metadata is None or "orientation" not in metadata:
        raise ValueError(
            "upstream product metadata does not record a frequency orientation"
        )
    orientation = metadata["orientation"]
    if orientation not in {"phase_decreasing", "phase_increasing", "unsigned"}:
        raise ValueError(f"Unsupported spectral orientation: {orientation!r}")
    return cast(SpectralProductOrientation, orientation)


def _require_phase_orientation(orientation: str) -> None:
    """Guard phase-signed helpers against implicit ``else`` interpretation."""
    if orientation not in {"phase_decreasing", "phase_increasing"}:
        raise ValueError(
            f"phase-signed frequency helper does not support orientation "
            f"{orientation!r}; one-sided real-signal spectra use 'unsigned' "
            "and carry no phase sign"
        )


def orientation_sign(orientation: FrequencyOrientation) -> float:
    """Return the sign converting phase velocity to reported frequency."""
    _require_phase_orientation(orientation)
    return -1.0 if orientation == "phase_decreasing" else 1.0


def orient_spectrum(
    axis: np.ndarray,
    *spectra: np.ndarray,
    orientation: FrequencyOrientation,
) -> tuple[np.ndarray, ...]:
    """Orient a raw forward-DFT spectrum and keep its axis increasing."""
    _require_phase_orientation(orientation)
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
    _require_phase_orientation(orientation)
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
    "ORIENTATION_ALIASES",
    "OrientationInput",
    "SpectralProductOrientation",
    "orientation_metadata",
    "orientation_schema_extra",
    "orientation_sign",
    "orient_spectrum",
    "resolve_frequency_orientation",
    "resolve_spectral_orientation",
]
