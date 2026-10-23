"""qphase_sde: Analyser Config Expansion
---------------------------------------------------------
The SDE engine's compiled products form one graph, so an analyser that needs
another analyser's output declares a requirement and consumes a typed product.
Some configurations are still authored as a nested analyser config (for
example ``band_limited_carrier.center.spectral_ridge``); this module promotes
those nested configs into sibling analyser instances before validation, graph
compilation and snapshotting.

Public API
----------
expand_analyser_config
    Rewrite the merged analyser section into its graph-expanded form.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .analyser.spectral_ridge import SpectralRidgeConfig

__all__ = ["RIDGE_LABEL_SUFFIX", "expand_analyser_config"]

#: Label suffix of the spectral-ridge analyser expanded from a carrier's
#: ridge-conditioned center configuration.
RIDGE_LABEL_SUFFIX = ".__ridge"

_CARRIER_PLUGIN = "band_limited_carrier"
_RIDGE_PLUGIN = "spectral_ridge"


def expand_analyser_config(analysers: Mapping[str, Any]) -> dict[str, Any]:
    """Return the analyser section with nested analyser configs promoted.

    The transform is deterministic and idempotent: an entry whose nested
    configuration has already been promoted passes through unchanged, and a
    carrier without ``center.spectral_ridge`` is left untouched.
    """
    expanded: dict[str, Any] = {}
    promoted: dict[str, Any] = {}
    for label, entry in analysers.items():
        expanded[label] = entry
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("plugin", "")) != _CARRIER_PLUGIN:
            continue
        carrier_config = entry.get("config")
        if not isinstance(carrier_config, Mapping):
            continue
        center = carrier_config.get("center")
        if not isinstance(center, Mapping):
            continue
        requested = center.get("spectral_ridge")
        if not isinstance(requested, Mapping):
            continue

        ridge_label = f"{label}{RIDGE_LABEL_SUFFIX}"
        if ridge_label in analysers:
            raise ValueError(
                f"analyser label {ridge_label!r} collides with the spectral-ridge "
                f"analyser expanded for {label!r}"
            )
        promoted[ridge_label] = {
            "plugin": _RIDGE_PLUGIN,
            "config": _resolve_ridge_config(requested, carrier_config),
        }
        carrier_entry = dict(entry)
        carrier_config = dict(carrier_config)
        center = dict(center)
        center.pop("spectral_ridge", None)
        center["ridge_label"] = ridge_label
        carrier_config["center"] = center
        carrier_entry["config"] = carrier_config
        expanded[label] = carrier_entry
    expanded.update(promoted)
    return expanded


def _resolve_ridge_config(
    requested: Mapping[str, Any], carrier_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve the ridge config against the carrier's scan/readout settings.

    The carrier drives the ridge over its own scan axis, PSD product and
    readout, and its explicit frequency bounds take precedence over the
    nested ones. Field defaults are left to ``SpectralRidgeConfig``.
    """
    resolved = dict(requested)
    resolved["scan_param"] = carrier_config["scan_param"]
    resolved["psd_key"] = carrier_config.get("psd_key", "psd")
    resolved["readouts"] = [carrier_config.get("readout", "trace")]
    freq_min = carrier_config.get("freq_min")
    if freq_min is not None:
        resolved["freq_min"] = freq_min
    freq_max = carrier_config.get("freq_max")
    if freq_max is not None:
        resolved["freq_max"] = freq_max
    return SpectralRidgeConfig.model_validate(resolved).model_dump()
