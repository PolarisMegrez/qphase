"""qphase_sde: 1.x → 2.x Migration Contracts
---------------------------------------------------------
Freezes the migration tables from qphase_sde 1.x to 2.x:

- ``ANALYSER_MIGRATION`` maps every 1.x analyser entry point to its 2.x
  analyser, child-plugin slots and output product.
- ``ROOT_MODULE_MIGRATION`` maps every current root module to its 2.x target
  (move/split/delete). ``utils.py`` is never moved wholesale.
- ``convert_analyser_config`` is the one-shot YAML converter: it produces the
  2.x analyser mapping plus an explicit diff, unmapped fields and items
  needing human review. The 2.x runtime never accepts legacy aliases.

Public API
----------
AnalyserMigration
    One analyser-level migration entry.
ModuleMigration
    One root-module migration entry.
ConfigConversion
    Result of converting a legacy analyser mapping.
ANALYSER_MIGRATION
    The analyser migration table.
ROOT_MODULE_MIGRATION
    The root-module migration table.
convert_analyser_config
    Convert a legacy ``analyser:`` mapping to the 2.x syntax.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "ANALYSER_MIGRATION",
    "ROOT_MODULE_MIGRATION",
    "AnalyserMigration",
    "ConfigConversion",
    "ModuleMigration",
    "convert_analyser_config",
]


@dataclass(frozen=True)
class AnalyserMigration:
    """Mapping of one 1.x analyser to its 2.x analyser and child plugins."""

    legacy_name: str
    target_analyser: str
    #: Mapping of slot name to (plugin-class namespace, default plugin).
    child_slots: dict[str, tuple[str, str]]
    product: str
    analyser_keys: tuple[str, ...] = ()
    notes: str = ""


ANALYSER_MIGRATION: dict[str, AnalyserMigration] = {
    entry.legacy_name: entry
    for entry in [
        AnalyserMigration(
            "psd",
            "spectrum",
            {"estimator": ("spectral_estimator", "periodogram")},
            "spectrum",
            notes="1.x 'method' selects the spectral_estimator child.",
        ),
        AnalyserMigration(
            "dist",
            "distributions",
            {},
            "distributions",
            notes="Cartesian distribution analysis.",
        ),
        AnalyserMigration(
            "pdist",
            "distributions",
            {},
            "distributions",
            notes="Polar distribution analysis; coordinates become a parameter.",
        ),
        AnalyserMigration(
            "lorentz_fitter",
            "spectral_peaks",
            {"finder": ("peak_finder", "lorentz")},
            "spectral_peaks",
            notes="Lorentz line-shape fitting is a peak_finder child.",
        ),
        AnalyserMigration(
            "trajectory_diagnostics",
            "trajectory_diagnostics",
            {},
            "diagnostics",
        ),
        AnalyserMigration(
            "allan_variance",
            "allan_variance",
            {},
            "allan_variance",
        ),
        AnalyserMigration(
            "allan_scaling",
            "allan_scaling",
            {},
            "allan_variance",
            notes="Scaling exponents attach to the Allan product.",
        ),
        AnalyserMigration(
            "coherence_matrix",
            "coherence_matrix",
            {},
            "coherence_matrix",
        ),
        AnalyserMigration(
            "coherence_carrier",
            "coherence_frequency",
            {"estimator": ("coherence_frequency", "short_delay")},
            "coherence_frequency",
        ),
        AnalyserMigration(
            "band_limited_carrier",
            "coherence_frequency",
            {"estimator": ("coherence_frequency", "band_limited")},
            "coherence_frequency",
        ),
        AnalyserMigration(
            "finite_delay_carrier",
            "coherence_frequency",
            {"estimator": ("coherence_frequency", "finite_delay")},
            "coherence_frequency",
        ),
        AnalyserMigration(
            "spectral_ridge",
            "spectral_peaks",
            {
                "finder": ("peak_finder", "scale_space"),
                "tracker": ("peak_tracker", "topk_huber"),
            },
            "spectral_peaks",
            analyser_keys=("scan_param", "psd_key", "readouts"),
            notes="Scale-space detection and scan-path tracking split into "
            "finder and tracker child plugins.",
        ),
        AnalyserMigration(
            "moment_statistics",
            "moments",
            {},
            "moments",
        ),
        AnalyserMigration(
            "quadratic_moments",
            "moments",
            {},
            "moments",
            notes="Quadratic observable moments share the moments analyser.",
        ),
    ]
}


@dataclass(frozen=True)
class ModuleMigration:
    """Mapping of one 1.x root module to its 2.x disposition."""

    source: str
    action: Literal["move", "split", "delete"]
    target: str
    notes: str = ""


ROOT_MODULE_MIGRATION: dict[str, ModuleMigration] = {
    entry.source: entry
    for entry in [
        ModuleMigration(
            "qphase_sde.batch", "move", "qphase_sde.runtime.batch"
        ),
        ModuleMigration(
            "qphase_sde.buffers", "move", "qphase_sde.runtime.buffers"
        ),
        ModuleMigration(
            "qphase_sde.scan", "move", "qphase_sde.runtime.scan"
        ),
        ModuleMigration(
            "qphase_sde.coordinates", "move", "qphase_sde.math.coordinates"
        ),
        ModuleMigration(
            "qphase_sde.ops",
            "move",
            "qphase_sde.math.ops",
            notes="Only backend-neutral numerical definitions survive; "
            "execution helpers go to runtime.",
        ),
        ModuleMigration(
            "qphase_sde.utils",
            "split",
            "",
            notes="resolve_mode_columns moves to the analyser base helpers; "
            "the deprecated expand_complex_noise_backend wrapper is deleted; "
            "no wholesale rename.",
        ),
        ModuleMigration(
            "qphase_sde.analyser.peak_finding",
            "move",
            "qphase_sde.peak_finder",
            notes="Phase 2: child-plugin classes become root-level namespaces.",
        ),
        ModuleMigration(
            "qphase_sde.analyser.spectral_estimator",
            "move",
            "qphase_sde.spectral_estimator",
            notes="Phase 2: child-plugin classes become root-level namespaces.",
        ),
        ModuleMigration(
            "qphase_sde.analyser.frequency_orientation",
            "move",
            "qphase_sde.contracts.quantities",
            notes="Contracts hold the canonical convention; the runtime "
            "module re-exports it until removal.",
        ),
    ]
}


@dataclass
class ConfigConversion:
    """Result of converting a legacy analyser mapping to the 2.x syntax."""

    converted: dict[str, Any] = field(default_factory=dict)
    diff: list[str] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)
    needs_review: list[str] = field(default_factory=list)


#: spectral_ridge keys routed to the scale_space finder child.
_RIDGE_FINDER_KEYS = frozenset(
    {
        "freq_min",
        "freq_max",
        "maximum_profile_bins",
        "smoothing_scale_bins",
        "local_window_scale",
        "maximum_candidates",
        "minimum_scale_support",
        "minimum_prominence_fraction",
        "cluster_scale_factor",
        "confidence_sigma",
        "frequency_bin_covariance",
        "plateau_fraction",
    }
)


def _convert_psd(params: dict[str, Any], report: ConfigConversion) -> dict[str, Any]:
    params = dict(params)
    estimator_cfg: dict[str, Any]
    nested = params.pop("estimator", None)
    if isinstance(nested, dict) and nested:
        estimator_cfg = nested
        report.diff.append("psd: nested 'estimator' slot kept verbatim")
    else:
        method = params.pop("method", "periodogram")
        estimator_cfg = {method: dict(params)}
        report.diff.append(
            f"psd: method={method!r} became estimator slot "
            f"'estimator.{method}'"
        )
        if params:
            report.needs_review.append(
                "spectrum: verify estimator-level parameter split for "
                f"{sorted(params)}"
            )
    return {"spectrum": {"estimator": estimator_cfg}}


def _convert_spectral_ridge(
    params: dict[str, Any], report: ConfigConversion
) -> dict[str, Any]:
    analyser_cfg: dict[str, Any] = {}
    finder_cfg: dict[str, Any] = {}
    tracker_cfg: dict[str, Any] = {}
    for key, value in params.items():
        if key in ANALYSER_MIGRATION["spectral_ridge"].analyser_keys:
            analyser_cfg[key] = value
        elif key in _RIDGE_FINDER_KEYS:
            finder_cfg[key] = value
        elif key.startswith("tracking_"):
            tracker_cfg[key.removeprefix("tracking_")] = value
            report.needs_review.append(
                f"spectral_peaks.tracker: renamed 'tracking_' key {key!r}"
            )
        else:
            finder_cfg[key] = value
            report.needs_review.append(
                f"spectral_peaks.finder.scale_space: unrouted key {key!r} "
                "placed in finder config"
            )
    analyser_cfg["finder"] = {"scale_space": finder_cfg}
    analyser_cfg["tracker"] = {"topk_huber": tracker_cfg}
    report.diff.append(
        "spectral_ridge: split into spectral_peaks with finder.scale_space "
        "and tracker.topk_huber"
    )
    return {"spectral_peaks": analyser_cfg}


def _convert_single_slot(
    legacy_name: str,
    params: dict[str, Any],
    report: ConfigConversion,
) -> dict[str, Any]:
    entry = ANALYSER_MIGRATION[legacy_name]
    slot, (namespace, plugin) = next(iter(entry.child_slots.items()))
    del namespace  # namespace recorded in the migration table
    analyser_cfg: dict[str, Any] = {}
    child_cfg: dict[str, Any] = {}
    for key, value in params.items():
        if key in entry.analyser_keys:
            analyser_cfg[key] = value
        else:
            child_cfg[key] = value
    analyser_cfg[slot] = {plugin: child_cfg}
    report.diff.append(
        f"{legacy_name}: became {entry.target_analyser} with "
        f"{slot}.{plugin}"
    )
    if "ridge" in child_cfg or any("ridge" in k for k in child_cfg):
        report.needs_review.append(
            f"{legacy_name}: ridge conditioning now composes through typed "
            "peak inputs, not analyser imports"
        )
    return {entry.target_analyser: analyser_cfg}


def convert_analyser_config(legacy: dict[str, Any]) -> ConfigConversion:
    """Convert a 1.x ``analyser:`` mapping to the 2.x syntax.

    Returns a ``ConfigConversion`` with the converted mapping, an explicit
    human-readable diff, the names that could not be mapped and the items
    requiring human confirmation. Unknown analysers are reported as unmapped
    and never silently copied.
    """
    report = ConfigConversion()
    for name, params in (legacy or {}).items():
        params = dict(params or {})
        if name not in ANALYSER_MIGRATION:
            report.unmapped.append(name)
            continue
        entry = ANALYSER_MIGRATION[name]
        if name == "psd":
            converted = _convert_psd(params, report)
        elif name == "spectral_ridge":
            converted = _convert_spectral_ridge(params, report)
        elif entry.child_slots:
            converted = _convert_single_slot(name, params, report)
        else:
            converted = {entry.target_analyser: dict(params)}
            report.diff.append(f"{name}: renamed to {entry.target_analyser}")
        for target, cfg in converted.items():
            if target in report.converted and target != name:
                report.needs_review.append(
                    f"{name}: merges into already-converted {target!r}; "
                    "verify parameter compatibility"
                )
            existing = report.converted.setdefault(target, {})
            for key, value in cfg.items():
                if isinstance(value, dict) and isinstance(existing.get(key), dict):
                    existing[key].update(value)
                else:
                    existing[key] = value
    return report
