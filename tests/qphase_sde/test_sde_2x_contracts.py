"""Contract tests for the qphase_sde 2.0 contract package.

These tests freeze the Phase 0 contract surface: the contracts package must
stay declaration-only (no concrete plugin/model/CAM imports), the migration
tables must cover every 1.x analyser entry point, and the one-shot config
converter must produce explicit diffs, unmapped fields and review items.
"""

import ast
import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError
from qphase.data import AxisRole, ProductSchema
from qphase_sde.contracts import (
    analyser,
    bundle,
    coherence,
    migration,
    peaks,
    quantities,
    tasks,
)

SDE_ROOT = Path(__file__).resolve().parents[2] / "packages" / "qphase_sde"
CONTRACTS_DIR = SDE_ROOT / "qphase_sde" / "contracts"
MANIFEST_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "resource_manifests"
    / "sde.json"
)

#: Top-level packages the contracts package may import.
_ALLOWED_IMPORT_ROOTS = {
    "qphase",
    "qphase_sde",
    "pydantic",
    "numpy",
    "typing",
    "typing_extensions",
    "dataclasses",
    "enum",
    "collections",
    "__future__",
}


def test_package_dependency_matches_resource_core_compatibility():
    pyproject = tomllib.loads((SDE_ROOT / "pyproject.toml").read_text("utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    assert "qphase>=2.0.0a0,<3.0" in dependencies
    requirements = (SDE_ROOT / "requirements.txt").read_text("utf-8").splitlines()
    assert "qphase>=2.0.0a0,<3.0" in requirements
    assert MANIFEST_FIXTURE.exists()
    fixture = json.loads(MANIFEST_FIXTURE.read_text("utf-8"))
    assert fixture["compatibility"]["qphase_core"] == ">=2.0a0,<3.0"


def test_sde_provenance_is_resource_versioned():
    from qphase_sde.contracts.bundle import SDEProvenance

    provenance = SDEProvenance(dt=0.05)
    assert provenance.schema_version == "qphase_sde.provenance/1"
    assert provenance.dt == 0.05


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_contracts_do_not_import_cam_or_concrete_plugins():
    """Contracts import only core schema packages — never CAM or models."""
    for module in CONTRACTS_DIR.glob("*.py"):
        roots = _imported_roots(module)
        assert roots <= _ALLOWED_IMPORT_ROOTS, (
            f"{module.name} imports disallowed packages: "
            f"{sorted(roots - _ALLOWED_IMPORT_ROOTS)}"
        )
        assert "qphase_cam" not in roots
        assert "models" not in roots


def test_contracts_never_reference_concrete_sde_plugins():
    """Contract modules must not import concrete SDE plugin modules."""
    forbidden_prefixes = (
        "qphase_sde.analyser.psd",
        "qphase_sde.analyser.spectral_ridge",
        "qphase_sde.integrator.",
        "qphase_sde.engine",
        "qphase_sde.runtime",
    )
    for module in CONTRACTS_DIR.glob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(forbidden_prefixes), (
                    f"{module.name} imports {node.module}"
                )


def test_migration_table_covers_all_1x_analyser_entry_points():
    """Every 1.x analyser entry point has an explicit 2.x mapping."""
    pyproject = tomllib.loads(
        (SDE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    entry_points = pyproject["project"]["entry-points"]["qphase"]
    legacy_analysers = {
        name.split(".", 1)[1]
        for name in entry_points
        if name.startswith("analyser.")
    }
    assert legacy_analysers, "expected analyser entry points in pyproject"
    assert set(migration.ANALYSER_MIGRATION) == legacy_analysers


def test_migration_targets_use_declared_plugin_classes():
    """2.x targets only use plugin classes declared by the SDE manifest."""
    manifest = json.loads(MANIFEST_FIXTURE.read_text(encoding="utf-8"))
    declared = {pc["namespace"] for pc in manifest["plugin_classes"]}
    for entry in migration.ANALYSER_MIGRATION.values():
        assert "analyser" in declared
        for slot, (namespace, _plugin) in entry.child_slots.items():
            assert namespace in declared, (
                f"{entry.legacy_name}: slot {slot!r} targets undeclared "
                f"plugin class {namespace!r}"
            )


def test_root_module_migration_is_complete():
    """Every current root module has an explicit move/split/delete decision."""
    table = migration.ROOT_MODULE_MIGRATION
    assert table["qphase_sde.batch"].target == "qphase_sde.runtime.batch"
    assert table["qphase_sde.scan"].action == "move"
    assert table["qphase_sde.ops"].target == "qphase_sde.math.ops"
    # utils.py must never be moved wholesale.
    assert table["qphase_sde.utils"].action == "split"
    assert table["qphase_sde.utils"].target == ""
    # Nested child-plugin classes migrate to root-level namespaces.
    assert table["qphase_sde.analyser.peak_finding"].target.endswith(
        "peak_finder"
    )


def test_frequency_orientation_single_canonical_source():
    """Contracts are the canonical orientation definition; 1.x agrees."""
    from qphase_sde.analyser import frequency_orientation as legacy

    assert quantities.DEFAULT_FREQUENCY_ORIENTATION == (
        legacy.DEFAULT_FREQUENCY_ORIENTATION
    )
    assert quantities.LEGACY_FREQUENCY_ORIENTATION == (
        legacy.LEGACY_FREQUENCY_ORIENTATION
    )


def test_manifest_schema_refs_resolve():
    """Every data-product schema_ref of the SDE manifest fixture resolves."""
    import importlib

    manifest = json.loads(MANIFEST_FIXTURE.read_text(encoding="utf-8"))
    for product in manifest["data_products"]:
        module_name, _, attr = product["schema_ref"].partition(":")
        module = importlib.import_module(module_name)
        schema = getattr(module, attr)
        assert isinstance(schema, ProductSchema), product["name"]
        assert schema.kind.value == product["kind"]


def test_product_templates_are_open_and_typed():
    """Product templates are plan-time-open with the declared data kind."""
    assert not quantities.SPECTRUM_PRODUCT.is_closed
    assert quantities.SPECTRUM_PRODUCT.kind.value == "spectral"
    assert bundle.TRAJECTORY_PRODUCT.kind.value == "time_series"
    assert peaks.PEAK_PRODUCT.kind.value == "statistics"
    assert coherence.COHERENCE_FREQUENCY_PRODUCT.kind.value == "statistics"


def test_peak_candidate_and_path_roundtrip():
    """Peak candidates carry explicit uncertainty scope and no path data."""
    candidate = peaks.PeakCandidate(
        location=1.25,
        intensity=42.0,
        conditional_location_std=0.01,
        sampling_location_std=0.02,
        confidence_lower=1.2,
        confidence_upper=1.3,
        width=0.05,
        prominence=0.9,
        curvature=-3.0,
        support=16.0,
        quality=0.98,
    )
    dumped = candidate.model_dump()
    assert "path" not in dumped
    assert "fit_payload" not in dumped  # fit payloads are not frozen
    payload = json.loads(json.dumps(candidate.model_dump(mode="json")))
    assert peaks.PeakCandidate.model_validate(payload) == candidate

    path = peaks.PeakPathResult(
        path_id="p0",
        candidate_rows=[0, 3, 7],
        scan_positions=[0, 1, 2],
        uncertainty_scope=peaks.UncertaintyScope.PATH_MODEL_SELECTION,
    )
    assert path.uncertainty_scope.value == "path_model_selection"

    with pytest.raises(ValidationError):
        peaks.PeakCandidate(location=1.0, intensity=2.0, bogus_field=True)
    with pytest.raises(ValidationError):
        peaks.PeakCandidate(
            location=1.0,
            intensity=2.0,
            fit_payload={"model": "lorentz"},
        )


def test_coherence_frequency_estimate_roundtrip():
    """Coherence estimates carry lag/bandwidth/model diagnostics explicitly."""
    estimate = coherence.CoherenceFrequencyEstimate(
        frequency=-0.42,
        bandwidth=0.05,
        lag=12.0,
        estimator="band_limited",
        sampling_std=0.01,
        independent_count=64,
        diagnostics={"platform": "flat", "rayleigh": 0.001},
    )
    payload = json.loads(json.dumps(estimate.model_dump(mode="json")))
    assert coherence.CoherenceFrequencyEstimate.model_validate(payload) == (
        estimate
    )


def test_analyser_contract_is_structural():
    """A minimal structural implementation satisfies the analyser contract."""

    class FakeAnalyser:
        def input_requirements(self):
            return []

        def output_spec(self):
            return None

        def execution_capabilities(self):
            return analyser.AnalyserExecutionCapabilities(
                preferred_location="backend",
                supports_trajectory_batching=True,
                deterministic_merge=True,
            )

        def workspace(self, request):
            return analyser.AnalyserWorkspaceEstimate(device_bytes=1024)

        def work_estimate(self, request):
            return analyser.WorkEstimate(unit="transform", total=None)

        def reducer(self):
            return None

    instance = FakeAnalyser()
    assert isinstance(instance, analyser.AnalyserContract)
    estimate = instance.work_estimate(None)
    assert estimate.total is None  # heartbeat-only stage, no fake ETA


def test_convert_psd_config_to_estimator_slot():
    """Legacy psd method becomes the estimator child slot."""
    report = migration.convert_analyser_config(
        {"psd": {"method": "welch", "segment_length": 512, "overlap": 0.5}}
    )
    assert report.converted == {
        "spectrum": {
            "estimator": {"welch": {"segment_length": 512, "overlap": 0.5}}
        }
    }
    assert report.unmapped == []
    assert any("welch" in line for line in report.diff)
    assert report.needs_review  # parameter split needs human confirmation


def test_convert_spectral_ridge_splits_finder_and_tracker():
    """spectral_ridge splits into scale_space finder + topk_huber tracker."""
    report = migration.convert_analyser_config(
        {
            "spectral_ridge": {
                "scan_param": "delta",
                "psd_key": "psd",
                "smoothing_scale_bins": [2.0, 4.0],
                "tracking_enabled": True,
            }
        }
    )
    peaks_cfg = report.converted["spectral_peaks"]
    assert peaks_cfg["scan_param"] == "delta"
    assert peaks_cfg["finder"]["scale_space"] == {
        "smoothing_scale_bins": [2.0, 4.0]
    }
    assert peaks_cfg["tracker"]["topk_huber"] == {"enabled": True}
    assert any("tracking_enabled" in item for item in report.needs_review)


def test_convert_coherence_carriers_to_estimator_slot():
    """All three carriers become coherence_frequency estimator children."""
    report = migration.convert_analyser_config(
        {
            "coherence_carrier": {"lag": 4.0},
            "band_limited_carrier": {"min_points": 8},
            "finite_delay_carrier": {"delay": 2.0},
        }
    )
    estimators = report.converted["coherence_frequency"]["estimator"]
    assert set(estimators) == {"short_delay", "band_limited", "finite_delay"}
    # Three legacy analysers merge into one target — review is required.
    assert report.needs_review


def test_convert_unknown_analyser_is_unmapped_not_copied():
    """Unknown analysers are reported, never silently copied."""
    report = migration.convert_analyser_config({"mystery": {"x": 1}})
    assert report.unmapped == ["mystery"]
    assert "mystery" not in report.converted


def test_peak_candidate_offsets_contract():
    """Candidate offsets: start 0, monotone, terminal count, scan+1 length."""
    import numpy as np

    scan_count = 3
    columns = {
        "location": np.zeros(4),
        "intensity": np.zeros(4),
        "status_code": np.zeros(4, dtype=np.int64),
    }
    peaks.validate_candidate_table([0, 2, 2, 4], scan_count, columns)

    with pytest.raises(ValueError, match="start at 0"):
        peaks.validate_candidate_table([1, 2, 2, 4], scan_count, columns)
    with pytest.raises(ValueError, match="non-decreasing"):
        peaks.validate_candidate_table([0, 3, 2, 4], scan_count, columns)
    with pytest.raises(ValueError, match=r"scan_count \+ 1"):
        peaks.validate_candidate_table([0, 2, 4], scan_count, columns)
    with pytest.raises(ValueError, match="one entry per candidate"):
        # Terminal offset (4) disagrees with the column length (3).
        peaks.validate_candidate_table(
            [0, 2, 2, 4],
            scan_count,
            {"location": np.zeros(3), "intensity": np.zeros(3),
             "status_code": np.zeros(3, dtype=np.int64)},
        )
    with pytest.raises(ValueError, match="required column"):
        peaks.validate_candidate_table([0, 2, 2, 4], scan_count, {})


def test_optional_capability_requires_valid_mask():
    """Optional columns need an explicit 0/1 '<name>_valid' mask column."""
    import numpy as np

    columns = {
        "location": np.zeros(2),
        "intensity": np.zeros(2),
        "status_code": np.zeros(2, dtype=np.int64),
        "width": np.array([0.1, 0.2]),
        "width_valid": np.array([1, 0], dtype=np.int8),
    }
    peaks.validate_candidate_table([0, 2], 1, columns)

    with pytest.raises(ValueError, match="mask column"):
        peaks.validate_candidate_table(
            [0, 2], 1, {k: v for k, v in columns.items() if k != "width_valid"}
        )
    with pytest.raises(ValueError, match="0/1"):
        peaks.validate_candidate_table(
            [0, 2],
            1,
            {**columns, "width_valid": np.array([1, 7], dtype=np.int8)},
        )


def test_confidence_bounds_ordering():
    """Confidence lower bounds must never exceed upper bounds."""
    import numpy as np

    peaks.validate_confidence_bounds([1.0, 2.0], [1.5, 2.5])
    with pytest.raises(ValueError, match="lower bound exceeds upper"):
        peaks.validate_confidence_bounds([1.0, 3.0], [1.5, 2.5])
    # Masked-out entries are not checked.
    peaks.validate_confidence_bounds(
        np.array([1.0, 3.0]),
        np.array([1.5, 2.5]),
        valid_mask=np.array([1, 0], dtype=np.int8),
    )


def test_path_table_validation():
    """Path offsets, candidate rows and scan positions stay consistent."""
    peaks.validate_path_table(
        path_offsets=[0, 2, 5],
        candidate_row=[0, 3, 1, 2, 4],
        scan_position=[0, 1, 0, 1, 2],
        candidate_count=5,
    )
    with pytest.raises(ValueError, match="equal length"):
        peaks.validate_path_table([0, 2], [0, 3], [0], 5)
    with pytest.raises(ValueError, match="terminal path offset"):
        peaks.validate_path_table([0, 3], [0, 3], [0, 1], 5)
    with pytest.raises(ValueError, match="outside"):
        peaks.validate_path_table([0, 2], [0, 9], [0, 1], 5)


def test_fit_parameter_table_has_no_object_payload():
    """Fit parameters are typed numeric rows; names live in attributes."""
    import numpy as np

    fit = peaks.PEAK_FIT_PRODUCT
    for variable in fit.variables:
        dtype = np.dtype(variable.dtype)
        assert dtype != np.dtype(object)
        assert dtype.kind in "if", variable.name
    # code → name mapping is an attribute list, not a string column.
    assert "parameter_names" in fit.attributes
    assert fit.attributes["foreign_key"] == "candidate_row"


def test_uncertainty_scopes_are_typed_and_roundtrip():
    """conditional/sampling/path_model_selection scopes are typed entries."""
    peak_scopes = {u.scope for u in peaks.PEAK_PRODUCT.uncertainties}
    assert peak_scopes == {"conditional", "sampling"}
    # Scope references resolve to typed variables, not metadata dicts.
    data_vars = {u.data_variable for u in peaks.PEAK_PRODUCT.uncertainties}
    variable_names = {v.name for v in peaks.PEAK_PRODUCT.variables}
    assert data_vars <= variable_names

    for scope in peaks.UncertaintyScope:
        assert scope.value in {
            "conditional",
            "sampling",
            "path_model_selection",
        }
    assert peaks.PEAK_PATH_PRODUCT.attributes["uncertainty_scope"] == (
        "path_model_selection"
    )

    payload = json.loads(json.dumps(peaks.PEAK_PRODUCT.model_dump(mode="json")))
    assert ProductSchema.model_validate(payload) == peaks.PEAK_PRODUCT


def test_moment_family_descriptor_is_sde_private_and_explicit():
    """Moment descriptors are SDE-private with explicit positive orders."""
    descriptor = quantities.SDEMomentFamilySchema(
        family_id="f", moment_kind="central", ordering="normal", orders=[2, 1]
    )
    assert descriptor.orders == [1, 2]  # stored deterministically
    assert descriptor.order_axis == "order"
    with pytest.raises(ValidationError, match="positive"):
        quantities.SDEMomentFamilySchema(
            family_id="f", moment_kind="raw", ordering="c_number", orders=[0]
        )
    with pytest.raises(ValidationError, match="unique"):
        quantities.SDEMomentFamilySchema(
            family_id="f", moment_kind="raw", ordering="c_number",
            orders=[1, 1],
        )

    # The core schema has no moment_family field; the descriptor is embedded
    # as JSON attributes of the SDE product.
    assert "moment_family" not in ProductSchema.model_fields
    embedded = quantities.MOMENT_FAMILY_PRODUCT.attributes["moment_family"]
    reparsed = quantities.SDEMomentFamilySchema.model_validate(embedded)
    assert reparsed == quantities.DEFAULT_MOMENT_FAMILY

    # Only a single explicit 'order' index axis is frozen — no mixed-rank
    # tensors disguised as one dense variable.
    moment = quantities.MOMENT_FAMILY_PRODUCT.variable("moment")
    assert moment.dims == ("scan", "order", "channel")
    order_axis = quantities.MOMENT_FAMILY_PRODUCT.axis("order")
    assert order_axis.role is AxisRole.INDEX


def test_uncertainties_use_reduced_sampling_bases_not_scan_axes():
    """Aggregated products retain scan but describe trajectories as a basis."""
    for schema in (
        quantities.SPECTRUM_PRODUCT,
        quantities.ALLAN_PRODUCT,
        quantities.MOMENT_FAMILY_PRODUCT,
        coherence.COHERENCE_FREQUENCY_PRODUCT,
    ):
        scan_axis = schema.axis("scan")
        assert scan_axis.role is AxisRole.PARAMETER
        assert "trajectory" not in {axis.name for axis in schema.axes}
        assert {basis.name for basis in schema.sampling_bases} == {"trajectory"}
        for uncertainty in schema.uncertainties:
            if uncertainty.scope == "sampling":
                assert uncertainty.sampling_basis == "trajectory"


def test_sde_task_profiles_are_frozen():
    """The three SDE task profiles match the frozen design."""
    assert {p.id for p in tasks.SDE_TASK_PROFILES} == {
        "simulate",
        "analyze",
        "simulate_analyze",
    }

    simulate = tasks.sde_task_profile("simulate")
    assert simulate.requirements.required == ["backend", "integrator", "model"]
    assert simulate.requirements.optional == ["analyser", "observer"]

    analyze = tasks.sde_task_profile("analyze")
    assert analyze.requirements.required == ["analyser", "backend"]
    assert analyze.requirements.forbidden == [
        "integrator",
        "model",
        "observer",
    ]

    both = tasks.sde_task_profile("simulate_analyze")
    assert both.requirements.required == [
        "analyser",
        "backend",
        "integrator",
        "model",
    ]
    assert both.requirements.optional == ["observer"]

    with pytest.raises(KeyError):
        tasks.sde_task_profile("unknown-task")

    # Profiles stay JSON-serializable and round-trip losslessly.
    payload = json.loads(json.dumps(analyze.model_dump(mode="json")))
    from qphase.core.task_profile import EngineTaskProfile

    assert EngineTaskProfile.model_validate(payload) == analyze
