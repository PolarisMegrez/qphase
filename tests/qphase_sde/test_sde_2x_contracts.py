"""Contract tests for the qphase_sde 2.0 contract package.

These tests freeze the Phase 0 contract surface: the contracts package must
stay declaration-only (no concrete plugin/model/CAM imports).
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
    peaks,
    quantities,
    tasks,
)

SDE_ROOT = Path(__file__).resolve().parents[2] / "packages" / "qphase_sde"
CONTRACTS_DIR = SDE_ROOT / "qphase_sde" / "contracts"
MANIFEST_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "resource_manifests" / "sde.json"
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
    # The default candidate status asserts computation only; center
    # identifiability is opt-in and PATH_UNIQUE belongs to path products.
    assert candidate.status is peaks.PeakStatus.COMPUTED
    payload = json.loads(json.dumps(candidate.model_dump(mode="json")))
    assert payload["status"] == int(peaks.PeakStatus.COMPUTED)
    assert peaks.PeakCandidate.model_validate(payload) == candidate

    path = peaks.PeakPathResult(
        path_id="p0",
        candidate_rows=[0, 3, 7],
        scan_positions=[0, 1, 2],
        uncertainty_scope=peaks.UncertaintyScope.PATH_MODEL_SELECTION,
    )
    assert path.uncertainty_scope.value == "path_model_selection"
    assert path.status is peaks.PeakStatus.COMPUTED

    with pytest.raises(ValidationError):
        peaks.PeakCandidate(location=1.0, intensity=2.0, bogus_field=True)
    with pytest.raises(ValidationError):
        peaks.PeakCandidate(
            location=1.0,
            intensity=2.0,
            fit_payload={"model": "lorentz"},
        )


def test_coherence_frequency_estimate_roundtrip():
    """Coherence estimates carry lag/bandwidth/status explicitly."""
    estimate = coherence.CoherenceFrequencyEstimate(
        frequency=-0.42,
        bandwidth=0.05,
        lag=12.0,
        estimator="band_limited",
        frequency_sem=0.01,
        independent_count=64,
        status=(
            coherence.CoherenceFrequencyStatus.COMPUTED
            | coherence.CoherenceFrequencyStatus.UNIQUE
        ),
        diagnostics={"platform": "flat", "rayleigh": 0.001},
    )
    payload = json.loads(json.dumps(estimate.model_dump(mode="json")))
    assert coherence.CoherenceFrequencyEstimate.model_validate(payload) == (estimate)

    # The default status asserts computation only; uniqueness and window
    # selection are asserted explicitly once established.
    default = coherence.CoherenceFrequencyEstimate(
        frequency=0.0, estimator="short_delay"
    )
    assert default.status is coherence.CoherenceFrequencyStatus.COMPUTED
    with pytest.raises(ValidationError, match="CoherenceFrequencyStatus mask"):
        coherence.CoherenceFrequencyEstimate(
            frequency=0.0, estimator="short_delay", status=8
        )


def test_coherence_frequency_product_declares_explicit_estimate_fields():
    """Frequency + uncertainty payload, lag, bandwidth and status_code are
    explicit schema variables, grounded in what the coherence estimators
    produce; the sampling uncertainty closes on ``frequency_sem``.
    """
    import numpy as np

    schema = coherence.COHERENCE_FREQUENCY_PRODUCT
    variables = {variable.name: variable for variable in schema.variables}
    assert set(variables) == {
        "frequency",
        "frequency_sem",
        "lag",
        "bandwidth",
        "status_code",
    }
    for name, variable in variables.items():
        assert variable.dims == ("scan", "channel"), name
    assert np.dtype(variables["status_code"].dtype).kind == "i"
    assert variables["frequency"].quantity == "coherence_frequency"

    (uncertainty,) = schema.uncertainties
    assert uncertainty.target == "frequency"
    assert uncertainty.data_variable == "frequency_sem"
    assert uncertainty.sampling_basis == "trajectory"
    assert uncertainty.scope == "sampling"

    # status_bits decodes codes without importing this module.
    assert schema.attributes["status_bits"] == {
        "computed": 1,
        "unique": 2,
        "interior_window": 4,
    }
    # The template stays closed under core validation and JSON roundtrip.
    payload = json.loads(json.dumps(schema.model_dump(mode="json")))
    assert ProductSchema.model_validate(payload) == schema


def test_coherence_frequency_table_validation():
    """The table validator checks required columns, cells and status mask."""
    import numpy as np

    columns = {
        "frequency": np.zeros((2, 2)),
        "frequency_sem": np.full((2, 2), 0.01),
        "lag": np.full((2, 2), 12.0),
        "bandwidth": np.full((2, 2), 0.5),
        "status_code": np.array([[7, 5], [1, 0]], dtype=np.int64),
    }
    coherence.validate_coherence_frequency_table(columns, scan_count=2, channel_count=2)

    with pytest.raises(ValueError, match="required column"):
        coherence.validate_coherence_frequency_table(
            {k: v for k, v in columns.items() if k != "lag"},
            scan_count=2,
            channel_count=2,
        )
    with pytest.raises(ValueError, match=r"per \(scan, channel\) cell"):
        coherence.validate_coherence_frequency_table(
            {**columns, "frequency": np.zeros(4)},
            scan_count=2,
            channel_count=2,
        )
    with pytest.raises(ValueError, match="CoherenceFrequencyStatus mask"):
        coherence.validate_coherence_frequency_table(
            {**columns, "status_code": np.full((2, 2), 8, dtype=np.int64)},
            scan_count=2,
            channel_count=2,
        )
    with pytest.raises(ValueError, match="integer dtype"):
        coherence.validate_coherence_frequency_table(
            {**columns, "status_code": np.zeros((2, 2))},
            scan_count=2,
            channel_count=2,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        coherence.validate_coherence_frequency_table(
            {**columns, "bandwidth": np.full((2, 2), -0.5)},
            scan_count=2,
            channel_count=2,
        )


def test_analyser_contract_is_structural():
    """A minimal structural implementation satisfies the analyser contract."""

    class FakeAnalyser:
        def input_requirements(self):
            return []

        def output_specs(self, *, label):
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
            family_id="f",
            moment_kind="raw",
            ordering="c_number",
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
    assert analyze.requirements.required == ["analyser"]
    assert analyze.requirements.optional == ["backend"]
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


def test_select_sde_task_profile_mapping():
    """User-facing ``mode`` plus analyser presence select the task profile."""
    # simulate (default mode) without analysers.
    assert tasks.select_sde_task_profile({"engine": {"sde": {}}}) == "simulate"
    assert tasks.select_sde_task_profile({}) == "simulate"
    # Empty analyser sections do not count as configured.
    assert (
        tasks.select_sde_task_profile({"engine": {"sde": {}}, "analyser": {}})
        == "simulate"
    )
    # simulate with at least one analyser upgrades to simulate_analyze.
    assert (
        tasks.select_sde_task_profile(
            {"engine": {"sde": {"mode": "simulate"}}, "analyser": {"psd": {}}}
        )
        == "simulate_analyze"
    )
    # The plugins-block spelling of the analyser section counts as well.
    assert (
        tasks.select_sde_task_profile(
            {"engine": {"sde": {}}, "plugins": {"analyser": {"psd": {}}}}
        )
        == "simulate_analyze"
    )
    # analyze mode maps to analyze; profile validation enforces the required
    # analyser afterwards, so selection does not depend on its presence.
    assert (
        tasks.select_sde_task_profile({"engine": {"sde": {"mode": "analyze"}}})
        == "analyze"
    )
    assert (
        tasks.select_sde_task_profile(
            {
                "engine": {"sde": {"mode": "analyze"}},
                "analyser": {"allan_scaling": {}},
            }
        )
        == "analyze"
    )
