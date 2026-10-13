import importlib.metadata
import inspect

import pytest
from qphase.core.registry import discovery, registry
from qphase.data import DataKind, ProductDeclaration, ProductRequirement
from qphase_sde.contracts.analyser import (
    AnalyserDeclarationContract,
    AnalyserExecutionContract,
    AnalyserWorkspaceEstimate,
    AnalyserWorkspaceRequest,
    WorkEstimate,
)
from qphase_sde.contracts.reducer import ReducerProtocol

# Minimal constructor kwargs per analyser; analysers absent from this mapping
# construct with defaults.
MINIMAL_CONFIGS = {
    "psd": {"kind": "complex", "modes": [0]},
    "dist": {"modes": [0]},
    "pdist": {"modes": [0]},
    "trajectory_diagnostics": {"modes": [0]},
    "allan_variance": {"modes": [0]},
    "allan_scaling": {"scan_param": "x", "critical_value": 1.0},
    "band_limited_carrier": {"scan_param": "x"},
    "finite_delay_carrier": {"scan_param": "x"},
    "lorentz_fitter": {"scan_param": "x"},
    "spectral_ridge": {"scan_param": "x"},
    "peaks": {"scan_param": "x"},
    "peak_paths": {"scan_param": "x"},
    "coherence_frequency": {"scan_param": "x"},
    "quadratic_moments": {"observables": {"n": {"matrix": [[1.0]]}}},
}


def _analyser_names() -> list[str]:
    # Entry points are the discovery source; other tests may register
    # additional names into the global registry, so never enumerate it.
    return sorted(
        entry.name.split(".", 1)[1]
        for entry in importlib.metadata.entry_points(group="qphase")
        if entry.name.startswith("analyser.")
    )


def _request() -> AnalyserWorkspaceRequest:
    return AnalyserWorkspaceRequest(
        trajectory_bytes=2 * 128 * 16,
        n_traj=2,
        saved_samples=128,
        n_record_modes=1,
        real_itemsize=8,
        backend_name="numpy",
    )


@pytest.fixture(scope="module", autouse=True)
def _discovered_plugins():
    discovery.discover_plugins()


@pytest.mark.parametrize("name", _analyser_names())
def test_registered_analyser_satisfies_contract(name):
    cls = registry.get_plugin_class("analyser", name)
    instance = cls(**MINIMAL_CONFIGS.get(name, {}))
    assert isinstance(instance, AnalyserDeclarationContract)
    assert isinstance(instance, AnalyserExecutionContract)
    assert "context" in inspect.signature(instance.execute).parameters

    requirements = instance.input_requirements()
    assert isinstance(requirements, list)
    assert all(isinstance(item, ProductRequirement) for item in requirements)
    requirement_names = [item.name for item in requirements]
    assert len(set(requirement_names)) == len(requirement_names)

    specs = instance.output_specs(label="probe")
    assert isinstance(specs, tuple) and specs
    assert all(isinstance(item, ProductDeclaration) for item in specs)
    spec_names = [item.name for item in specs]
    assert len(set(spec_names)) == len(spec_names)
    for spec in specs:
        assert isinstance(spec.kind, DataKind)
        assert all(isinstance(field, str) for field in spec.fields)

    request = _request()
    assert isinstance(instance.workspace(request), AnalyserWorkspaceEstimate)
    work = instance.work_estimate(request)
    assert isinstance(work, WorkEstimate)
    assert work.total is None or work.total >= 0
    reducer = instance.reducer()
    assert reducer is None or isinstance(reducer, ReducerProtocol)


def test_psd_does_not_declare_inline_peaks():
    from qphase_sde.analyser.psd import PsdAnalyzer

    default = PsdAnalyzer(kind="complex", modes=[0])
    assert [spec.name for spec in default.output_specs(label="probe")] == ["probe"]


def test_all_production_reducers_declare_retained_workspace():
    expected = {
        "allan_variance",
        "coherence_carrier",
        "coherence_matrix",
        "moment_statistics",
        "psd",
        "quadratic_moments",
    }
    retained: dict[str, AnalyserWorkspaceEstimate] = {}
    request = AnalyserWorkspaceRequest(
        trajectory_bytes=2 * 128 * 16,
        n_traj=2,
        total_n_traj=8,
        saved_samples=128,
        n_record_modes=1,
        real_itemsize=8,
        backend_name="numpy",
    )
    for name in _analyser_names():
        cls = registry.get_plugin_class("analyser", name)
        instance = cls(**MINIMAL_CONFIGS.get(name, {}))
        if instance.reducer() is not None:
            retained[name] = instance.workspace(request)

    assert set(retained) == expected
    for estimate in retained.values():
        assert estimate.retained_host_bytes + estimate.retained_device_bytes > 0


def test_analyser_base_does_not_require_legacy_analyze_api():
    from qphase_sde.analyser.base import Analyzer

    class TypedOnlyAnalyzer(Analyzer):
        def execute(
            self,
            inputs,
            *,
            backend,
            label,
            metadata=None,
            context=None,
        ):
            del backend, label, metadata, context
            return inputs

    analyser = TypedOnlyAnalyzer()
    assert not hasattr(analyser, "analyze")
