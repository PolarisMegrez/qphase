"""Tests for the typed parent analysers introduced in Wave 5B."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.registry import discovery, registry
from qphase_sde.analyser.band_limited_carrier import (
    BandLimitedCarrierAnalyzer,
    BandLimitedCarrierConfig,
)
from qphase_sde.analyser.coherence_frequency import (
    CoherenceFrequencyAnalyzer,
    CoherenceFrequencyAnalyzerConfig,
    _materialize_estimates,
)
from qphase_sde.analyser.finite_delay_carrier import (
    FiniteDelayCarrierAnalyzer,
    FiniteDelayCarrierConfig,
)
from qphase_sde.analyser.lorentz_fitter import LorentzFitter, LorentzFitterConfig
from qphase_sde.analyser.peak_paths import (
    PeakPathsAnalyzer,
    PeakPathsAnalyzerConfig,
)
from qphase_sde.analyser.peaks import PeaksAnalyzer, PeaksAnalyzerConfig
from qphase_sde.analyser.psd import PsdAnalyzer
from qphase_sde.analyser.spectral_ridge import (
    SpectralRidgeAnalyzer,
    SpectralRidgeConfig,
)
from qphase_sde.coherence_frequency import BandLimitedCoherenceFrequency
from qphase_sde.contracts.coherence import CoherenceFrequencyEstimate
from qphase_sde.engine import Engine, EngineConfig
from qphase_sde.graph import compile_product_graph
from qphase_sde.peak_finder import ScipyPeakFinder
from qphase_sde.peak_tracker import RidgePeakTracker
from qphase_sde.products import add_scan_parameter_coordinates
from qphase_sde.result import SDEDataBundle, SDEProvenance


def _compiled_graph(config):
    discovery.discover_plugins()
    return compile_product_graph(plugin_config={"analyser": config}, registry=registry)


def _spectrum_bundle() -> SDEDataBundle:
    axis = np.linspace(-4.0, 4.0, 2048, endpoint=False)
    payload = []
    for center in (-0.7, -0.6):
        psd = 0.03 + 2.0 * 0.08**2 / ((axis - center) ** 2 + 0.08**2)
        payload.append(
            {
                "axis": axis,
                "psd": psd[:, None],
                "psd_std": np.full((axis.size, 1), 0.01),
                "psd_sem": np.full((axis.size, 1), 0.001),
                "modes": [0],
                "orientation": "phase_decreasing",
                "uncertainty": {"n_independent": 4},
            }
        )
    analyzer = PsdAnalyzer(modes=[0], kind="complex")
    products = analyzer.build_products(payload, scan_size=2, label="psd")
    assert products is not None
    return SDEDataBundle(
        products,
        SDEProvenance(dt=0.1),
        scan_axes={"epsilon": np.asarray([0.0, 1.0])},
        scan_shape=(2,),
    )


def test_coherence_frequency_parent_publishes_typed_product(tmp_path):
    source = _spectrum_bundle()
    analyzer = CoherenceFrequencyAnalyzer(
        CoherenceFrequencyAnalyzerConfig(
            scan_param="epsilon",
            input_key="psd",
            estimator={
                "band_limited": {
                    "freq_min": -2.0,
                    "freq_max": 0.2,
                    "maximum_lag": 100.0,
                }
            },
        ),
        subplugins={
            "estimator": BandLimitedCoherenceFrequency(
                freq_min=-2.0, freq_max=0.2, maximum_lag=100.0
            )
        },
    )

    payload = analyzer.analyze(source, NumpyBackend()).data_dict
    products = analyzer.build_products(payload, scan_size=2, label="frequency")

    assert products is not None
    product = products["frequency"]
    assert product.attributes["graph_ready"] is True
    assert product.attributes["orientation"] == "phase_decreasing"
    assert product.attributes["channel_names"] == ["channel_0"]
    assert product.attributes["channel_indices"] == [0]
    assert product.axis("scan").size == 2
    assert product.handle("frequency").materialize().shape == (2, 1)

    runtime = Engine(
        config=EngineConfig(mode="analyze"),
        plugins={"analyser": {"coherence_frequency": analyzer}},
        product_graph=_compiled_graph(
            {
                "coherence_frequency": {
                    "scan_param": "epsilon",
                    "input_key": "psd",
                    "estimator": {"band_limited": {}},
                }
            }
        ),
    ).run(source)
    assert runtime.shape == (2,)
    assert runtime.products["coherence_frequency"].axis("scan").size == 2
    output = tmp_path / "coherence_frequency"
    runtime.save(output)
    assert (output / "artifact_manifest.json").is_file()


def test_peak_paths_parent_consumes_peak_product():
    source = _spectrum_bundle()
    from qphase_sde.peak_finder import ScipyPeakFinder

    peak_analyzer = PeaksAnalyzer(
        PeaksAnalyzerConfig(scan_param="epsilon"),
        subplugins={"finder": ScipyPeakFinder()},
    )
    peak_payload = peak_analyzer.analyze(source, NumpyBackend()).data_dict
    peak_products = peak_analyzer.build_products(
        peak_payload, scan_size=2, label="peaks"
    )
    assert peak_products is not None
    candidate_source = SDEDataBundle(
        peak_products,
        source.provenance,
        scan_axes=source.axes,
        scan_shape=source.shape,
    )
    analyzer = PeakPathsAnalyzer(
        PeakPathsAnalyzerConfig(scan_param="epsilon"),
        subplugins={"tracker": RidgePeakTracker()},
    )

    payload = analyzer.analyze(candidate_source, NumpyBackend()).data_dict
    products = analyzer.build_products(payload, scan_size=None, label="paths")

    assert products is not None
    assert products["paths"].attributes["graph_ready"] is True
    assert products["paths"].attributes["source_product"] == "peaks"

    runtime = Engine(
        config=EngineConfig(mode="analyze"),
        plugins={"analyser": {"peak_paths": analyzer}},
        product_graph=_compiled_graph(
            {"peak_paths": {"scan_param": "epsilon", "tracker": {"ridge": {}}}}
        ),
    ).run(candidate_source)
    assert runtime.shape == (2,)
    assert runtime.products["peak_paths"].axis("path").size == 1


def test_analyze_graph_connects_peak_candidates_to_paths():
    source = _spectrum_bundle()
    peaks = PeaksAnalyzer(
        PeaksAnalyzerConfig(scan_param="epsilon"),
        subplugins={"finder": ScipyPeakFinder()},
    )
    paths = PeakPathsAnalyzer(
        PeakPathsAnalyzerConfig(scan_param="epsilon"),
        subplugins={"tracker": RidgePeakTracker()},
    )

    runtime = Engine(
        config=EngineConfig(mode="analyze"),
        plugins={"analyser": {"peaks": peaks, "peak_paths": paths}},
        product_graph=_compiled_graph(
            {
                "peaks": {"scan_param": "epsilon", "finder": {"scipy": {}}},
                "peak_paths": {
                    "scan_param": "epsilon",
                    "tracker": {"ridge": {}},
                },
            }
        ),
    ).run(source)

    assert set(runtime.products) == {"peak_paths"}
    assert runtime.products["peak_paths"].attributes["source_product"] == "peaks"


def test_coherence_frequency_parent_resolves_child_through_registry():
    from qphase.core.registry import registry
    from qphase_sde.coherence_frequency import BandLimitedCoherenceFrequency

    class RegistryChild(BandLimitedCoherenceFrequency):
        name = "registry_test"

    registry.register(
        "coherence_frequency", "registry_test", RegistryChild, overwrite=True
    )
    source = _spectrum_bundle()
    analyzer = CoherenceFrequencyAnalyzer(
        CoherenceFrequencyAnalyzerConfig(
            scan_param="epsilon",
            input_key="psd",
            estimator={"registry_test": {"freq_min": -2.0, "freq_max": 0.2}},
        ),
        subplugins={"estimator": RegistryChild(freq_min=-2.0, freq_max=0.2)},
    )
    result = analyzer.analyze(source, NumpyBackend())
    assert len(result.data_dict["estimates"]) == 2


def test_short_delay_coherence_product_closes_sampling_basis(tmp_path):
    product = _materialize_estimates(
        [
            CoherenceFrequencyEstimate(
                frequency=0.2,
                estimator="short_delay",
                frequency_sem=0.01,
                independent_count=8,
            )
        ],
        scan_size=1,
        label="frequency",
    )

    assert product.schema.is_closed
    from qphase.data import save_products

    save_products(tmp_path / "frequency", {"frequency": product})


def test_coherence_frequency_plugin_graph_injects_selected_estimator():
    from qphase.core.plugin_graph import PluginGraphResolver
    from qphase.core.registry import discovery, registry

    discovery.discover_plugins()
    node = PluginGraphResolver(registry).resolve(
        "analyser",
        "coherence_frequency",
        {"scan_param": "epsilon", "estimator": {"short_delay": {}}},
    )

    assert node.instance._selected_estimator_name() == "short_delay"


def test_coherence_frequency_preserves_input_mode_identity():
    source = _spectrum_bundle()
    analyzer = CoherenceFrequencyAnalyzer(
        CoherenceFrequencyAnalyzerConfig(
            scan_param="epsilon",
            input_key="psd",
            estimator={"band_limited": {}},
        ),
        subplugins={"estimator": BandLimitedCoherenceFrequency()},
    )
    product = analyzer.execute(
        {"psd": source.products["psd"]},
        backend=NumpyBackend(),
        label="frequency",
    )["frequency"]
    assert product.attributes["channel_names"] == ["mode_0"]


def test_product_graph_rejects_unknown_published_output():
    with pytest.raises(ValueError, match="unknown products"):
        compile_product_graph(
            plugin_config={
                "analyser": {
                    "peaks": {"scan_param": "epsilon", "finder": {"scipy": {}}}
                }
            },
            registry=registry,
            publish_outputs={"peaks": ["not_declared"]},
        )


def test_cross_result_analysers_execute_from_typed_psd(monkeypatch):
    source = _spectrum_bundle()
    psd = add_scan_parameter_coordinates(
        source.products["psd"], {"epsilon": np.asarray([0.0, 1.0])}
    )

    def fail_legacy_loader(*args, **kwargs):
        raise AssertionError("typed execute unexpectedly used the legacy loader")

    monkeypatch.setattr(
        "qphase_sde.analyser.spectral_ridge.load_sde_results", fail_legacy_loader
    )
    monkeypatch.setattr(
        "qphase_sde.analyser.band_limited_carrier.load_sde_results",
        fail_legacy_loader,
    )
    monkeypatch.setattr(
        "qphase_sde.analyser.finite_delay_carrier.load_sde_results",
        fail_legacy_loader,
    )
    monkeypatch.setattr(
        "qphase_sde.analyser.lorentz_fitter._load_input", fail_legacy_loader
    )

    analysers = [
        SpectralRidgeAnalyzer(
            SpectralRidgeConfig(scan_param="epsilon", readouts=[0])
        ),
        BandLimitedCarrierAnalyzer(
            BandLimitedCarrierConfig(scan_param="epsilon", readout=0, export=[])
        ),
        FiniteDelayCarrierAnalyzer(
            FiniteDelayCarrierConfig(scan_param="epsilon", readout=0, export=[])
        ),
        LorentzFitter(
            LorentzFitterConfig(scan_param="epsilon", mode=0, export=[])
        ),
    ]
    for analyser in analysers:
        products = analyser.execute(
            {"psd": psd}, backend=NumpyBackend(), label="output"
        )
        assert products
        assert all(product.attributes["table"] for product in products.values())
