"""Tests for the typed parent analysers introduced in Wave 5B."""

from __future__ import annotations

import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.coherence_frequency import (
    CoherenceFrequencyAnalyzer,
    CoherenceFrequencyAnalyzerConfig,
)
from qphase_sde.analyser.peak_paths import (
    PeakPathsAnalyzer,
    PeakPathsAnalyzerConfig,
)
from qphase_sde.analyser.peaks import PeaksAnalyzer, PeaksAnalyzerConfig
from qphase_sde.analyser.psd import PsdAnalyzer
from qphase_sde.engine import Engine, EngineConfig
from qphase_sde.result import SDEDataBundle, SDEProvenance


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
            estimator="band_limited",
            child={"freq_min": -2.0, "freq_max": 0.2, "maximum_lag": 100.0},
        )
    )

    payload = analyzer.analyze(source, NumpyBackend()).data_dict
    products = analyzer.build_products(payload, scan_size=2, label="frequency")

    assert products is not None
    product = products["frequency"]
    assert product.attributes["graph_ready"] is True
    assert product.axis("scan").size == 2
    assert product.handle("frequency").materialize().shape == (2, 1)

    runtime = Engine(
        config=EngineConfig(mode="analyze"),
        plugins={"analyser": {"coherence_frequency": analyzer}},
    ).run(source)
    assert runtime.shape == (2,)
    assert runtime.products["coherence_frequency"].axis("scan").size == 2
    output = tmp_path / "coherence_frequency"
    runtime.save(output)
    assert (output / "artifact_manifest.json").is_file()


def test_peak_paths_parent_consumes_peak_product():
    source = _spectrum_bundle()
    peak_analyzer = PeaksAnalyzer(
        PeaksAnalyzerConfig(scan_param="epsilon", finder="scipy")
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
    analyzer = PeakPathsAnalyzer(PeakPathsAnalyzerConfig(scan_param="epsilon"))

    payload = analyzer.analyze(candidate_source, NumpyBackend()).data_dict
    products = analyzer.build_products(payload, scan_size=None, label="paths")

    assert products is not None
    assert products["paths"].attributes["graph_ready"] is True
    assert products["paths"].attributes["source_product"] == "peaks"

    runtime = Engine(
        config=EngineConfig(mode="analyze"),
        plugins={"analyser": {"peak_paths": analyzer}},
    ).run(candidate_source)
    assert runtime.shape == (2,)
    assert runtime.products["peak_paths"].axis("path").size == 1


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
            estimator="registry_test",
            child={"freq_min": -2.0, "freq_max": 0.2},
        )
    )
    result = analyzer.analyze(source, NumpyBackend())
    assert len(result.data_dict["estimates"]) == 2
