"""qphase_sde: Resource Package Manifest (2.0)
---------------------------------------------------------
Static declaration of the SDE resource package's managed assets: the unique
engine reference, composable resource profiles, plugin-class namespaces,
public data products, backend capabilities and the core compatibility range.

This module only assembles declarations: it must never import concrete
plugins, initialize backends or pull heavy dependencies. Engine, protocol and
schema targets are recorded as dotted strings and resolved lazily by the core
resource catalog.

Public API
----------
MANIFEST
    The static resource package manifest (schema ``qphase.resource/1``).
"""

from qphase.resources import (
    BackendCapabilities,
    CompatibilityRange,
    DataProductDeclaration,
    EngineDeclaration,
    OptionalDependencyDeclaration,
    PluginClassDeclaration,
    ResourceAssetDeclaration,
    ResourcePackageManifest,
    ResourceProfile,
)

__all__ = ["MANIFEST"]

#: The SDE resource package manifest. Plugin-class namespaces mirror the
#: entry-point namespaces declared in ``pyproject.toml``, plus overlay-only
#: namespaces (``model``: concrete models ship as project-local or
#: third-party overlay plugins, never as package-owned entry points).
#: Child-plugin namespaces whose root-level directories only land in
#: Phase 2 point at the protocols hosted under ``analyser/`` until then.
MANIFEST = ResourcePackageManifest(
    resource_id="sde",
    package_version="2.0.0",
    engine=EngineDeclaration(
        entry_point="engine.sde",
        target="qphase_sde.engine:Engine",
    ),
    profiles=[
        ResourceProfile.BASE,
        ResourceProfile.COMPUTE,
        ResourceProfile.SIMULATION,
    ],
    plugin_classes=[
        PluginClassDeclaration(
            namespace="model",
            protocol="qphase_sde.model:SDEModel",
            description="Ito SDE model protocol, noise specification and "
            "initial state; concrete models are project overlays.",
        ),
        PluginClassDeclaration(
            namespace="integrator",
            protocol="qphase_sde.integrator.base:Integrator",
            description="Fixed-step and adaptive SDE integrators.",
        ),
        PluginClassDeclaration(
            namespace="observer",
            protocol="qphase_sde.observer.base:SDEObserverProtocol",
            description="Online trajectory observers (first passage, bounds).",
        ),
        PluginClassDeclaration(
            namespace="analyser",
            protocol="qphase_sde.analyser.base:AnalyzerProtocol",
            description="Engine-facing analysis adapters owning data kinds, "
            "reducers, workspace and progress.",
        ),
        PluginClassDeclaration(
            namespace="spectral_estimator",
            protocol=(
                "qphase_sde.analyser.spectral_estimator.base:SpectralEstimator"
            ),
            description="Replaceable spectral estimators (periodogram, Welch, "
            "multitaper); hosted under analyser/ until Phase 2.",
        ),
        PluginClassDeclaration(
            namespace="peak_finder",
            protocol="qphase_sde.analyser.peak_finding.base:PeakFinder",
            description="Peak-candidate finders returning unified PeakCandidate "
            "results; hosted under analyser/ until Phase 2.",
        ),
    ],
    data_products=[
        DataProductDeclaration(
            name="trajectories",
            kind="time_series",
            schema_ref="qphase_sde.contracts.bundle:TRAJECTORY_PRODUCT",
            description="Simulated stochastic trajectories with sampling axis "
            "and realization semantics.",
        ),
        DataProductDeclaration(
            name="spectrum",
            kind="spectral",
            schema_ref="qphase_sde.contracts.quantities:SPECTRUM_PRODUCT",
            description="Spectral estimates (amplitude, PSD, cross spectrum, "
            "coherence).",
        ),
        DataProductDeclaration(
            name="spectral_peaks",
            kind="statistics",
            schema_ref="qphase_sde.contracts.peaks:PEAK_PRODUCT",
            description="Unified peak candidates and scan-resolved peak paths.",
        ),
        DataProductDeclaration(
            name="coherence_frequency",
            kind="statistics",
            schema_ref=(
                "qphase_sde.contracts.coherence:COHERENCE_FREQUENCY_PRODUCT"
            ),
            description="Coherence-frequency estimates with lag/bandwidth "
            "diagnostics.",
        ),
        DataProductDeclaration(
            name="allan_variance",
            kind="statistics",
            schema_ref="qphase_sde.contracts.quantities:ALLAN_PRODUCT",
            description="Allan variance and scaling exponents.",
        ),
        DataProductDeclaration(
            name="moments",
            kind="statistics",
            schema_ref="qphase_sde.contracts.quantities:MOMENT_FAMILY_PRODUCT",
            description="Moment families with shared independent counts and "
            "joint covariance.",
        ),
    ],
    # No SDE-owned materializer exists yet: the NPZ adapter lives in core
    # (``qphase.data.npz``).
    materializers=[],
    backend_capabilities=BackendCapabilities(
        backends=["numpy", "cupy"],
        devices=["cpu", "cuda"],
        streaming=True,
    ),
    optional_dependencies=[
        OptionalDependencyDeclaration(
            name="cupy-cuda12x",
            purpose="CUDA-resident integration, FFT and device-side reductions.",
            required_for=["backend"],
        ),
    ],
    compatibility=CompatibilityRange(qphase_core=">=2.0a0,<3.0"),
    extra_assets=[
        ResourceAssetDeclaration(
            path="math",
            kind="directory",
            visibility="public",
            purpose="Backend-neutral numerical operations (coordinates, "
            "transforms).",
        ),
    ],
)
