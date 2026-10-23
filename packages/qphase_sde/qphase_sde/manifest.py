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
#: ``peak_tracker``/``coherence_frequency`` declare their root-level
#: plugin-class protocols; concrete child plugins are registered separately.
MANIFEST = ResourcePackageManifest(
    resource_id="sde",
    package_version="2.0.0.dev0",
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
            protocol="qphase_sde.contracts.analyser:AnalyserExecutionContract",
            description="Analysis plugins declaring typed inputs/outputs, "
            "capabilities, workspace and progress; graph execution invokes "
            "the typed analyser execution contract.",
        ),
        PluginClassDeclaration(
            namespace="spectral_estimator",
            protocol="qphase_sde.estimators.spectral_estimator.base:SpectralEstimator",
            description="Replaceable spectral estimators (periodogram, Welch, "
            "multitaper).",
        ),
        PluginClassDeclaration(
            namespace="peak_finder",
            protocol="qphase_sde.estimators.peak_finder.base:TypedPeakFinder",
            description="Peak finders returning the grouped typed candidate "
            "table used by PSD and peak-path analysis.",
        ),
        PluginClassDeclaration(
            namespace="peak_tracker",
            protocol="qphase_sde.estimators.peak_tracker.base:PeakTrackerProtocol",
            description="Scan-axis peak-path trackers linking candidate rows "
            "into paths.",
        ),
        PluginClassDeclaration(
            namespace="coherence_frequency",
            protocol=("qphase_sde.estimators.coherence_frequency.base:CoherenceFrequencyProtocol"),
            description="Coherence-frequency estimators (short-delay, "
            "band-limited, finite-delay).",
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
            name="sampled_observable",
            kind="time_series",
            schema_ref="qphase_sde.contracts.quantities:SAMPLED_OBSERVABLE_PRODUCT",
            description="Real-valued sampled observable time series derived "
            "from field trajectories by a typed measurement.",
        ),
        DataProductDeclaration(
            name="spectrum",
            kind="spectral",
            schema_ref="qphase_sde.contracts.quantities:SPECTRUM_PRODUCT",
            description="Spectral estimates (amplitude, PSD, cross spectrum, "
            "coherence).",
        ),
        DataProductDeclaration(
            name="coherence_frequency",
            kind="statistics",
            schema_ref=("qphase_sde.contracts.coherence:COHERENCE_FREQUENCY_PRODUCT"),
            description="Coherence-frequency estimates with lag/bandwidth diagnostics.",
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
        DataProductDeclaration(
            name="peak_candidates",
            kind="statistics",
            schema_ref="qphase_sde.contracts.peaks:PEAK_PRODUCT",
            description="Grouped typed peak candidates with ragged offsets.",
        ),
        DataProductDeclaration(
            name="peak_paths",
            kind="statistics",
            schema_ref="qphase_sde.contracts.peaks:PEAK_PATH_PRODUCT",
            description="Typed scan-axis paths referencing peak candidates.",
        ),
    ],
    # No SDE-owned materializer exists yet: the NPZ adapter lives in core
    # (``qphase.data.npz``).
    materializers=[],
    backend_capabilities=BackendCapabilities(
        backends=["numpy", "cupy"],
        devices=["cpu", "cuda"],
        # No time-streaming reducer exists: trajectory batching is not
        # streaming, so the package must not claim it.
        streaming=False,
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
            purpose="Backend-neutral numerical operations (coordinates, transforms).",
        ),
    ],
)
