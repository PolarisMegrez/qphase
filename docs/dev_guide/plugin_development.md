---
description: Plugin Development Guide
---

# Plugin Development Guide

This guide outlines the procedure for developing extensions (plugins) for the QPhase framework. The most common extension point is the **Model**, which defines the physical system to be simulated.

## The Plugin Contract

QPhase utilizes **Structural Subtyping** (Duck Typing). A class is recognized as a valid plugin if it satisfies the interface contract defined by the corresponding Protocol (e.g., `PluginBase`, `ModelBase`). Inheritance from framework base classes is optional but not required.

To implement a plugin, three components are necessary:
1.  **Configuration Schema**: A Pydantic model defining the parameters.
2.  **Implementation Class**: The class containing the logic.
3.  **Registration**: An entry in the plugin registry.

---

## 1. Defining the Configuration Schema

Parameters are defined using **Pydantic** models. This ensures strict type validation and automatic documentation generation.

```python
from pydantic import BaseModel, Field

class MyModelConfig(BaseModel):
    """Configuration schema for MyModel."""

    # Required parameter (no default)
    chi: float = Field(..., description="Nonlinearity strength")

    # Optional parameter with default
    kappa: float = Field(1.0, gt=0, description="Decay rate (must be positive)")
```

---

## 2. Implementing the Logic

The implementation class must accept the configuration object and a backend instance in its constructor.

**Critical Requirement**: All mathematical operations must be performed using the injected `backend` instance (conventionally `self.xp`). Direct usage of `numpy` or `torch` breaks hardware agnosticism.

### Example: SDE Model Implementation

An SDE model typically implements `drift` and `diffusion` methods.

```python
from typing import Any, ClassVar
from qphase.backend.xputil import get_xp

class MyModel:
    # Metadata for the Registry
    name: ClassVar[str] = "my_model"
    description: ClassVar[str] = "Kerr oscillator with additive noise"
    config_schema: ClassVar[type] = MyModelConfig

    def __init__(self, config: MyModelConfig, **kwargs: Any):
        self.cfg = config
        # Backend is inferred from data in drift/diffusion

    def drift(self, state: Any, t: float, params: dict) -> Any:
        """
        Calculate the deterministic drift vector: A(X, t)
        dx = A(X, t)dt + B(X, t)dW
        """
        xp = get_xp(state)
        x = state
        chi = self.cfg.chi
        kappa = self.cfg.kappa

        # Use xp for tensor operations
        # -1j * chi * |x|^2 * x - kappa * x
        term1 = -1j * chi * (xp.abs(x)**2) * x
        term2 = -kappa * x
        return term1 + term2

    def diffusion(self, state: Any, t: float, params: dict) -> Any:
        """
        Calculate the diffusion matrix: B(X, t)
        """
        xp = get_xp(state)
        # Additive noise: returns a scalar or constant tensor
        return xp.sqrt(self.cfg.kappa)
```

### Example: Analyser Implementation

Analysers process the raw simulation results.

```python
from typing import Any, ClassVar
from qphase.backend.base import BackendBase
from qphase.core.protocols import ResultProtocol
from qphase_sde.analyser.result import AnalysisResult

class MyAnalyser:
    name: ClassVar[str] = "my_analyser"
    description: ClassVar[str] = "Calculates mean photon number"
    config_schema: ClassVar[type] = MyAnalyserConfig

    def __init__(self, config: MyAnalyserConfig, **kwargs: Any):
        self.cfg = config

    def analyze(self, data: Any, backend: BackendBase) -> ResultProtocol:
        """
        Process the simulation result.
        """
        # Example: Calculate mean of trajectory
        # data is expected to be a tensor or TrajectorySet
        if hasattr(data, "data"):
            traj = data.data
        else:
            traj = data

        mean_val = backend.mean(traj, axis=0)
        mean_power = backend.abs(mean_val) ** 2

        return AnalysisResult(data_dict={"mean_power": mean_power})
```

### Example: Engine Implementation with Manifest

Engines orchestrate the simulation. They must declare their dependencies using `EngineManifest`.

```python
from typing import ClassVar
from qphase.core.protocols import EngineManifest

class MyEngine:
    name: ClassVar[str] = "my_engine"
    description: ClassVar[str] = "Custom simulation engine"
    config_schema: ClassVar[type] = MyEngineConfig

    # Declare dependencies
    manifest: ClassVar[EngineManifest] = EngineManifest(
        required_plugins={"model", "backend"},
        optional_plugins={"analyser"}
    )

    def __init__(self, config: MyEngineConfig, plugins: dict):
        self.cfg = config
        self.model = plugins["model"]
        self.backend = plugins["backend"]
        # Handle optional plugin
        self.analyser = plugins.get("analyser")

    def run(self, input_data=None, *, context=None):
        # ... simulation loop ...
        pass
```

SDE analysers should inherit `qphase_sde.analyser.Analyzer`. The base class
provides conservative planner defaults. Override `capabilities()` with an
`AnalyzerExecutionCapabilities` value and `estimate_workspace()` with separate
host/device byte estimates whenever the analyser transfers a full trajectory,
supports trajectory batching or time streaming, or allocates a substantial
workspace. Do not advertise batching/streaming until the corresponding public
accumulator/stream protocol is implemented; these declarations control memory
planning, not only documentation.

### Scan-Aware Engines

The scheduler passes scan and runtime services through `ExecutionContext`.
Engines own the numerical strategy; core does not expand the grid into jobs.

```python
from qphase.core.scan import execute_pointwise

def run(self, input_data=None, *, context=None):
    grid = None if context is None else context.parameter_grid
    if grid is None:
        return self.solve_one()

    # Use a resource-specific tile/batch strategy, or the core helper.
    rows = execute_pointwise(
        grid,
        self.solve_point,
        context=context,
        chunk_size=16,
    )
    return MyDatasetResult.from_points(grid, rows)
```

A scan result should implement `DatasetResultProtocol`: named `axes`, logical
`shape`, `point_view(index)`, and `save_dataset(path, layout,
shard_target_bytes)`. This lets downstream jobs use `dataset` or lazy `map`
input without exposing internal tiles as scheduler jobs. Engines may also read
resource hints, report progress, check cancellation, and save completed chunks
through the context.

Do not add a generic core planner for a resource-specific algorithm. If the
engine can fuse trajectories, run batched Newton, or schedule model-specific
tiles, keep that policy in the resource package and treat `ParameterGrid` as the
portable input contract.

### Subplugin Slots

A plugin may expose a recursively validated child selection without turning the
child into a scheduler job. Keep child implementations in a flat reusable
registry namespace and declare their relationship in the parent manifest:

```python
from qphase.core.protocols import PluginManifest, SubpluginSlot

class SpectrumAnalyser:
    manifest = PluginManifest(
        subplugins={
            "estimator": SubpluginSlot(
                namespace="spectral_estimator",
                cardinality="one",
                default="periodogram",
                protocol="my_package.base:SpectralEstimator",
            )
        }
    )

    def __init__(self, config, *, subplugins=None):
        self.config = config
        self.estimator = subplugins["estimator"]
```

The YAML slot contains exactly one selected implementation:

```yaml
analyser:
  spectrum:
    estimator:
      welch:
        nperseg: 4096
```

Each child remains a normal plugin with `name`, `description`, and
`config_schema`. Core performs cardinality, schema, protocol, cycle, and depth
validation before constructing the parent. Parents must not call the registry
again. Child plugins do not own job directories, artifacts, progress streams,
or scheduler DAG nodes.

---

## 3. Registration

Plugins can be registered via two mechanisms:

### A. Local Registration (Development)
Create a `.qphase_plugins.yaml` file in your project root. This maps the plugin namespace and name to the Python class path.

```yaml
model.my_model: "plugins.my_physics:MyModel"
analyser.my_analyser: "plugins.my_analysis:MyAnalyser"
```

### B. Package Registration (Distribution)
If distributing the plugin as a Python package, use standard entry points in `pyproject.toml`.

```toml
[project.entry-points.qphase]
"model.my_model" = "my_package.models:MyModel"
"analyser.my_analyser" = "my_package.analysis:MyAnalyser"
```

## Online SDE Observer Plugins

An online observer inherits `qphase_sde.observer.Observer`, declares a strict
`config_schema`, and implements `initialize`, `observe`, and `finalize`.
`observe` returns either `None` or an `ObserverDecision`; it must not return
plugin-specific action strings or require the SDE engine to inspect its result
fields. The observer also owns `merge_payloads` and `split_payload`, normally by
declaring `per_trajectory_keys` and `invariant_payload_keys` on the base
implementation. Override `_finalize_group_payload` when aggregate counters must
be recomputed after either trajectory-batch merging or fused-scan splitting.

Observers may use backend array operations on the live state but must not
mutate it or consume RNG. Keep device-to-host transfers limited to event-size
payloads. Child-specific error details belong in `ObserverDecision.details`;
the engine wraps them in the generic `ObserverTriggeredError`.

## Optional: Kernelized Terms for CuPy

For models that are evaluated many times per time step, you can provide an optional **fused drift+diffusion kernel** that the SDE integrator will use when the active backend is CuPy. This is purely optional: if the kernel is not available or the backend is not CuPy, the integrator falls back to the standard `drift`/`diffusion` methods.

### Protocol

Add two optional methods to your model class:

```python
def has_kernelized_terms(self, backend: BackendBase) -> bool:
    """Return True if a fused kernel is available for *backend*."""
    return str(backend.backend_name()).lower() == "cupy"

def kernelized_terms(
    self, y: Any, t: float, params: dict[str, Any], backend: BackendBase
) -> tuple[Any, Any]:
    """Return (drift, diffusion) for the whole ensemble in one call."""
    ...
```

* `has_kernelized_terms` should be conservative: return `False` unless the kernel has been tested for the active backend.
* `kernelized_terms` receives the full state ensemble `y` of shape `(n_traj, n_modes)` and the model parameters. It must return drift and diffusion tensors with the same shapes as `drift()` and `diffusion()` would.
* Parameters may be scalars or per-trajectory arrays (for batched scans), so the kernel wrapper must broadcast them to shape `(n_traj,)` before launching.

### Reusing the Kernel Cache

Model-specific CUDA compilation and caching belongs beside the model kernels,
not in `qphase_sde`. The workspace models provide
`models.kernels.cupy_utils.compile_cached_kernel`, which caches by name, dtype,
compiler options, and source hash for the current process.

```python
from models.kernels.cupy_utils import compile_cached_kernel

def kernelized_terms(self, y, t, params, backend):
    import cupy as cp
    import numpy as np

    n = y.shape[0]
    rdtype = y.real.dtype
    if rdtype == np.float32:
        source = _MY_SOURCE.replace("$T$", "float").replace("$CT$", "float2")
        ctype = "complex<float>"
    else:
        source = _MY_SOURCE.replace("$T$", "double").replace("$CT$", "double2")
        ctype = "complex<double>"

    kernel = compile_cached_kernel("my_model_terms", ctype, source)

    # Broadcast scalar parameters to per-trajectory arrays.
    p = cp.full((n,), float(params["kappa"]), dtype=cp.float64)

    drift = cp.empty_like(y)
    diffusion = cp.zeros((n, n_modes, n_noise), dtype=y.dtype)
    kernel((blocks,), (threads,), (y, p, n, drift, diffusion))
    return drift, diffusion
```

Keep kernel implementations under the integration-scheme namespace. See `models/kernels/euler_maruyama/vdp_2mode.py` for a complete example.

For an integration-specific fused step or chunk, implement a
`ModelKernelPlugin` and register it from the model's `kernel_plugins()` method.
Declare `scheme`, `backend_name`, and the supported `operations` (`step`,
`step_chunk`, or `terms`). This keeps CUDA formulas model-owned while the SDE
integrator consumes one stable capability interface.

A RawKernel wrapper must define output ownership and stream completion. In
particular, it must not return while a caller-visible cached output or a noise
buffer that the engine may recycle can still be written or read on an
untracked stream. Use the active CuPy stream consistently and establish a
completion boundary before returning whenever the surrounding buffer protocol
cannot carry a CUDA event. Test both isolated launches and `step -> chunk`
call order; passing only a single kernel comparison does not detect reuse
races.
