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
from qphase_sde.result import SDEResult

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
        result_data = backend.abs(mean_val)**2

        return SDEResult(trajectory=result_data, kind="trajectory")
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

    def run(self):
        # ... simulation loop ...
        pass
```

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

`qphase_sde.kernels.compile_cached_kernel` compiles a CuPy `RawKernel` once and caches it by name, dtype, and source hash, so repeated scheduler runs do not recompile.

```python
from qphase_sde.kernels import compile_cached_kernel

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
