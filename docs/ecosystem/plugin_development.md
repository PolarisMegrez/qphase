---
layout: default
title: Plugin Development Guide
parent: Ecosystem
nav_order: 1
---

# Plugin Development Guide

This guide explains how to develop plugins that comply with the QPhase core specifications. The core package defines structural contracts (Protocols), and third-party plugins only need to satisfy these contracts to be discovered and used by the system, without inheriting from any base classes.

## 1. Plugin Architecture Overview

**The Three-Element Contract**:

Each plugin needs to provide three elements:
1.  **Configuration Class** (`Config`): Inherits from Pydantic `BaseModel`, defining plugin parameters.
2.  **Implementation Class** (`Plugin`): Satisfies the `PluginBase` Protocol, providing functional implementation.
3.  **Entry Point**: Declared in `pyproject.toml` for automatic discovery.

**Why not use inheritance?**

The core package uses Protocols (structural subtyping) instead of ABCs (nominal subtyping):
*   Plugins can be developed without importing the core package.
*   Avoids complexities like diamond inheritance.
*   Supports duck typing; only method signatures need to match.
*   Optional runtime validation via `@runtime_checkable`.

## 2. Developing the Configuration Class

The configuration class is a container for plugin parameters, using Pydantic v2 for validation and serialization.

**Basic Pattern**:

```python
from pydantic import BaseModel, Field

class MyPluginConfig(BaseModel):
    """Plugin configuration description (shown in qps config template output)"""

    # Parameter with default value
    param1: float = Field(default=1.0, description="Parameter description")

    # Required parameter (no default value)
    param2: int = Field(..., description="Required parameter")

    # Parameter with validation
    step_size: float = Field(1e-3, gt=0, description="Must be greater than 0")

    # Enumerated options
    mode: str = Field("auto", pattern="^(auto|manual|disabled)$")
```

**Design Points**:

1.  **Field description**: Extracted by `qps config template` as YAML comments.
2.  **Validators**: Use Pydantic built-in validators (`gt`, `lt`, `ge`, `le`, `pattern`, etc.).
3.  **extra="allow"**: Recommended setting to allow undefined fields in the configuration file without error.
4.  **Default Value Strategy**:
    *   Provide `default` for parameters with reasonable defaults.
    *   Use `...` (Ellipsis) for parameters that must be specified by the user.

**Advanced Pattern - from_raw Factory**:

If the configuration requires special construction logic, you can provide a `from_raw()` class method:

```python
class MyPluginConfig(BaseModel):
    normalized_value: float

    @classmethod
    def from_raw(cls, raw: Any | None = None) -> "MyPluginConfig":
        if raw is None:
            return cls(normalized_value=0.0)
        if isinstance(raw, cls):
            return raw
        # Custom conversion logic
        data = dict(raw)
        if "raw_value" in data:
            data["normalized_value"] = data.pop("raw_value") / 100.0
        return cls.model_validate(data)
```

## 3. Developing the Plugin Class

The plugin class implements the functionality and must satisfy the structural contract of the `PluginBase` Protocol.

**Required Class Variables**:

```python
from typing import ClassVar

class MyPlugin:
    # Unique identifier (corresponds to entry point name)
    name: ClassVar[str] = "myplugin"

    # Human-readable description
    description: ClassVar[str] = "Plugin functionality description"

    # Configuration Schema Class (must be the class itself, not an instance)
    config_schema: ClassVar[type[MyPluginConfig]] = MyPluginConfig
```

**Required Initialization Signature**:

```python
def __init__(self, config: MyPluginConfig | None = None, **kwargs) -> None:
    # Handle configuration: support passing config object or unpacked arguments
    if config is not None:
        self.config = config
    else:
        self.config = MyPluginConfig(**kwargs)
```

**Design Principles**:

1.  **Optional config argument**: Allow `MyPlugin()` to be instantiated with all defaults.
2.  **Support kwargs**: Allow `MyPlugin(param1=2.0)` to pass arguments directly.
3.  **Lazy Initialization**: Heavy resources (like GPU contexts) should be initialized on first use.
4.  **Stateless Design**: Plugins should be as stateless as possible; state should be stored in configuration or externally.

**Complete Example** (Integrator):

```python
from typing import Any, ClassVar
from pydantic import BaseModel, Field

class EulerConfig(BaseModel):
    """Euler-Maruyama Integrator Configuration"""
    dt: float = Field(1e-3, gt=0, description="Time step size")
    adaptive: bool = Field(False, description="Adaptive step size (not implemented)")

class EulerMaruyama:
    """Euler-Maruyama Integrator Implementation"""

    name: ClassVar[str] = "euler"
    description: ClassVar[str] = "First-order explicit SDE integrator"
    config_schema: ClassVar[type[EulerConfig]] = EulerConfig

    def __init__(self, config: EulerConfig | None = None, **kwargs) -> None:
        self.config = config or EulerConfig(**kwargs)
        self._cache = None  # Lazy initialization cache

    def step(self, y: Any, t: float, dt: float, model: Any, dW: Any, backend: Any) -> Any:
        """Execute one integration step"""
        a = model.drift(y, t, model.params)
        L = model.diffusion(y, t, model.params)
        return a * dt + backend.einsum("tnm,tm->tn", L, dW)
```

## 4. Developing Engine Plugins

The Engine is a special type of plugin responsible for coordinating other plugins to execute computational tasks. It is the entry point for instantiation by the Scheduler.

**Differences from Ordinary Plugins**:

| Feature | Ordinary Plugin | Engine |
|---------|-----------------|--------|
| Init Signature | `__init__(config, **kwargs)` | `__init__(config, plugins, **kwargs)` |
| Receives Dependencies | No | Receives via `plugins` dictionary |
| Core Method | Custom | Must have `run(data)` |
| Return Value | Custom | Should return `ResultBase` |

**Engine Initialization Signature**:

```python
def __init__(
    self,
    config: MyEngineConfig,
    plugins: dict[str, Any],
    **kwargs: Any
) -> None:
    self.config = config
    # Extract dependencies from plugins dictionary
    self.backend = plugins.get("backend")
    self.integrator = plugins.get("integrator")
```

**Plugins Dictionary Structure**:

The Scheduler instantiates all declared plugins and passes them as a dictionary before calling the Engine:

```python
plugins = {
    "backend": <NumpyBackend instance>,
    "integrator": <EulerMaruyama instance>,
    "noise": <GaussianNoise instance>,
    # ... other plugins
}
```

**Run Method Contract**:

```python
def run(self, data: Any | None = None) -> ResultBase | Any:
    """
    Execute computational task

    Parameters
    ----------
    data : Any | None
        Output from upstream task, could be:
        - None (no upstream)
        - Python object (in-memory transfer)
        - Path object (file path)

    Returns
    -------
    ResultBase | Any
        Computation result, recommended to implement save/load methods
    """
    # 1. Process input data
    if data is not None:
        input_data = self._load_input(data)

    # 2. Execute computation
    result = self._compute(input_data)

    # 3. Return result
    return result
```

**Registering an Engine**:

```toml
[project.entry-points.qphase]
"engine.mypackage" = "mypackage.core.engine:MyEngine"
```

The namespace must be `engine`, and the name corresponds to the key in the `engine` field of `JobConfig` (i.e., the engine name).

## 5. Developing Backend Plugins

The Backend defines an abstract interface for computational operations, enabling multi-backend compatibility.

**Required Method Categories**:

1.  **Identification Methods**:
    *   `backend_name() -> str`: Return backend name.
    *   `device() -> str | None`: Return device identifier.
    *   `capabilities() -> dict`: Return capabilities dictionary.

2.  **Array Creation**:
    *   `array`, `asarray`, `zeros`, `empty`, `empty_like`, `copy`.

3.  **Math Operations**:
    *   `einsum`, `concatenate`, `cholesky`.
    *   `real`, `imag`, `abs`, `mean`.

4.  **FFT**:
    *   `fft`, `fftfreq`.

5.  **Random Numbers**:
    *   `rng(seed)`, `randn(rng, shape, dtype)`, `spawn_rngs(master_seed, n)`.

6.  **Optional Methods**:
    *   `stack`, `to_device`.

**Capability Declaration Design**:

```python
def capabilities(self) -> dict:
    return {
        "device": self.device(),
        "optimized_contractions": True,  # Whether einsum is optimized
        "supports_complex_view": False,  # Whether real/imag returns views
        "real_imag_split": True,
        "stack": True,
        "to_device": True,
        "numpy": True,  # Whether underlying implementation is NumPy-based
    }
```

Callers should detect features via `capabilities()` rather than assuming all backends support all features.

**RNG Design Points**:

```python
def rng(self, seed: int | None) -> Any:
    """Create RNG handle, return type is flexible"""
    return np.random.default_rng(seed)

def spawn_rngs(self, master_seed: int, n: int) -> list[Any]:
    """Derive n independent RNG streams from master seed"""
    ss = np.random.SeedSequence(master_seed)
    children = ss.spawn(n)
    return [np.random.default_rng(c) for c in children]

def randn(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any:
    """Generate standard normal samples using rng"""
    out = rng.normal(size=shape)
    return out.astype(dtype if dtype else np.float64, copy=False)
```

## 6. Entry Point Registration

Plugins are automatically discovered via the Python entry points mechanism.

**pyproject.toml Configuration**:

```toml
[project.entry-points.qphase]
# Format: "namespace.name" = "module.path:ClassName"

# Ordinary Plugins
"integrator.euler" = "mypackage.integrators:EulerMaruyama"
"integrator.milstein" = "mypackage.integrators:Milstein"

# Backend Plugins
"backend.mybackend" = "mypackage.backend:MyBackend"

# Engine Plugins
"engine.mypackage" = "mypackage.engine:MyEngine"
```

**Naming Conventions**:

| Namespace | Usage | Example |
|-----------|-------|---------|
| `backend` | Computational Backend | `backend.numpy`, `backend.torch` |
| `integrator` | SDE Integrator | `integrator.euler`, `integrator.milstein` |
| `engine` | Execution Engine | `engine.sde`, `engine.viz` |
| `noise` | Noise Model | `noise.gaussian`, `noise.colored` |
| `visualizer` | Visualizer | `visualizer.phase_plane`, `visualizer.psd` |

**Lazy Loading Behavior**:

Plugins registered via Entry points are lazy-loaded by default:
*   Only the path string is recorded at startup; no import is performed.
*   The module is imported only when `registry.create()` is called for the first time.
*   Optional dependencies (like CuPy) do not affect system startup even if not installed.

## 7. Using Plugins in Configuration

Referencing plugins in JobConfig YAML:

```yaml
jobs:
  - name: my_simulation

    # Engine Configuration (Nested Dictionary Format)
    engine:
      sde:
        t_end: 10.0
        n_steps: 10000
        n_traj: 100

    plugins:
      backend:
        name: numpy  # Corresponds to entry point name
        # No extra parameters

      integrator:
        name: euler
        params:
          dt: 0.001
          adaptive: false
```

**Validation Process**:

1.  `_validate_plugins()` is called during JobConfig initialization.
2.  `registry.validate_plugin_config()` is called for each plugin configuration.
3.  The plugin's `config_schema` is retrieved and validated using Pydantic.
4.  A `QPhaseConfigError` is raised if validation fails.

## 8. Development Checklist

When developing a new plugin, ensure the following requirements are met:

**Configuration Class**:
- [ ] Inherits from Pydantic `BaseModel`.
- [ ] All parameters have `Field` description (for template generation).
- [ ] Required parameters use `...`.
- [ ] Consider setting `model_config = ConfigDict(extra="allow")`.

**Plugin Class**:
- [ ] Defines `name`, `description`, `config_schema` class variables.
- [ ] `__init__` supports `config` argument and `**kwargs`.
- [ ] Heavy resources are lazily initialized.

**Entry Point**:
- [ ] Correctly declared in `pyproject.toml`.
- [ ] Correct namespace (backend/integrator/engine/...).
- [ ] Correct path format (`module.path:ClassName`).

**Testing**:
- [ ] Can be instantiated with `MyPlugin()` (no args).
- [ ] Can be instantiated with `MyPlugin(config=MyConfig(...))`.
- [ ] `qps plugin list` can discover the plugin.
- [ ] `qps config template namespace name` can generate a template.
