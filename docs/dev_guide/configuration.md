---
layout: default
title: Configuration System
---

# Configuration System

The QPhase configuration system is designed to be hierarchical, type-safe, and extensible. It separates **system-level** settings (paths, environment) from **job-level** settings (physics parameters, solver options).

## Architecture

The configuration is loaded in layers, with each layer overriding the previous one:

1.  **System Config** (`system.yaml`): Defines global paths, plugin directories, and runtime behavior.
2.  **Global Config** (`configs/global.yaml`): Defines project-wide defaults for physics and solvers.
3.  **Job Config** (`configs/jobs/*.yaml`): Defines specific parameters for a single simulation run.

## System Configuration

The `SystemConfig` controls *where* QPhase looks for things and *how* it behaves.

### Loading Priority
1.  **Environment Variable**: `QPHASE_SYSTEM_CONFIG` (Highest priority)
2.  **User Config**: `~/.qphase/system.yaml`
3.  **Default Config**: Built-in package defaults (Lowest priority)

### Schema
```python
class SystemConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    auto_save_results: bool = True
    parameter_scan: dict = Field(default_factory=lambda: {"enabled": True})

class PathsConfig(BaseModel):
    output_dir: str = "./runs"
    global_file: str = "./configs/global.yaml"
    plugin_dirs: list[str] = ["./plugins"]
    config_dirs: list[str] = ["./configs"]
```

## Job Configuration

A `JobConfig` defines a single unit of work. It is typically loaded from a YAML file in `configs/jobs/`.

```yaml
# configs/jobs/my_simulation.yaml
name: my_simulation
engine:
  sde:
    t_end: 10.0
    dt: 0.01
plugins:
  backend:
    name: numpy
    params:
      precision: float64
  model:
    name: vdp_oscillator
    params:
      mu: 2.0
```

## Plugin Configuration (For Developers)

The most important feature for developers is **Schema Validation**. QPhase uses [Pydantic](https://docs.pydantic.dev/) to validate plugin configurations automatically.

### Defining a Schema

When creating a plugin, you should define a Pydantic model for its configuration.

```python
from pydantic import BaseModel, Field, PositiveFloat

class VDPConfig(BaseModel):
    """Configuration schema for the Van der Pol model."""
    mu: float = Field(1.0, description="Non-linearity parameter")
    eta: PositiveFloat = Field(0.1, description="Noise strength")
    
class VDPModel:
    # Link the schema to the class
    config_schema = VDPConfig

    def __init__(self, config: VDPConfig):
        self.mu = config.mu
        self.eta = config.eta
```

### Automatic Validation

When the Registry instantiates your plugin:
1.  It detects the `config_schema` attribute.
2.  It extracts the `params` dictionary from the YAML config.
3.  It validates `params` against your Pydantic model.
4.  It passes the *validated model instance* to your `__init__` method.

If the YAML contains invalid types (e.g., `mu: "string"` instead of float), QPhase will raise a clear error message before the simulation starts.

## Configuration Merging

QPhase uses a **Deep Merge** strategy.

*   **Dictionaries** are merged recursively.
*   **Lists** and **Scalars** in the higher-priority config *replace* the values in the lower-priority config.

### Example

**Global Config (`global.yaml`):**
```yaml
plugins:
  backend:
    name: numpy
    params:
      precision: float64
      device: cpu
```

**Job Config (`job.yaml`):**
```yaml
plugins:
  backend:
    params:
      device: cuda  # Overrides 'cpu'
      # 'precision' is inherited as 'float64'
```

**Result:**
```python
{
    "name": "numpy",
    "params": {
        "precision": "float64",
        "device": "cuda"
    }
}
```

## Accessing Configuration in Code

In most core components (like Engines), the full configuration is available via `self.config`.

```python
class MyEngine(EngineBase):
    def run(self):
        # Access job-level params
        t_end = self.config.engine['sde']['t_end']
        
        # Access system-level paths
        out_dir = self.context.system_config.paths.output_dir
```

