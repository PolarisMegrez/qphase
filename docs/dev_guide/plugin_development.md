---
layout: default
title: Plugin Development
parent: Developer Guide
nav_order: 8
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

class MyModel:
    # Metadata for the Registry
    name: ClassVar[str] = "my_model"
    description: ClassVar[str] = "Kerr oscillator with additive noise"
    config_schema: ClassVar[type] = MyModelConfig

    def __init__(self, config: MyModelConfig, backend: Any):
        self.cfg = config
        self.xp = backend  # Abstract backend (numpy/torch/cupy)

    def drift(self, t: float, state: Any) -> Any:
        """
        Calculate the deterministic drift vector: A(X, t)
        dx = A(X, t)dt + B(X, t)dW
        """
        x = state
        chi = self.cfg.chi
        kappa = self.cfg.kappa

        # Use self.xp for tensor operations
        # -1j * chi * |x|^2 * x - kappa * x
        term1 = -1j * chi * (self.xp.abs(x)**2) * x
        term2 = -kappa * x
        return term1 + term2

    def diffusion(self, t: float, state: Any) -> Any:
        """
        Calculate the diffusion matrix: B(X, t)
        """
        # Additive noise: returns a scalar or constant tensor
        return self.xp.sqrt(self.cfg.kappa)
```

---

## 3. Registration

Plugins can be registered via two mechanisms:

### A. Local Registration (Development)
Create a `.qphase_plugins.yaml` file in your project root. This maps the plugin name to the Python class path.

```yaml
model:
  my_model: "plugins.my_physics:MyModel"
```

### B. Package Registration (Distribution)
If distributing the plugin as a Python package, use standard entry points in `pyproject.toml`.

```toml
[project.entry-points."qphase.plugins"]
"model:my_model" = "my_package.models:MyModel"
```
