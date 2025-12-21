---
layout: default
title: Protocol System
---

# Protocol System

QPhase uses **Python Protocols** (PEP 544) to define the interfaces between the core framework and its plugins. This approach allows for structural subtyping (duck typing), meaning a class is considered a valid plugin if it implements the required methods, regardless of its inheritance hierarchy.

## Why Protocols?

| Feature | Abstract Base Classes (ABC) | Protocols (Structural Subtyping) |
| :--- | :--- | :--- |
| **Inheritance** | Mandatory (`class MyPlugin(PluginBase)`) | Optional (Implicit) |
| **Coupling** | High (Plugin depends on Core) | Low (Plugin depends on Interface) |
| **Flexibility** | Rigid hierarchy | Flexible implementation |
| **Verification** | Import time | Static analysis (Mypy) & Runtime |

By using Protocols, we ensure that:
1.  **Third-party plugins** do not need to import `qphase` to be developed (they just need to match the signature).
2.  **Testing** is easier because you can mock components by simply creating an object with the right methods.

---

## Core Protocols

### 1. PluginBase
The fundamental contract for all plugins.

```python
@runtime_checkable
class PluginBase(Protocol):
    """The minimal contract for any QPhase plugin."""
    
    # Metadata (Class Variables)
    name: ClassVar[str]                 # Unique identifier
    description: ClassVar[str]          # Human-readable description
    config_schema: ClassVar[type[Any]]  # Pydantic model for configuration

    def __init__(self, config: Any | None = None, **kwargs: Any) -> None:
        """
        Initialize the plugin.
        
        Args:
            config: A validated configuration object (instance of config_schema).
            **kwargs: Alternative way to pass configuration parameters.
        """
        ...
```

### 2. EngineBase
A specialized plugin that orchestrates a simulation. It extends `PluginBase` with a `run` method.

```python
@runtime_checkable
class EngineBase(PluginBase, Protocol):
    """Contract for simulation engines."""

    def __init__(
        self, 
        config: Any, 
        plugins: dict[str, Any], 
        **kwargs: Any
    ) -> None:
        """
        Engines receive a dictionary of other instantiated plugins 
        (e.g., backend, integrator) upon initialization.
        """
        ...

    def run(self, data: Any | None = None) -> Any:
        """
        Execute the main logic.
        
        Args:
            data: Input data from a previous job (or None).
            
        Returns:
            The result of the computation (preferably a ResultBase).
        """
        ...
```

### 3. ResultBase
Defines how results should be persisted.

```python
@runtime_checkable
class ResultBase(Protocol):
    """Contract for saveable/loadable results."""

    def save(self, path: str | Path) -> None:
        """Save the result to the specified path."""
        ...

    @classmethod
    def load(cls, path: str | Path) -> "ResultBase":
        """Load the result from the specified path."""
        ...
```

---

## Backend Protocol

The `BackendBase` protocol abstracts numerical operations, allowing the same physics code to run on NumPy, PyTorch, or other backends.

### Key Methods

| Category | Methods | Description |
| :--- | :--- | :--- |
| **Creation** | `asarray`, `zeros`, `empty`, `copy` | Array instantiation and conversion. |
| **Math** | `einsum`, `cholesky`, `mean`, `abs` | Core mathematical operations. |
| **Random** | `rng`, `randn`, `spawn_rngs` | Deterministic random number generation. |
| **FFT** | `fft`, `fftfreq` | Fast Fourier Transforms. |
| **Device** | `to_device` (Optional) | Moving data between CPU and GPU. |

### Capability Negotiation

Since not all backends support all features (e.g., NumPy doesn't support GPUs), backends implement a `capabilities()` method:

```python
def capabilities(self) -> dict[str, Any]:
    return {
        "device": "cuda:0",
        "supports_complex_view": True,
        "to_device": True,
        ...
    }
```

Plugins should check these capabilities at runtime to decide which code path to take.

---

## Configuration Protocol

While not a formal Python Protocol, all configuration classes must adhere to the **Pydantic BaseModel** interface.

```python
class PluginConfigBase(BaseModel):
    """Base class for all plugin configurations."""
    
    model_config = ConfigDict(extra="allow")  # Allow unknown fields for forward compatibility

    @classmethod
    def from_raw(cls, raw: Any) -> "PluginConfigBase":
        """Factory method to create config from dict, instance, or None."""
        ...
```

This ensures that every plugin has a strongly-typed, self-validating configuration.

