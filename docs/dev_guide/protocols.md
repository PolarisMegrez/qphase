---
layout: default
title: Protocol System
parent: Developer Guide
nav_order: 3
---

# Protocol System

QPhase utilizes **Python Protocols** (introduced in PEP 544) to define the interfaces between the core framework and its extensible components. This design choice favors **Structural Subtyping** (Duck Typing) over the traditional Nominal Subtyping enforced by Abstract Base Classes (ABCs).

## Structural vs. Nominal Subtyping

### Nominal Subtyping (ABCs)
In a nominal system, a class is a subtype of another only if it explicitly inherits from it.
*   **Requirement**: `class MyPlugin(PluginBase): ...`
*   **Drawback**: This creates a hard dependency on the framework code. Third-party plugins must import the base class, leading to potential version conflicts and tighter coupling.

### Structural Subtyping (Protocols)
In a structural system, a class is a subtype if it implements the required methods and attributes, regardless of inheritance.
*   **Requirement**: The class simply needs to have the correct methods.
*   **Advantage**: Decoupling. A plugin can be developed, tested, and distributed without importing `qphase` at runtime. The dependency is only on the *interface contract*, not the implementation.

## Core Protocols

The framework defines several key protocols that plugins must satisfy.

### 1. `PluginBase`
The fundamental contract for all discoverable components.

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
            **kwargs: Additional dependencies injected by the Registry.
        """
        ...
```

### 2. `EngineBase`
The contract for simulation engines.

```python
@runtime_checkable
class EngineBase(PluginBase, Protocol):
    def run(self) -> ResultBase:
        """Execute the simulation and return a result object."""
        ...
```

### 3. `BackendBase`
The contract for computational backends (see [Backend System](backend.md)).

## Runtime Verification

While Protocols are primarily a static analysis tool (used by MyPy/Pyright), QPhase utilizes the `@runtime_checkable` decorator to perform runtime validation during plugin instantiation. This allows the Registry to raise informative errors if a loaded plugin fails to satisfy the required contract.
