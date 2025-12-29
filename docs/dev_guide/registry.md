---
layout: default
title: Registry System
parent: Developer Guide
nav_order: 4
---

# Registry System

The **Registry System** functions as the central service locator and dependency injection container for QPhase. It manages the lifecycle of all extensible components, providing a unified mechanism for registration, discovery, and instantiation.

## Core Architecture

The registry is implemented as a singleton `RegistryCenter` that maintains a hierarchical lookup table: `Namespace -> Name -> Entry`.

### Namespaces

To ensure modularity and prevent naming collisions, plugins are segregated into namespaces. Standard namespaces include:

| Namespace | Description | Example |
|-----------|-------------|---------|
| `backend` | Computational backends | `numpy`, `torch` |
| `engine`  | Simulation engines | `sde`, `viz` |
| `model`   | Physical models | `kerr_cavity`, `vdp` |
| `integrator`| Numerical integrators | `euler_maruyama`, `srk` |
| `analyser`| Result analysis tools | `mean_photon`, `wigner` |

### Entry Management Strategy

The registry employs a dual-strategy for entry management to balance startup latency with runtime flexibility:

1.  **Eager Entries (Callable)**:
    *   **Mechanism**: The plugin class or factory function is imported and stored directly in memory during initialization.
    *   **Application**: Used for core plugins and testing scenarios where immediate availability is required.

2.  **Lazy Entries (Dotted Path)**:
    *   **Mechanism**: The registry stores a string reference (e.g., `"pkg.module:ClassName"`). The actual module import is deferred until the first request for instantiation.
    *   **Application**: Used for third-party plugins and optional dependencies. This minimizes the application's startup time and memory footprint.

## Discovery Mechanisms

QPhase supports two primary discovery mechanisms:

### 1. Entry Points (Package-based)
For distributable Python packages, QPhase utilizes the standard `entry_points` mechanism (defined in `pyproject.toml`). The registry scans the `qphase` group at startup.

```toml
[project.entry-points.qphase]
"model.my_model" = "my_package.models:MyModel"
```

### 2. Local Configuration (Development-based)
For local development and ad-hoc extensions, the registry parses a `.qphase_plugins.yaml` file located in the project root. This allows researchers to register scripts without packaging them.

```yaml
model.custom_hamiltonian: "plugins.physics:Hamiltonian"
```

## Instantiation Factory

The `create()` method serves as the universal factory for all components. It handles:
1.  **Resolution**: Looking up the entry by namespace and name.
2.  **Loading**: Importing the module if it is a lazy entry.
3.  **Validation**: Validating the provided configuration dictionary against the plugin's `config_schema` (Pydantic model).
4.  **Injection**: Instantiating the class with the validated configuration and any additional dependencies (e.g., injecting the `backend` instance into a `model`).

```python
# Example: Instantiating a model with dependency injection
model = registry.create(
    "model:kerr_cavity",
    config={"chi": 1.0},
    backend=numpy_backend_instance
)
```

## Dependency Resolution

While the Registry provides the mechanism to create plugins, the **Scheduler** acts as the orchestrator. It uses the `EngineManifest` of the selected Engine to determine which plugins to request from the Registry.
