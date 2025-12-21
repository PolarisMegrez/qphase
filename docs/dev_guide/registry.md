---
layout: default
title: Registry System
---

# Registry System

The **Registry System** is the central nervous system of QPhase. It provides a unified mechanism for registering, discovering, configuring, and instantiating plugins across the entire application.

## Core Architecture

The registry is implemented as a singleton `RegistryCenter` that manages a two-level lookup table: `Namespace -> Name -> Entry`.

### Namespaces

To prevent naming collisions, plugins are organized into namespaces. Common namespaces include:

| Namespace | Description | Example |
|-----------|-------------|---------|
| `backend` | Computational backends | `numpy`, `torch`, `cupy` |
| `engine`  | Simulation engines | `sde`, `ode` |
| `model`   | Physics models | `kerr_cavity`, `vdp` |
| `command` | CLI commands | `run`, `config` |

### Entry Types

The registry supports two types of entries to balance startup performance with flexibility:

1.  **Callable Entry (Eager)**:
    *   **Description**: The plugin class or factory function is imported and stored directly.
    *   **Use Case**: Core plugins, testing, or when immediate availability is required.
    *   **Pros**: Fast instantiation.
    *   **Cons**: Increases application startup time (imports dependencies immediately).

2.  **Dotted Entry (Lazy)**:
    *   **Description**: Stores a string path (e.g., `"pkg.module:ClassName"`) instead of the object.
    *   **Use Case**: Third-party plugins, optional dependencies.
    *   **Pros**: Zero startup cost. The module is only imported when the plugin is requested.
    *   **Cons**: Slight delay on first use.

## Registration

### Manual Registration

You can manually register plugins using the `register` (eager) or `register_lazy` (lazy) methods.

```python
from qphase.core.registry import registry

# Eager Registration
class MyBackend:
    ...
registry.register("backend", "my_backend", MyBackend)

# Lazy Registration
registry.register_lazy("backend", "heavy_backend", "my_pkg.heavy:HeavyBackend")
```

### Decorator Registration

For internal or core plugins, the `@register` decorator is often the most convenient method.

```python
from qphase.core.registry import register

@register("backend", "numpy")
class NumpyBackend:
    ...
```

## Discovery Mechanism

QPhase automatically discovers plugins installed in the environment using Python's standard `entry_points` mechanism. This allows other packages (like `qphase_sde` or `qphase_viz`) to register plugins without modifying the core code.

### `pyproject.toml` Configuration

To expose a plugin to QPhase, add it to the `[project.entry-points.qphase]` section of your `pyproject.toml`:

```toml
[project.entry-points.qphase]
# Format: "namespace.name" = "package.module:Class"

"backend.custom" = "my_plugin.backend:CustomBackend"
"model.new_physics" = "my_plugin.models:NewPhysicsModel"
```

When `qphase` starts, it scans these entry points and registers them as **Lazy Entries**.

## Instantiation

Plugins are instantiated via the `create` method. The registry handles the logic of resolving the entry, importing the module (if lazy), and validating the configuration.

```python
# Basic instantiation
backend = registry.create("backend:numpy")

# With configuration
config = {"precision": "float64"}
backend = registry.create("backend:numpy", **config)
```

### Configuration Validation

If a plugin defines a `config_schema` (a Pydantic model), the registry can automatically validate the configuration before instantiation.

```python
# 1. Registry looks up "backend:numpy"
# 2. Registry sees it has a config_schema
# 3. Registry validates `kwargs` against the schema
# 4. Registry passes validated config to the constructor
```

## Introspection

The registry provides tools to inspect available plugins, which is useful for CLI help generation and debugging.

```python
# List all namespaces
all_plugins = registry.list()

# List plugins in a specific namespace
backends = registry.list("backend")
# Output: {'numpy': {'kind': 'callable'}, 'torch': {'kind': 'dotted'}}
```

