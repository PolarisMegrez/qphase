---
layout: default
title: Configuration System
parent: Developer Guide
nav_order: 5
---

# Configuration System

The **Configuration System** is responsible for parsing, validating, and merging simulation parameters. It employs a hierarchical loading strategy and leverages **Pydantic** for strict schema validation.

## Configuration Hierarchy

The system constructs the final execution context by merging configuration data from three distinct layers, in increasing order of precedence:

1.  **System Defaults**: Hardcoded defaults within the package and plugin definitions.
2.  **Global Configuration** (`configs/global.yaml`): User-defined project-wide settings (e.g., default backend, logging verbosity).
3.  **Job Configuration** (`configs/jobs/*.yaml`): Experiment-specific parameters.

## The Loading Pipeline

The configuration loading process follows a strict pipeline:

1.  **File I/O**: The YAML file is read and parsed into a raw Python dictionary.
2.  **Structure Normalization**: The raw dictionary is normalized to ensure consistent structure (e.g., handling shorthand notations).
3.  **Plugin Extraction**: The system identifies keys that correspond to registered plugin namespaces (e.g., `backend`, `model`).
4.  **Schema Validation**:
    *   The core job structure is validated against the `JobConfig` model.
    *   Each plugin configuration block is validated against its respective `config_schema` defined by the plugin class.
5.  **Merging**: Global defaults are merged into the job configuration, filling in missing optional fields.

## Schema Validation with Pydantic

QPhase uses Pydantic v2 to enforce type safety and data integrity.

### `JobConfig` Model

The `JobConfig` model defines the structural skeleton of a simulation job.

```python
class JobConfig(BaseModel):
    name: str
    engine: dict[str, Any]
    plugins: dict[str, dict[str, Any]]
    params: dict[str, Any]
    # ...
```

### Plugin Schemas

Each plugin must define a `config_schema` class variable pointing to a Pydantic model. This allows the Registry to validate plugin-specific parameters *before* the plugin is instantiated.

**Example:**
```python
class KerrCavityConfig(BaseModel):
    chi: float = Field(..., gt=0, description="Nonlinearity")
    detuning: float = Field(0.0, description="Frequency detuning")
```

If a user provides a string for `chi` or a negative value, the Pydantic validator will raise a descriptive error during the loading phase, preventing runtime failures deep in the simulation loop.
