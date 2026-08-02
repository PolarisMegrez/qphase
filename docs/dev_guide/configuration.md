---
description: Configuration System
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
    scan: ScanSpec | None
    input: InputSpec | None
    system: SystemConfig | None
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

## Parameter Scan Schema

Parameter scanning is represented only by the core `ScanSpec`. It contains
ordered named `ScanAxisSpec` values, each with a plugin target path and exactly
one of `values`, `linspace`, or `logspace`. Compiling the schema produces an
immutable runtime `ParameterGrid`.

Plugin schemas continue to describe scalar and structured plugin values. Core
does not use `scanable` metadata or inspect arbitrary lists to build scans;
legacy metadata may remain only to identify old syntax for migration errors.
This avoids ambiguity between a physical vector and a collection of points.

The loader deliberately rejects removed workflow forms with actionable errors:

- scalar model fields configured with list-as-scan syntax;
- string-valued `input`;
- `aggregate_input`;
- job-level runtime policy fields that belong to `SystemConfig`.

## System Runtime Schema

Core storage, checkpoint, resource-hint, progress, and CLI behavior belongs to
`SystemConfig`. `SystemConfigStore` merges packaged defaults with sparse
machine/user overrides and never creates a user file during a read. A job may
override the same schema through its existing `system` field. Dynamic hardware
facts remain outside persisted configuration and are sampled into the
`ResourceSnapshot` carried by `ExecutionContext`.
