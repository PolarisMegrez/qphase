---
description: Configuration Guide
---

# Configuration Guide

QPhase utilizes a hierarchical, YAML-based configuration system designed to ensure reproducibility, flexibility, and ease of use.

## Configuration Schemes

QPhase supports two configuration schemes. We **strongly recommend** the **Flat Scheme** for most use cases due to its clarity and composability.

### 1. Flat Scheme (Recommended)
In this scheme, **one file represents one job**. This avoids deep indentation and makes it easy to reuse jobs in different pipelines.

**File:** `configs/jobs/my_simulation.yaml`
```yaml
name: my_simulation

# Engine Configuration
# The key (e.g., 'sde') determines the engine type.
engine:
  sde:
    dt: 0.01
    t_end: 100.0
    n_traj: 1000

# Plugin Configurations (Top-level keys)
# Keys must match registered plugin types (e.g., 'model', 'backend').
model:
  kerr_cavity:
    chi: 1.0
    epsilon: 2.5

backend:
  numpy:
    float_dtype: float64

integrator:
  euler: {}
```

### 2. Nested Scheme (Legacy/Batch)
In this scheme, multiple jobs are defined in a single file under a `jobs` list. This is useful for defining a strict sequence of tasks that always run together, but it can be harder to read and maintain.

**File:** `configs/pipelines/full_pipeline.yaml`
```yaml
name: full_pipeline
jobs:
  - name: step_1
    engine:
      sde:
        dt: 0.01
    model:
      kerr_cavity:
        chi: 1.0

  - name: step_2
    input: step_1
    engine:
      viz:
        format: png
```

## Configuration Hierarchy

To determine the final settings for a simulation job, QPhase merges configurations from multiple sources. The priority order, from highest to lowest, is as follows:

1.  **Job Configuration** (`configs/jobs/*.yaml`): Settings specific to a single simulation run. These settings override all others.
2.  **Global Configuration** (`configs/global.yaml`): Project-wide defaults (e.g., default backend, logging preferences).
3.  **System Defaults**: Built-in defaults provided by the QPhase package and its plugins.

## Anatomy of a Job Configuration

A job configuration file defines the specific parameters for a simulation.

### Key Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `name` | `str` | **Required.** A unique identifier for the job. Used for logging and output filenames. |
| `engine` | `dict` | **Required.** Configuration for the simulation engine. Must contain exactly one key corresponding to the engine name (e.g., `sde`). |
| `model` | `dict` | **Optional.** Configuration for the physical model plugin. |
| `backend` | `dict` | **Optional.** Configuration for the computational backend plugin. |
| `integrator` | `dict` | **Optional.** Configuration for the integrator plugin. |
| `input` | `str` | **Optional.** Specifies an input source, such as the name of an upstream job. |
| `output` | `str` | **Optional.** Specifies the output destination. |

## Parameter Scanning

QPhase includes built-in support for parameter sweeps.

### Cartesian Product (Grid Search)
If lists are provided for multiple parameters, QPhase generates a job for every possible combination.

```yaml
model:
  kerr_cavity:
    chi: [1.0, 2.0]
    epsilon: [0.1, 0.5, 1.0]
```

This generates **6 jobs** ($2 \times 3$).
