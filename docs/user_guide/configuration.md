---
layout: default
title: Configuration Guide
parent: User Guide
nav_order: 3
---

# Configuration Guide

QPhase utilizes a hierarchical, YAML-based configuration system designed to ensure reproducibility, flexibility, and ease of use. By defining simulation parameters in declarative configuration files, users can manage complex experimental setups without modifying the underlying codebase.

## Configuration Hierarchy

To determine the final settings for a simulation job, QPhase merges configurations from multiple sources. The priority order, from highest to lowest, is as follows:

1.  **Job Configuration** (`configs/jobs/*.yaml`): Settings specific to a single simulation run. These settings override all others.
2.  **Global Configuration** (`configs/global.yaml`): Project-wide defaults (e.g., default backend, logging preferences).
3.  **System Defaults**: Built-in defaults provided by the QPhase package and its plugins.

**Recommendation**: Utilize `global.yaml` for settings that remain consistent across most experiments (such as the preferred computational backend or precision level) and reserve Job Configurations for experiment-specific parameters.

## Anatomy of a Job Configuration

A job configuration file defines the specific parameters for a simulation. It specifies the engine to be used, the plugins to be loaded, and the parameters for the physical model.

Below is an example of a complete job configuration:

```yaml
# configs/jobs/example_job.yaml

# [Required] Unique identifier for the job
name: example_experiment

# [Required] Engine Configuration
# Specifies the simulation engine and its parameters.
# The key must match a registered engine name (e.g., 'sde', 'viz').
engine:
  sde:
    t_end: 100.0
    dt: 0.01
    n_traj: 1000

# [Optional] Plugin Configuration
# Overrides defaults for specific components like backends, models, or integrators.
plugins:
  backend:
    numpy:  # The specific plugin implementation to use
      float_dtype: float64

  model:
    kerr_cavity:
      chi: 1.0
      epsilon: 2.5
      kappa: 0.5

  # [Optional] Analyser Configuration
  # If provided, the engine will perform analysis (e.g., PSD) instead of just outputting trajectories.
  analyser:
    psd:
      kind: complex
      modes: [0]
      convention: symmetric

# [Optional] Job Metadata
tags: ["test", "kerr", "sde"]

# [Optional] I/O Configuration
# input: upstream_job_name  # For chained jobs
# output: custom_filename   # Defaults to the job name
```

### Key Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `name` | `str` | **Required.** A unique identifier for the job. This name is used for logging and output file generation. |
| `engine` | `dict` | **Required.** Configuration for the simulation engine. Must contain exactly one key corresponding to the engine name (e.g., `sde`), with its configuration as the value. |
| `plugins` | `dict` | **Optional.** Configuration for plugins. Keys are plugin types (e.g., `backend`, `model`), and values are dictionaries mapping plugin names to their configurations. |
| `params` | `dict` | **Optional.** A flexible dictionary for job-specific parameters that do not fit into the engine or plugin schemas. |
| `tags` | `list[str]` | **Optional.** A list of tags for categorizing and filtering jobs. |
| `input` | `str` | **Optional.** Specifies an input source, such as the name of an upstream job or a file path. |
| `output` | `str` | **Optional.** Specifies the output destination. If omitted, the job name is used as the filename. |
| `depends_on` | `list[str]` | **Optional.** A list of job names that this job depends on. Used for scheduling execution order. |

## Parameter Scanning

QPhase includes built-in support for parameter sweeps, allowing users to define ranges of values for parameters. The scheduler automatically expands these ranges into multiple individual jobs.

### Cartesian Product (Grid Search)
By default, if lists are provided for multiple parameters, QPhase generates a job for every possible combination (Cartesian product).

```yaml
model:
  kerr_cavity:
    chi: [1.0, 2.0]
    epsilon: [0.1, 0.5, 1.0]
```

In this example, QPhase will generate **6 jobs** ($2 \times 3$):
1. `chi=1.0, epsilon=0.1`
2. `chi=1.0, epsilon=0.5`
...
6. `chi=2.0, epsilon=1.0`

### Zipped Expansion
For scenarios requiring parameters to vary in lockstep (e.g., scanning along a specific trajectory in parameter space), the "zipped" expansion method can be used. This is configured via the system settings or CLI flags.

## Configuration Snapshots

To ensure reproducibility, QPhase saves a **Configuration Snapshot** (`config_snapshot.json`) in the output directory for every executed job.

This snapshot includes:
*   The fully merged configuration (Global + Job).
*   The versions of all active plugins.
*   The random seed used (if applicable).
*   System metadata (timestamp, user, machine).

This mechanism ensures that every result file can be traced back to the exact parameters and software environment that produced it.
