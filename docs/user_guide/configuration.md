---
layout: default
title: Configuration Guide
parent: User Guide
nav_order: 2
---

# Configuration Guide

QPhase employs a hierarchical, YAML-based configuration system designed to ensure **reproducibility** and **flexibility**. Instead of hardcoding parameters in Python scripts, you define them in declarative configuration files.

## Configuration Hierarchy

QPhase merges configurations from multiple sources to determine the final settings for a job. The priority order (highest to lowest) is:

1.  **Job Configuration** (`configs/jobs/*.yaml`): Settings specific to a single run.
2.  **Global Configuration** (`configs/global.yaml`): Project-wide defaults (e.g., default backend, logging level).
3.  **System Defaults**: Built-in defaults provided by the QPhase package.

**Best Practice**: Use `global.yaml` for settings that rarely change (like your preferred backend or precision) and Job Configs for experiment-specific parameters.

## Anatomy of a Job Configuration

A job configuration file defines *what* to run and *how* to run it.

```yaml
# configs/jobs/example_job.yaml

# [Required] Unique identifier for the job
name: example_experiment

# [Required] Engine Configuration
# Defines the simulation loop parameters.
engine:
  sde:  # The engine type (Stochastic Differential Equation)
    t_end: 100.0
    dt: 0.01
    n_trajectories: 1000

# [Optional] Plugin Configuration
# Overrides defaults for specific components.
plugins:
  backend:
    name: torch
    params:
      device: "cuda:0"  # Use GPU
      float_dtype: float32

  model:
    name: kerr_cavity
    params:
      chi: 1.0
      epsilon: 2.5
      kappa: 0.5

# [Optional] I/O Configuration
output: custom_filename  # Defaults to job name
```

## Parameter Scanning

QPhase has built-in support for **parameter sweeps**. You can define a list of values for any parameter, and the scheduler will automatically expand it into multiple jobs.

### Cartesian Product (Grid Search)
By default, if you provide lists for multiple parameters, QPhase generates a job for every combination (Cartesian product).

```yaml
model:
  kerr_cavity:
    chi: [1.0, 2.0]
    epsilon: [0.1, 0.5, 1.0]
```

This generates **6 jobs** ($2 \times 3$):
1. `chi=1.0, epsilon=0.1`
2. `chi=1.0, epsilon=0.5`
...
6. `chi=2.0, epsilon=1.0`

### Zipped Expansion (One-to-One)
If you want to vary parameters in lockstep (e.g., scanning along a specific diagonal in phase space), you can configure the scan method in `system.yaml` (advanced usage) or rely on future CLI flags. *Note: The default behavior is Cartesian.*

## Configuration Snapshots

Reproducibility is a core tenet of QPhase. Every time a job runs, QPhase saves a **Configuration Snapshot** (`config_snapshot.json`) in the output directory.

This snapshot contains:
*   The fully merged configuration (Global + Job).
*   The exact versions of all plugins used.
*   The random seed used (if applicable).
*   System metadata (timestamp, user, machine).

**You never need to guess which parameters produced a specific result file.**

## Generating Templates

To avoid looking up documentation for every parameter, QPhase can generate configuration templates for any registered plugin.

```bash
# Generate a template for the 'vdp_oscillator' model
qps template model.vdp_oscillator
```

Output:
```yaml
model:
  vdp_oscillator:
    # Nonlinear damping parameter
    mu: 1.0
    # Noise strength
    eta: 0.1
```

You can copy-paste this output directly into your job file.

