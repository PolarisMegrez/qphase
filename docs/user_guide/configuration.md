---
description: Job Configuration
---

# Job Configuration

QPhase utilizes a hierarchical, YAML-based configuration system designed to ensure reproducibility, flexibility, and ease of use.

## Writing Configuration Files

A QPhase "Job" is defined by a YAML file located in the `configs/jobs/` directory. Each file represents a single simulation task (or a set of tasks if parameter scanning is used).

### Using Templates

The easiest way to create a new configuration is to use the CLI to generate a template.

```bash
# Generate a template for a specific model
qphase template model.vdp_two_mode > configs/jobs/my_new_job.yaml
```

This file can then be edited to suit specific requirements.

### Instantiating Plugins

QPhase is plugin-based. Defining a section in the config file (like `model` or `backend`) instructs QPhase to instantiate a specific plugin class.

For example:

```yaml
model:
  vdp_two_mode:   # This matches the plugin ID "model.vdp_two_mode"
    D: 0.5        # These arguments are passed to the plugin's __init__ method
```

### Configuration Hierarchy

To determine the final settings for a simulation job, QPhase merges configurations from multiple sources. The priority order, from highest to lowest, is as follows:

1.  **Job Configuration** (`configs/jobs/*.yaml`): Settings specific to a single simulation run. These settings override all others.
2.  **Global Configuration** (`configs/global.yaml`): Project-wide defaults (e.g., default backend, logging preferences).
3.  **System Defaults**: Built-in defaults provided by the QPhase package and its plugins.

## Anatomy of a Job Configuration

A job configuration file defines the specific parameters for a simulation.

### General Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `name` | `str` | **Required.** A unique identifier for the job. Used for logging and output filenames. |
| `engine` | `dict` | **Required.** Configuration for the simulation engine. Must contain exactly one key corresponding to the engine name (e.g., `sde`). |
| `input` | `str` | **Optional.** Specifies an input source, such as the name of an upstream job. |
| `output` | `str` | **Optional.** Specifies the output destination. |

### QPhase-SDE Fields

When using the SDE engine (`qphase-sde`), the following top-level keys are available. These correspond to the plugin types used by the SDE solver.

| Field | Description |
| :--- | :--- |
| `model` | Configuration for the physical model plugin (e.g., `vdp_two_mode`, `kerr_cavity`). Defines the drift and diffusion terms. |
| `backend` | Configuration for the computational backend plugin (e.g., `numpy`, `torch`). Defines how arrays are handled. |
| `integrator` | Configuration for the SDE integrator plugin (e.g., `euler`, `milstein`). Defines the numerical stepping scheme. |

### QPhase-Viz Fields

When using the Visualization engine (`qphase-viz`), the following keys are relevant:

| Field | Description |
| :--- | :--- |
| `analyser` | Configuration for data analysis plugins (e.g., `psd`, `trajectory`). Processes raw simulation data. |
| `visualizer` | Configuration for plotting plugins. Defines how the analyzed data is rendered. |

## Parameter Scanning

QPhase includes built-in support for parameter sweeps, allowing you to run multiple simulations with varying parameters from a single config file.

### Cartesian Product (Grid Search)

If lists are provided for multiple parameters, QPhase generates a job for every possible combination (Cartesian product).

```yaml
model:
  kerr_cavity:
    chi: [1.0, 2.0]       # 2 values
    epsilon: [0.1, 0.5, 1.0] # 3 values
# Total jobs = 2 * 3 = 6
```

### Zipped Scanning

To scan parameters in lock-step (e.g., varying `chi` and `epsilon` together), you can use the zipped scan method. This is configured in `system.yaml` (see below) or by using specific syntax if supported by the plugin.

*(Note: Currently, the default behavior is Cartesian scan. Zipped scan requires enabling `parameter_scan.method = "zip"` in `system.yaml`)*

## System Configuration

The `configs/system.yaml` file controls the behavior of the QPhase framework itself, rather than specific simulation parameters.

### Key Settings

*   **`paths`**:
    *   `output_dir`: The root directory for all simulation outputs (default: `runs/`).
*   **`logging`**:
    *   `level`: Global logging level (INFO, DEBUG, WARNING).
*   **`parameter_scan`**:
    *   `enabled`: Enable/disable parameter scanning (default: `true`).
    *   `method`: Scanning method (`cartesian` or `zip`).
    *   `numbered_outputs`: Whether to append numbers to output directories for scanned jobs (default: `true`).

Example `system.yaml`:

```yaml
paths:
  output_dir: "runs"

logging:
  level: "INFO"

parameter_scan:
  enabled: true
  method: "cartesian"
  numbered_outputs: true
```
