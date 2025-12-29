---
layout: default
title: CLI Reference
parent: User Guide
nav_order: 4
---

# CLI Reference

The `qps` (QPhase Shell) command-line interface is the primary tool for interacting with the QPhase framework. It facilitates project initialization, job execution, plugin management, and configuration generation.

## Global Options

All commands support the following global flags:

*   `--help`: Display the help message and exit.
*   `--version`: Display the installed version number.
*   `--verbose` / `-v`: Enable verbose debug logging.
*   `--log-file PATH`: Redirect logs to a specific file.

---

## Project Management

### `qps init`

Initializes a new QPhase project in the current directory.

```bash
qps init
```

**Functionality:**
1.  Creates the standard directory structure:
    *   `configs/`: Configuration files.
    *   `plugins/`: Directory for local user plugins.
    *   `runs/`: Output directory for simulation results.
2.  Generates a default `configs/global.yaml` file.

---

## Simulation Execution

### `qps run jobs`

Executes a simulation job defined in the `configs/jobs/` directory.

```bash
qps run jobs [JOB_NAME] [OPTIONS]
```

*   **Arguments**:
    *   `JOB_NAME`: The name of the job configuration file (without the extension) located in `configs/jobs/`.
*   **Options**:
    *   `--list`: List all available job configurations in the `configs/jobs/` directory and exit.
    *   `--dry-run`: Parse the configuration and build plugins without executing the simulation loop. Useful for validation.
    *   `--parallel` / `--serial`: (Experimental) Force parallel or serial execution modes.

**Examples**:
```bash
# Execute the job defined in configs/jobs/vdp_oscillator.yaml
qps run jobs vdp_oscillator

# List all available jobs
qps run jobs --list

# Run with verbose logging enabled
qps run jobs --verbose my_simulation
```

---

## Plugin Management

### `qps list`

Lists all registered plugins available in the current environment.

```bash
qps list [CATEGORIES]
```

*   **Arguments**:
    *   `CATEGORIES` (Optional): A comma-separated list of namespaces to filter by (e.g., `backend`, `model`). Use `.` to list all categories.

**Example**:
```bash
qps list backend
# Output:
# Available Plugins
# backend: (2 plugins)
#   numpy  (qphase.backend.numpy)
#   torch  (qphase.backend.torch)
```

### `qps show`

Displays detailed information about a specific plugin, including its description, source code location, and configuration schema.

```bash
qps show [PLUGIN_ID]
```

*   **Arguments**:
    *   `PLUGIN_ID`: The full identifier of the plugin (e.g., `backend.numpy` or `model.kerr_cavity`).

### `qps template`

Generates a configuration template for a specific plugin. This is useful for quickly creating new configuration files.

```bash
qps template [PLUGIN_ID]
```

*   **Arguments**:
    *   `PLUGIN_ID`: The full identifier of the plugin.

**Example**:
```bash
qps template model.kerr_cavity
# Output:
# # Configuration for model.kerr_cavity
# chi: 1.0  # Nonlinearity parameter
# ...
```

---

## Configuration Management

### `qps config`

Manages the system configuration.

*(Detailed documentation for `qps config` subcommands to be added)*
