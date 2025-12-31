---
description: CLI Reference
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

### `qps run`

Executes simulation jobs defined in the `configs/jobs/` directory.

```bash
qps run [JOB_NAME] [OPTIONS]
```

*   **Arguments**:
    *   `JOB_NAME`: The name of the job configuration file (without extension) located in `configs/jobs/`.
*   **Options**:
    *   `--list`: List all available job configurations and exit.
    *   `--verbose` / `-v`: Enable verbose debug logging.

**Examples**:
```bash
# Execute a single job
qps run vdp_sde

# List available jobs
qps run --list

# Run with verbose logging
qps run --verbose vdp_sde
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
