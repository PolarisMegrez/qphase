---
layout: default
title: CLI Reference
parent: User Guide
nav_order: 3
---

# CLI Reference

The `qps` (QPhase Shell) command-line interface is the primary tool for interacting with the QPhase framework. It handles project initialization, job execution, plugin management, and configuration.

## Global Options

All commands support the following global flags:

*   `--help`: Show help message and exit.
*   `--version`: Show the version number.
*   `--verbose` / `-v`: Enable debug logging.
*   `--log-file PATH`: Save logs to a specific file.

---

## Project Management

### `qps init`

Initializes a new QPhase project in the current directory.

```bash
qps init
```

**What it does:**
1.  Creates the standard directory structure:
    *   `configs/`: Configuration files.
    *   `plugins/`: Local user plugins.
    *   `runs/`: Output directory for simulation results.
2.  Generates a default `configs/global.yaml`.

---

## Simulation Execution

### `qps run jobs`

Runs a simulation job defined in a YAML file.

```bash
qps run jobs [JOB_NAME] [OPTIONS]
```

*   **Arguments**:
    *   `JOB_NAME`: The name of the job file located in `configs/jobs/` (without the `.yaml` extension).
*   **Options**:
    *   `--dry-run`: Parse the configuration and build plugins but do not execute the simulation loop. Useful for validation.
    *   `--parallel / --serial`: (Experimental) Force parallel or serial execution.

**Example**:
```bash
# Runs configs/jobs/vdp_oscillator.yaml
qps run jobs vdp_oscillator
```

### `qps run list`

Lists all available job configurations found in the `configs/jobs/` directory.

```bash
qps run list
```

---

## Plugin System

### `qps list`

Lists all registered plugins available in the current environment.

```bash
qps list [NAMESPACE]
```

*   **Arguments**:
    *   `NAMESPACE` (Optional): Filter by namespace (e.g., `backend`, `model`).

**Example**:
```bash
qps list backend
# Output:
# - numpy (qphase.backend.numpy)
# - torch (qphase.backend.torch)
```

### `qps show`

Displays detailed information about a specific plugin, including its description, source code location, and configuration schema.

```bash
qps show [PLUGIN_ID]
```

**Example**:
```bash
qps show model.kerr_cavity
```

### `qps template`

Generates a configuration template for a specific plugin. This is extremely useful for creating new job files.

```bash
qps template [PLUGIN_ID]
```

**Example**:
```bash
qps template engine.sde > my_config.yaml
```

---

## Configuration

### `qps config show`

Displays the current effective configuration. This combines the system defaults (from `qphase` package) and the project overrides (from `configs/global.yaml`).

```bash
qps config show
```

### `qps config set`

Updates a value in the project's `configs/global.yaml` file.

```bash
qps config set [KEY] [VALUE]
```

**Example**:
```bash
# Change the default output directory
qps config set paths.output_dir ./results_v2

# Enable debug mode globally
qps config set debug true
```
