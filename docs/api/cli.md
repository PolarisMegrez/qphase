---
description: CLI Reference
---

# CLI Reference

The `qphase` command-line interface is the primary tool for interacting with the QPhase framework. It facilitates project initialization, job execution, plugin management, and configuration generation.

## Global Options

All commands support the following global flags:

*   `--help`: Display the help message and exit.
*   `--version`: Display the installed version number.

---

## Project Management

### `qphase init`

Initializes a new QPhase project in the current directory.

```bash
qphase init
```

**Functionality:**

1.  Creates the standard directory structure:
    *   `configs/`: Configuration files.
    *   `plugins/`: Directory for local user plugins.
    *   `runs/`: Output directory for simulation results.
2.  Generates a default `configs/global.yaml` file.

---

## Simulation Execution

### `qphase run`

Executes simulation jobs defined in the `configs/jobs/` directory.

```bash
qphase run [JOB_NAME] [OPTIONS]
```

*   **Arguments**:
    *   `JOB_NAME`: The name of the job configuration file (without extension) located in `configs/jobs/`.
*   **Options**:
    *   `--list`: List all available job configurations and exit.
    *   `--verbose` / `-v`: Enable verbose debug logging.

**Examples**:

```bash
# Execute a single job
qphase run vdp_sde

# List available jobs
qphase run --list

# Run with verbose logging
qphase run --verbose vdp_sde
```

---

## Plugin Management

### `qphase list`

Lists all registered plugins available in the current environment.

```bash
qphase list [OPTIONS]
```

*   **Options**:
    *   `--category` / `-c`: Filter plugins by category (comma-separated).

**Example**:

```bash
qphase list
# Lists all plugins (backend, model, engine, etc.)

qphase list -c backend
# Lists only backend plugins
```

### `qphase show`

Displays detailed information about a specific plugin, including its description, source code location, and configuration schema.

```bash
qphase show [PLUGIN_ID]... [OPTIONS]
```

*   **Arguments**:
    *   `PLUGIN_ID`: One or more plugin identifiers in `namespace.name` format (e.g., `model.vdp_two_mode`).
*   **Options**:
    *   `--verbose` / `-v`: Show additional metadata (e.g., file path, package version).

**Example**:

```bash
qphase show model.vdp_two_mode
qphase show backend.numpy --verbose
```

### `qphase template`

Generates a configuration template for a specific plugin. This is useful for copy-pasting into your job config files.

```bash
qphase template [PLUGIN_ID]... [OPTIONS]
```

*   **Arguments**:
    *   `PLUGIN_ID`: One or more plugin identifiers in `namespace.name` format.
*   **Options**:
    *   `--output` / `-o`: Output file path. Default is `-` (stdout).
    *   `--format`: Output format, either `yaml` (default) or `json`.

**Example**:

```bash
# Print YAML template to console
qphase template model.vdp_two_mode

# Save to file
qphase template model.vdp_two_mode -o my_config.yaml
```

---

## Configuration Management

### `qphase config show`

Displays the current configuration.

```bash
qphase config show [OPTIONS]
```

*   **Options**:
    *   `--system` / `-s`: Show the system configuration (`system.yaml`) instead of the global project configuration (`global.yaml`).

### `qphase config set`

Sets a configuration value in `global.yaml` (or `system.yaml`).

```bash
qphase config set [KEY] [VALUE] [OPTIONS]
```

*   **Arguments**:
    *   `KEY`: Dot-separated configuration key (e.g., `paths.output_dir`).
    *   `VALUE`: The value to set.
*   **Options**:
    *   `--system` / `-s`: Update the system configuration instead of global.

**Example**:

```bash
qphase config set paths.output_dir ./my_runs
```

### `qphase config reset`

Resets the configuration to defaults.

```bash
qphase config reset [OPTIONS]
```

*   **Options**:
    *   `--system` / `-s`: Reset system configuration.
    *   `--force` / `-f`: Force reset without confirmation.
