---
description: CLI Reference
---

# CLI Reference

The `qphase` command-line interface is the primary tool for interacting with the QPhase framework. It facilitates project initialization, job execution, plugin management, and configuration generation.

## Global Options

All commands support the following global flag:

*   `--help`: Display the help message and exit.

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

## Result Postprocessing

Postprocessing is expressed as a scheduler workflow rather than a standalone CLI command. Use `engine.sde` with `mode: analyze` and the `analyser.lorentz_fitter` plugin.

```yaml
- name: sim
  save: true
  scan:
    axes:
      omega_a:
        target: model.kerr_2mode.omega_a
        values: [0.9, 1.1]
  engine:
    sde: { ... }
  model:
    kerr_2mode:
      omega_a: 0.9
      omega_b: 1.0
      chi: 0.01
      gamma_a: 0.1
      gamma_b: 0.1
      g: 0.1
  analyser:
    psd:
      modes: [0]
      kind: complex

- name: fit
  input:
    from: sim
    mode: dataset
  engine:
    sde:
      mode: analyze
  analyser:
    lorentz_fitter:
      scan_param: omega_a
      mode: 0
```

Run it with:

```bash
qphase run my_workflow
```

The `fit` job produces `fit_results.csv` and `psd_merged.csv` in its run directory. The NPZ/PKL distribution bundles include `__schema_version__` metadata via `qphase.core.aggregation`.

---

## Plugin Management

### `qphase list`

Lists all registered plugins available in the current environment.

```bash
qphase list [OPTIONS]
```

*   **Options**:
    *   `--category` / `-c`: Filter plugins by category (comma-separated).
    *   `--tree`: Expand declared subplugin slots and child implementations.
    *   `--parent`: Show one parent plugin and its child tree.

**Example**:

```bash
qphase list
# Lists all plugins (backend, model, engine, etc.)

qphase list -c backend
# Lists only backend plugins

qphase list --parent analyser.psd
# Lists estimator.periodogram, estimator.welch, and estimator.multitaper
```

### `qphase show`

Displays detailed information about a specific plugin, including its description, source code location, and configuration schema.

```bash
qphase show [PLUGIN_ID]... [OPTIONS]
```

*   **Arguments**:
    *   `PLUGIN_ID`: One or more plugin identifiers in `namespace.name` format (e.g., `model.vdp_2mode`).
*   **Options**:
    *   `--verbose` / `-v`: Show additional metadata (e.g., file path, package version).

**Example**:

```bash
qphase show model.vdp_2mode
qphase show backend.numpy --verbose
qphase show analyser.psd/estimator.welch
```

### `qphase template`

Generates a configuration template for a specific plugin. This is useful for copy-pasting into your job config files.

```bash
qphase template [PLUGIN_ID]... [OPTIONS]
```

*   **Arguments**:
    *   `PLUGIN_ID`: One or more plugin identifiers in `namespace.name` format.
*   **Options**:
    *   `--output`: Output file path. Default is `-` (stdout).
    *   `--format`: Output format, either `yaml` (default) or `json`.
    *   `--select SLOT=CHILD`: Select a non-default child in the generated template.

**Example**:

```bash
# Print YAML template to console
qphase template model.vdp_2mode

# Save to file
qphase template model.vdp_2mode --output my_config.yaml
qphase template analyser.psd --select estimator=welch
```

---

## Configuration Management

`qphase config options analyser.psd/estimator` lists accepted child
implementations. `qphase config schema analyser.psd/estimator.welch` prints the
composite child schema. These are read-only registry queries.

### `qphase config show`

Displays the current configuration.

```bash
qphase config show [OPTIONS]
```

*   **Options**:
    *   `--system` / `-s`: Show the resolved `SystemConfig` instead of project plugin defaults (`global.yaml`).

### `qphase config set`

Sets a plugin default in `global.yaml`, or a sparse user SystemConfig override
in `~/.qphase/config.yaml` with `--system`.

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
