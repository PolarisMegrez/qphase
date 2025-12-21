---
layout: default
title: CLI Reference
---

# CLI Reference

The `qps` command-line interface is your main tool for interacting with QPhase.

## Project Management

### `qps init`
Initializes a new QPhase project in the current directory.
*   Creates `configs/`, `plugins/`, and `runs/` folders.
*   Generates a default `configs/global.yaml`.

## Running Simulations

### `qps run jobs [NAME]`
Runs a simulation job.
*   `NAME`: The name of the job file (without `.yaml`) in `configs/jobs/`.
*   Example: `qps run jobs my_sim`

### `qps run list`
Lists all available job configurations found in `configs/jobs/`.

## Plugin Inspection

### `qps list`
Lists all installed plugins, grouped by type (backend, model, integrator, etc.).

### `qps show [PLUGIN]`
Shows detailed information about a specific plugin, including its description and source.
*   Example: `qps show backend.numpy`

### `qps template [PLUGIN]`
Generates a configuration template for a plugin. This is very useful for copy-pasting into your YAML files.
*   Example: `qps template model.vdp_oscillator`

## Configuration Management

### `qps config show`
Displays the current effective configuration (System + Global).

### `qps config set [KEY] [VALUE]`
Updates a value in `configs/global.yaml`.
*   Example: `qps config set paths.output_dir ./my_results`
