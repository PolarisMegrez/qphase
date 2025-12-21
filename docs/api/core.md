---
layout: default
title: Core API Reference
parent: API Reference
nav_order: 1
---

# Core API Reference

## Scheduler

The `Scheduler` is responsible for orchestrating the execution of jobs.

### `class Scheduler`

**Parameters:**
*   `system_config` (SystemConfig, optional): System configuration.
*   `default_output_dir` (str, optional): Override default output directory.
*   `on_progress` (Callable, optional): Callback for progress updates.

**Methods:**

#### `run(job_list: JobList) -> list[JobResult]`
Executes a list of jobs serially. Handles dependency resolution and parameter scanning.

## Configuration

### `class JobConfig`
Represents a single simulation job.

**Fields:**
*   `name` (str): Unique identifier.
*   `engine` (dict): Engine configuration (e.g., `sde`).
*   `plugins` (dict): Plugin configurations (backend, model, etc.).
*   `params` (dict): Job-specific parameters.
*   `input` (str, optional): Upstream job name.
*   `output` (str, optional): Output filename/job name.

### `class SystemConfig`
Global system settings.

**Fields:**
*   `paths` (PathsConfig): Directory paths.
*   `auto_save_results` (bool): Whether to save results to disk.
*   `parameter_scan` (dict): Settings for batch execution.

## Registry

### `class RegistryCenter`
Central management for plugins.

**Methods:**
*   `register(namespace, name, builder)`: Register a new plugin.
*   `create(full_name, **kwargs)`: Instantiate a plugin.

