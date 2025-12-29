---
layout: default
title: Core API Reference
parent: API Reference
nav_order: 1
---

# Core API Reference

This section documents the core components of the QPhase framework, including the Scheduler, Registry, and Configuration models.

## Scheduler

The `Scheduler` is the central component responsible for orchestrating the execution of simulation jobs. It handles dependency resolution, parameter scanning, and result persistence.

### `class qphase.core.Scheduler`

**Parameters:**

*   `system_config` (`SystemConfig`, optional): The system configuration object. If not provided, it is loaded from `system.yaml`.
*   `default_output_dir` (`str`, optional): Overrides the default output directory specified in the system configuration.
*   `on_progress` (`Callable[[JobProgressUpdate], None]`, optional): A callback function invoked with progress updates during job execution.
*   `on_run_dir` (`Callable[[Path], None]`, optional): A callback function invoked with the path to the run directory after each job completes.

**Methods:**

#### `run(job_list: JobList) -> list[JobResult]`

Executes a list of jobs serially. This method handles:
1.  **Dependency Resolution**: Ensures jobs are executed in the correct order based on their dependencies.
2.  **Parameter Scanning**: Expands jobs with scanable parameters into multiple tasks.
3.  **Directory Management**: Creates unique run directories for each job execution.
4.  **Snapshotting**: Saves configuration snapshots for reproducibility.

**Returns:**
*   `list[JobResult]`: A list of result objects containing execution status and metadata for each job.

---

## Configuration

### `class qphase.core.JobConfig`

Represents the configuration for a single simulation job.

**Fields:**

*   `name` (`str`): **Required.** A unique identifier for the job.
*   `engine` (`dict[str, Any]`): **Required.** Configuration for the simulation engine. Must contain exactly one key (the engine name) mapping to its configuration dictionary.
*   `plugins` (`dict[str, dict[str, Any]]`): **Optional.** Configuration for plugins, organized by plugin type (e.g., `backend`, `model`).
*   `params` (`dict[str, Any]`): **Optional.** A dictionary of job-specific parameters.
*   `input` (`str | None`): **Optional.** The name of an upstream job or a file path to use as input.
*   `output` (`str | None`): **Optional.** The output destination (filename or downstream job name).
*   `tags` (`list[str]`): **Optional.** A list of tags for categorization.
*   `depends_on` (`list[str]`): **Optional.** A list of job names that this job depends on.

### `class qphase.core.SystemConfig`

Represents the global system settings.

**Fields:**

*   `paths` (`PathsConfig`): Directory paths for configuration, plugins, and output.
*   `auto_save_results` (`bool`): Whether to automatically save results to disk.
*   `parameter_scan` (`dict`): Settings for batch execution and parameter scanning strategies.

---

## Registry

### `class qphase.core.RegistryCenter`

The central registry for managing plugins. It supports dynamic discovery, registration, and factory-style instantiation of components.

**Methods:**

#### `register(namespace: str, name: str, target: Any)`
Registers a new plugin.

*   `namespace`: The category of the plugin (e.g., "backend", "model").
*   `name`: The unique name of the plugin within its namespace.
*   `target`: The plugin class or factory function.

#### `create(full_name: str, config: Any = None, **kwargs) -> Any`
Instantiates a plugin.

*   `full_name`: The full identifier of the plugin (e.g., "backend:numpy").
*   `config`: The configuration object to pass to the plugin constructor.
*   `**kwargs`: Additional keyword arguments passed to the constructor.

#### `list(namespace: str | None = None) -> dict`
Lists registered plugins.

*   `namespace`: If provided, filters the list to a specific namespace.
*   **Returns**: A dictionary mapping plugin names to their metadata.
