---
description: Core API reference
---

# Core API Reference

## Project And Workflow

`ProjectContext.discover()` resolves the current `qphase.project/2` manifest from
an explicit path, `QPHASE_PROJECT`, or an upward search. It exposes the Project's
Workflow catalog, defaults file, local plugin roots, and Session root.

`WorkflowCatalog` recursively lists and resolves strict `qphase.workflow/2`
documents by stable Workflow ID. `load_workflow()` validates one document and
returns a `WorkflowSpec` containing logical `JobConfig` nodes.

## Scheduler

```python
Scheduler(
    system_config: SystemConfig | None = None,
    project: ProjectContext | None = None,
    on_progress: Callable[[ProgressSnapshot], None] | None = None,
    on_job_dir: Callable[[Path], None] | None = None,
    cancellation: CancellationController | None = None,
)
```

`Scheduler.run(workflow, dry_run=False, resume_from=None)` validates the logical
Job graph, resolves engines and plugins, passes scan grids to resource engines,
persists Artifacts, and returns `list[JobResult]`.

Each new Execution creates one Session with a Workflow snapshot and content
hash. Resume requires matching Project ID, Workflow ID, and Workflow hash.

## Configuration

- `WorkflowSpec`: versioned Workflow identity, title, Collection, Tags, and Jobs.
- `JobConfig`: one engine, plugin config, structured input, output, ScanSpec, and
  persistence intent.
- `SystemConfig`: machine policy only: result-save default, scan storage and
  checkpoint policy, resource hints, progress, and logging.
- `ProjectManifest`: portable Project identity and Project-relative paths.

Project paths never belong to `SystemConfig`.

## Scan And Execution

`ScanSpec` and `ScanAxisSpec` validate explicit `values`, `linspace`, and
`logspace` axes. `ParameterGrid` is the compiled representation passed to an
engine. A scan remains one logical Job.

`ExecutionContext` carries the grid, resource snapshot, progress reporter,
cancellation token, ArtifactStore, and CheckpointStore. Engines should report
their natural work units through this context.

## Registry

`RegistryCenter` discovers, validates, and creates plugins by namespace and
name. Installed packages use entry points; Project-local plugins are discovered
from the paths declared by `qphase.toml`.

Important methods are `register()`, `register_lazy()`, `create()`, `list()`, and
`get_plugin_schema()`.
