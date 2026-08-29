---
description: Configuration contracts and ownership
---

# Configuration System

Configuration is intentionally split into four strict contracts:

1. `ProjectManifest` (`qphase.project/2`) owns identity and relative paths.
2. `WorkflowSpec` (`qphase.workflow/2`) owns metadata and logical Jobs.
3. `JobConfig` owns one Engine invocation, plugins, scan, input, and save intent.
4. `SystemConfig` owns project-independent machine/runtime policy.

Project plugin defaults are a plain validated mapping at
`ProjectContext.defaults_path`. They fill missing selected plugin values but do
not activate optional plugin namespaces.

## Loading Pipeline

1. `ProjectContext.discover()` resolves exactly one Project.
2. `WorkflowCatalog` recursively indexes metadata and enforces unique IDs.
3. `load_workflow()` rejects legacy documents and validates the strict wrapper.
4. Top-level plugin namespace blocks are extracted into `JobConfig.plugins`.
5. Project defaults are merged with explicit Job configuration.
6. `WorkflowCompiler` uses its Project-scoped `RegistryView` to validate each
   selected plugin and the Engine manifest.
7. The compiler validates the Job graph and freezes a `CompiledWorkflow`.

`JobConfig` validates only the versioned document structure; it never consults
the process-global plugin registry. Plugin validation belongs exclusively to
`WorkflowCompiler`. A persisted `CompiledWorkflow` can therefore be restored in
a clean control-plane process without importing scientific plugins again.

Unknown Workflow fields are forbidden. Job `extra="allow"` exists only because
plugin namespaces are dynamic; unknown non-plugin fields must not be treated as
new core behavior.

## Scans

Only explicit `ScanSpec` creates a scan. Lists in plugin schemas remain literal
scientific values. Compiling produces immutable `ParameterGrid` with stable
axis order. Scheduler forwards it once to the Engine and never expands points
into Jobs.

## System Policy Store

`SystemConfigStore` merges package, site, sparse user, environment-selected,
and explicit policies. Reads do not create files; writes persist only values
different from package/site defaults. Project paths are schema errors.

Tests and clients that need another Session root must create or inject a
different `ProjectContext`; they must not add output-path overrides to
SystemConfig or Scheduler.
