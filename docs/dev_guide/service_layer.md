---
description: Service layer
---

# Service Layer

The service layer is the structured Python API shared by the CLI, local GUI,
notebooks, and automation:

```text
client -> qphase.service -> qphase.core -> resource engine -> plugins
```

Clients render data; services own application use cases; core owns execution
contracts. No client should shell out to another client or copy scheduler rules.

## Services

- `ConfigService`: load Project defaults, preview merged Job config, validate
  plugins, and access SystemConfig separately.
- `RegistryService`: discover plugins and expose schemas and manifests.
- `SchedulerService`: load Workflows, build side-effect-light Execution plans,
  execute through the core Scheduler, and inspect Session Artifacts.
- `ExecutionManager`: queue asynchronous Executions, stream events, cancel,
  pause at Job boundaries, and revise Jobs that have not started.
- `ProjectService`: list Session history, classify stale Sessions, manage aliases
  and trash, and edit Workflow documents with revision checks.

Service return values in `qphase.service.models` are serializable DTOs. They use
the canonical Project, Workflow, Job, Execution, Session, and Artifact terms.

## Planning Boundary

`SchedulerService.build_plan()` validates Jobs, plugin requirements, scans,
dependencies, and expected Artifacts without creating a Session or importing
runtime-only engine state. Parameter points are summarized, never enumerated as
Jobs.

## Configuration Ownership

- `qphase.toml`: Project identity and portable Project-relative paths.
- Project defaults: reproducible plugin defaults.
- Workflow: scientific intent and logical Job graph.
- Plugin schema: plugin-specific validation and defaults.
- Engine manifest: required and optional plugin namespaces.
- SystemConfig: user/machine runtime policy with no Project paths.

Keep service methods thin. When CLI and GUI require the same rule, implement it
once in core or a service and return structured data to both clients.
