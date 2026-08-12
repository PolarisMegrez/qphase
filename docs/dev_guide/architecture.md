---
description: QPhase 2 architecture
---

# Architecture

QPhase is a project-oriented scientific workflow runtime. Core owns portable
Project context, Workflow validation, logical-Job orchestration, Execution
control, Session records, progress/logging, and Artifact persistence. Resource
packages own domain algorithms and their CPU/GPU execution strategies.

## Boundaries

```text
CLI / GUI / Python client
          |
qphase.service
          |
qphase.core: Project -> Workflow -> Execution -> Session -> Artifact
          |
resource-package Engine
          |
model / backend / solver / analyser / postprocessor plugins
```

The CLI and GUI are peer clients of `qphase.service`; neither wraps the other.
The CLI remains the complete interface. The GUI adds visual interaction but no
separate execution semantics.

## Runtime Concepts

- `ProjectContext` resolves a strict `qphase.project/2` manifest and all
  Project-relative paths.
- `WorkflowSpec` is a strict `qphase.workflow/2` document containing stable
  metadata and logical `JobConfig` nodes.
- `ExecutionManager` owns the local asynchronous queue and cooperative control.
- `Scheduler` executes one Workflow graph and persists one Session.
- `ArtifactStore` saves each logical result and describes physical layout.
- `ProjectService` indexes Workflow documents and Session history.

Stable IDs identify Projects, Workflows, Executions, Sessions, and Artifacts.
Paths are locations and must not become cross-client identity.

## Engine And Plugin Model

An Engine is the only scheduler-facing entry point of a resource package. It
declares plugin slots through `EngineManifest`; scheduler validates and injects
selected plugin instances. An Engine is not a Workflow: a Workflow may connect
several Jobs using different Engines.

Plugins own strict Pydantic configuration schemas and capability protocols.
Child-plugin slots may expose internal strategy families such as PSD estimators
without flattening them into unrelated top-level namespaces.

Core does not infer scientific parallelism from backend names. `ScanSpec`
compiles to `ParameterGrid`, then the Engine chooses pointwise, tiled, fused,
process, or GPU execution. A 101 x 101 scan remains one logical Job and one
Dataset Artifact.

## Configuration Ownership

| Owner | Content |
| --- | --- |
| `qphase.toml` | Project identity and relative Workflow/default/plugin/Session paths. |
| `configs/defaults.yaml` | Project-wide plugin defaults. |
| Workflow document | Scientific intent, Job graph, scans, and data flow. |
| `SystemConfig` | Project-independent machine policy, progress/logging, storage policy, and resource hints. |
| plugin schema | Plugin-specific values and validation. |
| Engine manifest | Required and optional plugin namespaces. |

SystemConfig must not contain Project paths. Dynamic hardware observations are
sampled into `ResourceSnapshot`, not persisted as configuration truth.

## Execution Lifecycle

1. Discover Project and load `qphase.toml`.
2. Discover package entry points and Project-local plugins.
3. Resolve a Workflow by stable ID or Project-relative path.
4. Validate Workflow schema, Job graph, Engine manifests, and plugin schemas.
5. Create an Execution; scheduler creates one Session attempt.
6. For each logical Job, resolve inputs and plugins, compile `ParameterGrid`,
   and construct `ExecutionContext`.
7. Invoke the resource Engine once for that logical Job.
8. Persist snapshots, events, logs, Artifacts, and manifest status.

Execution control is cooperative. Pause/revision occurs at Job boundaries;
cancellation is observed where an Engine checks its token. Core does not kill
GPU kernels or schedule several Executions against shared resources yet.

## Session Layout

```text
runs/YYYY/MM/<session-id>/
  session_manifest.json
  events.jsonl
  qphase.log
  <job-name>/
    config_snapshot.json
    artifact_manifest.json
    result.npz
    result/shard_*.npz
```

The physical `single`, `sharded`, or compatibility `per_point` layout does not
change the logical Dataset shape. No parameter point receives a Session or Job
directory.

## Extension Rule

Reusable lifecycle and infrastructure behavior belongs in core. Scientific
decisions, memory models, batching, fused kernels, and domain postprocessing
belong in resource packages. A feature should enter core only when at least two
resource packages need the same domain-independent contract.
