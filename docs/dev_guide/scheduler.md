---
description: Scheduler System
---

# Scheduler System

The scheduler is QPhase's logical workflow orchestrator. It resolves engines and
plugins, validates the DAG, transfers results between jobs, creates reproducible
run directories, reports progress, and delegates persistence to the artifact
store. It does not expand parameter points into jobs and it does not choose an
engine's numerical parallelization strategy.

## Logical Job Lifecycle

For each job in topological order, the scheduler:

1. Resolves the configured engine and validates its `EngineManifest`.
2. Instantiates required and optional plugins through the registry.
3. Compiles an optional `ScanSpec` into a `ParameterGrid`.
4. Resolves the structured upstream input.
5. Creates an `ExecutionContext` and invokes the engine.
6. Saves the final logical result through `ArtifactStore` when requested.
7. Records one status entry in `session_manifest.json`.

A Cartesian scan with shape `(101, 101)` is still one scheduler job. It has one
configuration snapshot, one manifest entry, and one job directory. The engine
may internally use 10,201 points, tiles, chunks, processes, or fused GPU work,
but those are execution details rather than scheduler nodes.

## Engine Manifest

Engines declare plugin requirements before execution:

```python
from qphase.core.protocols import EngineManifest

class MyEngine:
    manifest = EngineManifest(
        required_plugins={"backend", "model"},
        optional_plugins={"analyser"},
    )
```

The scheduler instantiates these plugin classes from validated config and
passes the resulting objects to the engine. Resource packages remain
responsible for deciding which combinations are numerically meaningful.

## ExecutionContext

The scheduler supplies a context containing:

- `parameter_grid`: the compiled `ParameterGrid`, or `None`.
- `resources`: a snapshot of workstation resource hints.
- `progress`: an engine-facing progress reporter.
- `cancellation`: a cancellation token reserved for CLI/service clients.
- `artifacts`: the logical job's `ArtifactStore`.
- `checkpoints`: a chunk-level `CheckpointStore`.

The preferred engine signature is:

```python
def run(self, input_data=None, *, context=None):
    ...
```

The legacy `progress_cb` argument remains available during one compatibility
period, but the scheduler itself uses `ExecutionContext`.

## Scan Ownership

Core provides `ParameterGrid` and a reusable `execute_pointwise()` helper. The
engine decides whether to use that helper or compile the grid into a specialized
strategy. This division is intentional: algorithm-level batching cannot be
selected correctly from backend names alone.

Examples:

- CAM `multistability` owns process tiles.
- CAM `batched_newton` owns NumPy/CuPy batched Newton arrays.
- SDE converts the grid to its existing per-trajectory parameter repetition and
  trajectory fusion representation.
- A simple engine may call `execute_pointwise()` and receive chunk checkpointing
  without implementing a custom planner.

There is no core `JobExpander`, batch negotiator, engine-specific batch planner,
or scheduler result splitter in this path.

## Structured Data Flow

An input declaration identifies an upstream logical result:

```yaml
input:
  from: simulation
  mode: dataset
```

`dataset` passes the complete result once. `map` lazily iterates point or group
views according to `select` and `group_by`, invokes the downstream engine for
each view, and wraps the outputs in one `MappedDatasetResult`. Map iterations do
not receive run directories or manifest entries.

## Session and Persistence

```text
runs/<session-id>/
  session_manifest.json
  scan_job/
    config_snapshot.json
    artifact_manifest.json
    result.npz                 # single layout
    # or result/shard_*.npz    # sharded layout
    .checkpoints/              # retained only when configured or on failure
```

`artifact_manifest.json` describes the result type, schema, axes, shape,
physical layout, files, and loader. `per_point` is an external compatibility
layout only; all files remain under the same logical job directory.

Checkpoint compatibility is tied to the configuration hash, plugin versions,
backend, and dtype. Successful jobs remove checkpoints after the final dataset
is safely stored unless `keep_on_success` is enabled. Current checkpoints cover
completed scan chunks, not an SDE integrator's intermediate time state.

## Planning and Progress

Normal CLI output stays at logical-job granularity: job name, scan shape,
status, and result directory. Axis and chunk details belong to `--plan` or
verbose output. Service DTOs expose scan summaries and internal progress events
for future clients without requiring GUI changes.

Engines report `completed`, `total`, and a natural unit for each stage. Core
estimates rate and remaining time only within the current `(stage, unit)` scope
after a short warm-up. Stage changes reset the estimator. Unknown totals show
elapsed time only, and heterogeneous jobs are never used to extrapolate a
workflow-wide ETA. Workflow progress is the completed logical-job count.

For `input.mode=map`, scheduler progress counts completed views. Child-engine
progress is exposed as verbose status inside the map stage, so a large mapped
dataset does not produce one terminal progress stream per point.
