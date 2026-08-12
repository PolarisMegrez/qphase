---
description: Results and reproducibility
---

# Results And Reproducibility

Each Workflow Execution creates one immutable **Session** under the current
Project. A Session records the exact Workflow, resolved Job configuration,
events, logs, and typed Artifacts.

## Session Layout

The Session root comes only from `qphase.toml`. The standard layout is:

```text
runs/YYYY/MM/<session-id>/
  session_manifest.json
  workflow_snapshot.yaml
  events.jsonl
  qphase.log
  simulate/                       # logical Job name
    config_snapshot.json
    artifact_manifest.json
    result.npz
  fit/
    config_snapshot.json
    artifact_manifest.json
    fit_results.csv
```

A parameter scan remains one Job and one logical dataset. It never creates one
Session or scheduler Job per parameter point.

## Reproducibility Records

`session_manifest.json` records `project_id`, `workflow_id`, a canonical
`workflow_hash`, Session status, and Job status. `workflow_snapshot.yaml` is the
complete `qphase.workflow/2` document captured when the Session starts.

Each Job's `config_snapshot.json` records merged Project defaults, resolved
plugin configuration, environment details, and the Job's input/output relation.
It is an audit record, not a standalone Workflow and cannot be passed directly
to `qphase run`.

Resume an interrupted Session with the same Project and unchanged Workflow:

```powershell
qphase run <workflow-id> --resume-from runs/2026/08/<session-id>
```

QPhase rejects resume when the Project ID, Workflow ID, or Workflow content hash
does not match. To create a modified experiment, edit or duplicate a Workflow
document and start a new Execution instead.

## Artifact Manifests

Every saved logical result has an `artifact_manifest.json` describing its result
type, schema version, named axes, logical shape, storage layout, physical files,
and loader. `system.scan_runtime.storage_layout` controls physical storage:

- `single`: one primary dataset file.
- `sharded`: a bounded collection of chunk files in the same Job directory.
- `per_point`: compatibility export; files still share one Job directory.
- `auto`: select `single` or `sharded` from the configured size threshold.

The packaged defaults use a 512 MiB automatic threshold and 128 MiB target
shards. Resource-package loaders restore the same logical dataset independently
of physical layout.

## SDE Artifacts

The SDE engine stores NumPy `.npz` datasets. Common fields include `t0`, `t1`,
`dt`, `meta`, and `analysis`; raw `data` with shape
`(n_traj, n_time, n_modes)` is included only when trajectory retention is
enabled. Analyser outputs are stored by key under `analysis`, for example PSD,
Allan variance, coherence, distributions, or moment summaries.

When `engine.sde.keep_traj` is false, analysis results remain available while
raw trajectories are discarded after analysis. See the
[SDE output reference](../api/qphase_sde/output.md) for its current schema.

## Cross-Job Postprocessing

Postprocessing is represented as downstream Jobs in the same Workflow. For
example, `analyser.lorentz_fitter` consumes a saved or in-memory PSD dataset in
`engine.sde.mode: analyze` and may produce `fit_results.csv` and
`psd_merged.csv`. These files are Artifacts of the downstream Job rather than
independent runs.

```yaml
schema: qphase.workflow/2
id: demo_psd_fit
title: Demo PSD fit
collection: examples
tags: [sde, psd]
jobs:
  - name: simulate
    save: true
    scan:
      axes:
        omega_a:
          target: model.kerr_2mode.omega_a
          values: [0.9, 1.0, 1.1]
    engine:
      sde: {t0: 0.0, t1: 1.0, dt: 0.01, n_traj: 8, seed: 42}
    analyser:
      psd: {modes: [0], kind: complex, find_peaks: true}

  - name: fit
    input: {from: simulate, mode: dataset}
    save: true
    engine:
      sde: {mode: analyze}
    analyser:
      lorentz_fitter: {scan_param: omega_a, mode: 0}
```
