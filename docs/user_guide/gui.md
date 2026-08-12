---
description: Local GUI and long-running Execution management
---

# GUI And Local API

QPhase Workbench is a visual client of the same service layer used by the CLI.
It manages Workflows, queued/running Executions, Session history, logs, and
Artifacts. Engines retain ownership of scientific algorithms and internal
parallelism.

## Start

```bash
pip install "qphase[gui]"
qphase gui
```

Open `http://127.0.0.1:8000`. Public bind addresses are rejected because this
local API has no authentication. Use an SSH tunnel for a remote workstation or
lab server.

## Long-running Executions

Submitting a Workflow creates an asynchronous Execution. The local Workbench
currently runs one active Execution and keeps a bounded FIFO queue. Closing the
browser does not stop the server-side worker.

- The Execution view shows concise current state, active Job/engine stage, scan
  progress, and configured plugins.
- `events.jsonl` stores sampled progress and control events.
- Session logs store complete diagnostics according to `SystemConfig`.
- `session.lock` provides an owner heartbeat. A stale `running` Session is shown
  as `interrupted` without rewriting its manifest.

Cancellation is cooperative; QPhase does not forcibly terminate an active GPU
kernel.

## Pause And Revise

Pause takes effect at a logical Job boundary. While queued or paused, an
unstarted Job may be replaced with a complete configuration preserving its
name. QPhase validates the revised graph and plugin configuration and journals
the revision.

This supports a long `SDE -> analysis` Workflow: after SDE completes but before
analysis starts, the pending analysis Job can be corrected. It cannot restore
data that the upstream Job discarded. For example, an analyzer requiring raw
trajectories cannot be appended after `keep_traj: false` has dropped them.

## Workflows And Sessions

The Workflow catalog supports text search and clickable Collection/Tag filters,
so large Projects do not require browsing one flat list.

Workflow documents are edited with content revisions, so stale GUI writes fail
instead of overwriting concurrent IDE edits. Sessions expose status, alias,
notes, logical Jobs, events, logs, and Artifacts. Deletion first moves a
non-running Session to Project-local trash; purge is explicit.

Future Archive metadata may provide virtual folders, favorites, and private
notes keyed by Project/Workflow/Session IDs. It remains user-local and cannot
affect execution or reproducibility.

## API Surface

| Endpoint | Purpose |
| --- | --- |
| `GET /workflows` | List Workflow catalog entries. |
| `POST /plans` | Build a Workflow Execution plan. |
| `POST /executions` | Validate and enqueue a Workflow. |
| `GET /executions/{id}` | Read active or recent Execution state. |
| `GET /executions/{id}/events` | Read live events after a sequence cursor. |
| `POST /executions/{id}/pause` | Pause at the next Job boundary. |
| `PUT /executions/{id}/jobs/{name}` | Revise an unstarted Job. |
| `POST /executions/{id}/resume` | Resume a paused Execution. |
| `POST /executions/{id}/cancel` | Request cooperative cancellation. |
| `GET /sessions` | List persisted Session history. |
| `GET /sessions/{id}/events` | Read persisted Session events. |
| `GET /sessions/{id}/artifacts` | List Session Artifacts. |
| `GET /workflow-docs` | List editable Workflow documents. |

## Current Boundaries

The GUI does not yet provide multi-user authentication, public network serving,
multi-Execution resource scheduling, plugin hot reload, SDE time-step
checkpointing, online FFT, or Archive virtual folders. These are explicit
future capabilities, not hidden behaviors of the current API.
