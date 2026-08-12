---
description: QPhase project and execution concepts
---

# Core Concepts

QPhase uses a small, explicit vocabulary across the CLI, GUI, service API, and
storage formats.

| Term | Contract |
| --- | --- |
| **Project** | A portable research boundary declared by `qphase.toml`. It owns Workflow documents, Project defaults, local plugin roots, and Session storage. |
| **Workflow** | A versioned YAML document with stable `id`, metadata, and a graph of logical Jobs. |
| **Job** | One logical Workflow node. A parameter scan is still one Job and produces one logical dataset. |
| **Execution** | One queued or running attempt to execute a Workflow. Retrying creates another Execution. |
| **Session** | The persisted record of one Execution, including manifest, snapshots, logs, events, and Artifacts. |
| **Artifact** | A typed output produced by a Job and described by an artifact manifest. |
| **Collection** | Portable, version-controlled grouping metadata stored in a Workflow. Directory placement may mirror it but is not identity. |
| **Tag** | Portable, many-to-many Workflow metadata used for discovery and filtering. |
| **Archive** | User-local organization state such as favorites, aliases, notes, and virtual folders. It is not part of reproducibility. |

## Identity And Location

Project and Workflow identities are stable; filesystem paths are locations.
Moving a Project does not change its `project_id`, and moving a Workflow inside
`configs/workflows/` does not change its `id`. The CLI and GUI therefore refer
to Workflows by ID and Sessions by Session ID rather than treating paths as
identities.

`qphase.toml` is the only source of Project paths. `SystemConfig` contains
machine policy and hardware/resource hints only. This separation allows the
same portable Project to run on another workstation or server without carrying
the original user's absolute paths.

## Reproducibility Boundary

Workflow metadata, Collection, Tags, plugin defaults, and local plugin code
belong to the Project and should be version controlled. GUI Archive metadata is
user-local and may reference `(project_id, workflow_id)` or `(project_id,
session_id)`; it must never be required to execute or reproduce a Workflow.

The CLI remains the complete automation interface. The GUI is a visual client
of the same service layer and does not introduce a second execution model.
