---
description: Scheduler System
---

# Scheduler System

The **Scheduler** is the execution orchestration layer of QPhase. It is responsible for translating high-level job definitions into concrete computational tasks, managing their lifecycle, and ensuring data integrity.

## Functional Responsibilities

1.  **Job Expansion**: Transforming abstract job configurations (which may contain parameter ranges) into a linear sequence of atomic tasks.
2.  **Dependency Resolution**: Analyzing the execution graph to ensure jobs are executed in topological order (e.g., ensuring input data exists before a dependent job starts).
3.  **Context Management**: Provisioning isolated execution environments (directories) for each job to prevent data contamination.
4.  **Error Handling**: Intercepting runtime exceptions to prevent batch failures (i.e., a single failed job should not terminate the entire campaign).

## Dependency Validation

The Scheduler enforces explicit dependency contracts declared by Engines via the `EngineManifest`.

1.  **Manifest Declaration**: Each Engine declares its required and optional plugins.
    ```python
    class MyEngine(EngineBase):
        manifest = EngineManifest(
            required_plugins={"backend", "model"},
            optional_plugins={"analyser"}
        )
    ```
2.  **Pre-flight Check**: Before any job is executed, the Scheduler validates that the job configuration provides all `required_plugins` declared by the target Engine. This prevents runtime failures due to missing dependencies.

## Plugin Instantiation

For each job, the Scheduler performs the following steps to instantiate the environment:

1.  **Engine Resolution**: The `engine` plugin is instantiated first.
2.  **Manifest Inspection**: The Scheduler reads `engine.manifest`.
3.  **Dependency Injection**: The Scheduler iterates through `required_plugins` and `optional_plugins`.
    *   It looks up the corresponding configuration in the Job Config.
    *   It calls `registry.create()` for each plugin.
    *   It collects these instances into a dictionary.
4.  **Engine Execution**: Finally, the Engine is initialized with the dictionary of instantiated plugins and the simulation is launched.

## The Execution Graph

While the current implementation primarily supports serial execution, the underlying data structure (`JobList`) is designed as a directed acyclic graph (DAG).

### JobResult Encapsulation

The outcome of every execution is encapsulated in a `JobResult` object, which serves as the contract between the Scheduler and the reporting system.

```python
@dataclass
class JobResult:
    job_index: int           # Topological index
    job_name: str            # Unique identifier
    run_dir: Path            # Isolated output directory
    run_id: str              # Timestamped UUID
    success: bool            # Execution status
    error: str | None = None # Exception trace if failed
```

## Parameter Scanning Logic

The Scheduler integrates with the `JobExpander` to support parameter sweeps. This process transforms a single "Job Definition" (which may contain lists of values) into multiple atomic "Job Configurations".

### Detection Mechanism
The `JobExpander` inspects the configuration dictionary for lists.
*   **Scanable Parameters**: By default, **any list** found in a plugin configuration (e.g., `model.chi: [0.1, 0.2, 0.3]`) is treated as a parameter to be scanned.
*   **Non-Scanable Lists**: To pass a list as a literal value (e.g., a vector `[1, 0, 0]`), the plugin must explicitly mark that field as non-scanable in its schema, or the user must wrap it (implementation dependent, currently all lists are candidates for expansion if the plugin registers them as scanable).

### Expansion Strategies
1.  **Cartesian Product (Default)**:
    *   If multiple parameters are lists, QPhase generates every possible combination.
    *   Example: `A=[1, 2]`, `B=[3, 4]` -> `(1,3), (1,4), (2,3), (2,4)`.
    *   Result: 4 separate Jobs.

2.  **Zipped Expansion**:
    *   Iterates over parameters in lockstep. Requires all lists to have the same length.
    *   Example: `A=[1, 2]`, `B=[3, 4]` -> `(1,3), (2,4)`.
    *   Result: 2 separate Jobs.

## Runtime Isolation & Session Management

QPhase uses a **Session-Based I/O** strategy to manage execution contexts.

### Session Structure
Every execution command (e.g., `qps run ...`) initiates a new **Session**. A session acts as a container for all jobs executed in that command, providing a shared context for data exchange and logging.

**Directory Layout:**
```text
runs/
  2025-12-31T10-00-00_a1b2c3/      <-- Session Root (Timestamp + Short UUID)
    ├── session_manifest.json      <-- Session Metadata & State
    ├── job_01_sde/                <-- Job Directory
    │     ├── config_snapshot.json
    │     └── result.h5
    └── job_02_viz/                <-- Job Directory
          ├── config_snapshot.json
          └── plot.png
```

### Session Manifest
The `session_manifest.json` file serves as the "brain" of the session, recording:
- **Session ID**: Unique identifier.
- **Status**: Global status (running, completed, failed).
- **Job Registry**: A map of all jobs, their status, output paths, and dependencies.

This manifest enables downstream features like **Resume Capability** (restarting failed jobs) and **DAG Visualization**.

### Job Isolation
Within a session, each job runs in its own subdirectory (`session_dir / job_name`).
*   **Concurrency Safety**: Jobs write to exclusive paths.
*   **Traceability**: Each directory contains a `config_snapshot.json` that records the *exact* scalar values used for that specific run.
