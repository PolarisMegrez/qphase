---
layout: default
title: Scheduler System
parent: Developer Guide
nav_order: 2
---

# Scheduler System

The **Scheduler** is the execution orchestration layer of QPhase. It is responsible for translating high-level job definitions into concrete computational tasks, managing their lifecycle, and ensuring data integrity.

## Functional Responsibilities

1.  **Job Expansion**: Transforming abstract job configurations (which may contain parameter ranges) into a linear sequence of atomic tasks.
2.  **Dependency Resolution**: Analyzing the execution graph to ensure jobs are executed in topological order (e.g., ensuring input data exists before a dependent job starts).
3.  **Context Management**: Provisioning isolated execution environments (directories) for each job to prevent data contamination.
4.  **Error Handling**: Intercepting runtime exceptions to prevent batch failures (i.e., a single failed job should not terminate the entire campaign).

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

The Scheduler integrates with the `JobExpander` to support parameter sweeps.

*   **Cartesian Product**: By default, lists in the configuration are interpreted as axes for a grid search. The expander generates the Cartesian product of all iterable parameters.
*   **Zipped Expansion**: For correlated parameters, the expander supports a "zipped" mode, iterating over parameter lists in lockstep.

## Runtime Isolation

To ensure reproducibility, the Scheduler enforces strict runtime isolation:

1.  **Directory Provisioning**: For each job, a directory is created following the pattern `runs/{timestamp}_{job_name}/`.
2.  **Configuration Snapshot**: Before execution begins, the fully resolved configuration (including all defaults and overrides) is serialized to `config_snapshot.json` within the run directory.
3.  **State Reset**: The Registry and Engine are re-initialized for each job to prevent state leakage (e.g., GPU memory fragmentation or global variable pollution) between iterations.
