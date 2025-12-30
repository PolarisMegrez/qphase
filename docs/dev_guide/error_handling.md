---
description: Error Handling Strategy
---

# Error Handling Strategy

QPhase implements a structured error handling strategy designed to provide clear, actionable feedback to users while maintaining robust execution control for the scheduler.

## Exception Hierarchy

All framework-specific exceptions inherit from a common base class, `QPhaseError`. This allows the top-level CLI wrapper to catch known errors and display clean messages without printing stack traces (unless verbose mode is enabled).

```python
class QPhaseError(Exception):
    """Base class for all QPhase exceptions."""

class QPhaseConfigError(QPhaseError):
    """Raised when configuration validation fails."""

class QPhasePluginError(QPhaseError):
    """Raised when plugin loading or instantiation fails."""

class QPhaseRuntimeError(QPhaseError):
    """Raised during simulation execution."""

class QPhaseIOError(QPhaseError):
    """Raised during file input/output operations."""
```

## Error Propagation

1.  **Validation Phase**: Errors during configuration loading (e.g., missing fields, invalid types) are caught early and raised as `QPhaseConfigError`. The CLI displays the specific validation message from Pydantic.
2.  **Dependency Check**: If a job is missing required plugins (as defined in `EngineManifest`), a `QPhaseConfigError` is raised before execution begins.
3.  **Execution Phase**: Exceptions occurring within a job (e.g., numerical instability, runtime assertions) are caught by the `Scheduler`.
    *   The exception is logged.
    *   The job is marked as `failed` in the `JobResult`.
    *   The scheduler proceeds to the next independent job (unless `fail_fast` is enabled).

## Logging

QPhase uses the standard Python `logging` module.
*   **Console**: By default, only `INFO` level and above are shown.
*   **File**: If configured, all logs (including `DEBUG`) are written to a log file.
*   **Format**: Logs include timestamps and module names to aid in debugging.
