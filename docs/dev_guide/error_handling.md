---
layout: default
title: Error Handling & Logging
parent: Developer Guide
nav_order: 5
---

# Error Handling & Logging

The core package defines a unified exception hierarchy and logging system to provide clear error classification, traceable error chains, and flexible logging configuration.

## Exception Hierarchy

**Exception Tree Structure**:

```
QPhaseError (Base Class)
├── QPhaseIOError          # File/Network I/O errors
├── QPhaseConfigError      # Configuration validation errors
├── QPhasePluginError      # Plugin discovery/instantiation errors
├── QPhaseSchedulerError   # Job scheduling errors
├── QPhaseRuntimeError     # Engine execution errors
└── QPhaseCLIError         # CLI argument/execution errors

QPhaseWarning (Warning Base Class)
```

**Design Principles**:

1.  **Single Root**: All framework exceptions inherit from `QPhaseError`, making them easy to catch.
2.  **Responsibility-Based**: Each exception class corresponds to a specific source of error.
3.  **Chain Preservation**: Uses `raise ... from e` to preserve the original exception stack.
4.  **Warning Separation**: `QPhaseWarning` is independent of the exception hierarchy, used for non-fatal issues.

## Usage Scenarios

### QPhaseIOError
Used for file system and network I/O failures.

```python
# File not found
if not path.exists():
    raise QPhaseIOError(f"File not found: {path}")

# Write failure
try:
    save_global_config(config, path)
except Exception as e:
    raise QPhaseIOError(f"Failed to save global config to {path}: {e}") from e
```

**Typical Triggers**: Missing config files, permission denied, snapshot save failure.

### QPhaseConfigError
Used for configuration validation and parsing errors.

```python
# YAML parse failure
try:
    data = yaml.safe_load(f)
except yaml.YAMLError as e:
    raise QPhaseConfigError(f"Failed to parse YAML file {path}: {e}") from e

# Pydantic validation failure
try:
    validated = schema.model_validate(params)
except ValidationError as e:
    raise QPhaseConfigError(f"Invalid configuration for '{plugin_type}:{name}': {e}") from e
```

**Typical Triggers**: YAML syntax errors, missing required fields, type mismatches.

### QPhasePluginError
Used for plugin discovery, import, and instantiation errors.

```python
# Plugin not registered
if entry is None:
    raise QPhasePluginError(f"Plugin '{nm}' not found in namespace '{ns}'")

# Import failure
try:
    obj = self._import_target(entry.target)
except Exception as e:
    raise QPhasePluginError(
        f"Failed to import plugin '{nm}' from '{entry.target}': {e}"
    ) from e
```

**Typical Triggers**: Requesting unknown plugins, missing dependencies, constructor exceptions.

### QPhaseRuntimeError
Used for errors during engine execution. Wraps exceptions from `Engine.run()`.

```python
try:
    output = engine.run(data=input_data)
except Exception as e:
    log.error(f"Job execution failed: {e}")
    raise QPhaseRuntimeError(
        f"Job '{job.name}' execution failed in engine '{job.get_engine_name()}': {e}"
    ) from e
```

**Typical Triggers**: SDE divergence, model calculation errors, backend operation failures.

## Exception Chaining

The core package uses the `raise ... from e` syntax to preserve the original exception.

```python
try:
    config = schema.model_validate(raw_data)
except ValidationError as e:
    raise QPhaseConfigError(
        f"Invalid configuration for plugin '{name}': {e}"
    ) from e  # Preserves original ValidationError
```

**Benefits**:
1.  Users see high-level error messages (`QPhaseConfigError`).
2.  Developers can access the original exception via `__cause__`.
3.  Full stack traces are available for debugging.

## Logging System

### Singleton Logger

The core package uses a shared logger named `"qphase"`.

```python
_logger: logging.Logger | None = None

def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = logging.getLogger("qphase")
        _logger.setLevel(logging.INFO)
        # Add default console handler
        if not _logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            h.setFormatter(fmt)
            _logger.addHandler(h)
    return _logger
```

### Configuration API

The `configure_logging` function allows customization of the logging behavior.

```python
def configure_logging(
    verbose: bool = False,
    log_file: str | None = None,
    as_json: bool = False,
    suppress_warnings: bool = False,
) -> None:
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `verbose` | `bool` | `False` | If True, sets level to DEBUG. |
| `log_file` | `str` | `None` | Path to log file (append mode). |
| `as_json` | `bool` | `False` | Output logs in JSON Lines format. |
| `suppress_warnings` | `bool` | `False` | Elevate warnings to ERROR level. |

### Best Practices

**For Plugin Developers**:
1.  **Don't swallow exceptions**: Let errors propagate to the scheduler.
2.  **Add context**: Include useful information when re-raising.
3.  **Use correct types**: Use `QPhaseConfigError` for config issues.
4.  **Log before raising**: Log DEBUG info before raising an exception.

```python
def my_plugin_method(self):
    log = get_logger()
    try:
        result = risky_operation()
    except SomeError as e:
        log.debug(f"Operation failed with details: {e}")
        raise QPhaseRuntimeError(f"MyPlugin operation failed: {e}") from e
```
