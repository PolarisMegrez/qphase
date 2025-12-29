---
layout: default
title: Developer FAQ
parent: Developer Guide
nav_order: 11
---

# Developer FAQ

This guide addresses common questions and issues encountered when developing plugins for QPhase.

## Plugin Development

### Why is my plugin not showing up in `qps list`?

If your plugin is not appearing in the registry, check the following:

1.  **Entry Point Configuration**: Ensure your `pyproject.toml` has the correct entry point group `[project.entry-points.qphase]`.
2.  **Installation**: Did you install your package? If you are developing locally, use `pip install -e .` to install in editable mode.
3.  **Namespace**: Ensure your entry point key follows the `namespace.name` format (e.g., `"backend.my_backend"`).
4.  **Cache**: QPhase caches entry points. Try reinstalling your package to refresh the metadata.

### How do I handle optional dependencies?

If your plugin requires a heavy library (like `torch` or `cupy`) that shouldn't be a hard dependency for QPhase, follow these steps:

1.  **Use Lazy Registration**: Register your plugin using a dotted path string so the module isn't imported at startup.
2.  **Import Inside Methods**: Import the heavy library inside your `__init__` or `run` method, not at the top level of your module.
3.  **Fail Gracefully**: Raise a clear `ImportError` if the dependency is missing.

```python
class MyHeavyPlugin:
    def __init__(self, config):
        try:
            import torch
        except ImportError:
            raise ImportError("This plugin requires 'torch'. Please install it.")
```

### Why is my configuration validation failing?

QPhase uses Pydantic for validation. Common issues include:

*   **Type Mismatch**: Passing a string `"1e-3"` to a `float` field. Pydantic usually coerces this, but strict mode might fail.
*   **Missing Fields**: A required field (no default value) is missing from the YAML.
*   **Extra Fields**: Your YAML has fields not defined in the schema. Set `model_config = ConfigDict(extra="allow")` if you want to permit this.

### Why does my Engine fail with "missing required plugins"?

This error comes from the `EngineManifest` validation. Your Engine class likely declares a `manifest` with `required_plugins={"model", ...}`. If the user's Job Configuration does not provide a `model` plugin in the `plugins` section, the Scheduler will block execution.

**Fix**: Ensure your YAML configuration includes all plugins listed in the Engine's `required_plugins`.

## Architecture & Internals

### What is the difference between Eager and Lazy registration?

*   **Eager (`register`)**: You pass the class object directly. The class is imported immediately when the registry is initialized. This is good for core plugins but bad for startup time.
*   **Lazy (`register_lazy`)**: You pass a string path (`"pkg.mod:Class"`). The class is only imported when someone calls `registry.create()` or asks for its schema. This is the recommended way for most plugins.

### How does the Registry find plugins?

The Registry uses Python's standard `importlib.metadata` to scan for entry points in the `qphase` group. It does this once at startup. It also scans directories defined in `SystemConfig.paths.plugin_dirs` for `.qphase_plugins.yaml` files to support local, non-installed plugins.

### Can I override a core plugin?

Yes. If you register a plugin with the same namespace and name as an existing one (e.g., `backend.numpy`), and set `overwrite=True` (or use a higher priority loading mechanism), your plugin will replace the core one. This allows for powerful customization but should be done with caution.
