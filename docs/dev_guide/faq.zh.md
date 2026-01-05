---
description: 开发者常见问题
---

# 开发者常见问题

本指南解答为 QPhase 开发插件时遇到的常见问题。

## 插件开发

### 为什么我的插件没有在 `qphase list` 中显示？

如果您的插件没有出现在注册表中，请检查以下几点：

1.  **入口点配置**：确保您的 `pyproject.toml` 具有正确的入口点组 `[project.entry-points.qphase]`。
2.  **安装**：您安装了您的包吗？如果您在本地开发，请使用 `pip install -e .` 以可编辑模式安装。
3.  **命名空间**：确保您的入口点键遵循 `namespace.name` 格式（例如 `"backend.my_backend"`）。
4.  **缓存**：QPhase 缓存入口点。尝试重新安装您的包以刷新元数据。

### 如何处理可选依赖项？

如果您的插件需要一个重量级库（如 `torch` 或 `cupy`），而该库不应该是 QPhase 的硬依赖项，请按照以下步骤操作：

1.  **使用延迟注册**：使用点分路径字符串注册您的插件，这样模块在启动时不会被导入。
2.  **在方法内导入**：在您的 `__init__` 或 `run` 方法内导入重量级库，而不是在模块顶层。
3.  **优雅失败**：如果缺少依赖项，则引发清晰的 `ImportError`。

```python
class MyHeavyPlugin:
    def __init__(self, config):
        try:
            import torch
        except ImportError:
            raise ImportError("This plugin requires 'torch'. Please install it.")
```

### 为什么我的配置验证失败？

QPhase 使用 Pydantic 进行验证。常见问题包括：

*   **类型不匹配**：将字符串 `"1e-3"` 传递给 `float` 字段。Pydantic 通常会强制转换，但严格模式可能会失败。
*   **缺少字段**：YAML 中缺少必需字段（无默认值）。
*   **额外字段**：您的 YAML 有模式中未定义的字段。如果您想允许这种情况，请设置 `model_config = ConfigDict(extra="allow")`。

### 为什么我的引擎因"缺少必需插件"而失败？

此错误来自 `EngineManifest` 验证。您的引擎类可能声明了一个带有 `required_plugins={"model", ...}` 的 `manifest`。如果用户的任务配置在 `plugins` 部分未提供 `model` 插件，调度器将阻止执行。

**修复**：确保您的 YAML 配置包含引擎 `required_plugins` 中列出的所有插件。

## 架构和内部机制

### 急切注册和延迟注册有什么区别？

*   **急切（`register`）**：您直接传递类对象。当注册表初始化时，类会立即被导入。这对核心插件很好，但对启动时间不利。
*   **延迟（`register_lazy`）**：您传递一个字符串路径（`"pkg.mod:Class"`）。只有当有人调用 `registry.create()` 或请求其模式时，类才会被导入。这是大多数插件的推荐方式。

### 注册表如何查找插件？

注册表使用 Python 的标准 `importlib.metadata` 扫描 `qphase` 组中的入口点。它在启动时执行一次。它还扫描 `SystemConfig.paths.plugin_dirs` 中定义的目录以查找 `.qphase_plugins.yaml` 文件，以支持本地非安装插件。

### 我可以覆盖核心插件吗？

可以。如果您使用与现有插件相同的命名空间和名称（例如 `backend.numpy`）注册插件，并设置 `overwrite=True`（或使用更高优先级的加载机制），您的插件将替换核心插件。这允许强大的自定义，但应谨慎操作。
