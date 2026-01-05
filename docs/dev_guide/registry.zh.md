---
description: 注册表系统
---

# 注册表系统

**注册表系统**作为 QPhase 的中央服务定位器和依赖注入容器。它管理所有可扩展组件的生命周期，提供统一的注册、发现和实例化机制。

## 核心架构

注册表实现为一个单例 `RegistryCenter`，维护分层查找表：`命名空间 -> 名称 -> 条目`。

### 命名空间

为了确保模块化并防止命名冲突，插件被隔离到命名空间中。标准命名空间包括：

| 命名空间 | 描述 | 示例 |
|-----------|-------------|---------|
| `backend` | 计算后端 | `numpy`、`torch` |
| `engine`  | 仿真引擎 | `sde`、`viz` |
| `model`   | 物理模型 | `kerr_cavity`、`vdp` |
| `integrator`| 数值积分器 | `euler_maruyama`、`srk` |
| `analyser`| 结果分析工具 | `mean_photon`、`wigner` |

### 条目管理策略

注册表采用双重策略进行条目管理，以平衡启动延迟和运行时灵活性：

1.  **急切条目（Callable）**：
    *   **机制**：在初始化期间直接导入并存储在内存中的插件类或工厂函数。
    *   **应用**：用于核心插件和需要立即可用的测试场景。

2.  **延迟条目（点分路径）**：
    *   **机制**：注册表存储字符串引用（例如 `"pkg.module:ClassName"`）。实际模块导入延迟到第一次请求实例化时。
    *   **应用**：用于第三方插件和可选依赖项。这最小化了应用程序的启动时间和内存占用。

## 发现机制

QPhase 支持两种主要的发现机制：

### 1. 入口点（基于包）
对于可分发的 Python 包，QPhase 使用标准的 `entry_points` 机制（在 `pyproject.toml` 中定义）。注册表在启动时扫描 `qphase` 组。

```toml
[project.entry-points.qphase]
"model.my_model" = "my_package.models:MyModel"
```

### 2. 本地配置（基于开发）
对于本地开发和临时扩展，注册表解析位于项目根目录的 `.qphase_plugins.yaml` 文件。这允许研究人员在不打包脚本的情况下注册它们。

```yaml
model.custom_hamiltonian: "plugins.physics:Hamiltonian"
```

## 实例化工厂

`create()` 方法作为所有组件的通用工厂。它处理：
1.  **解析**：通过命名空间和名称查找条目。
2.  **加载**：如果是延迟条目则导入模块。
3.  **验证**：根据插件的 `config_schema`（Pydantic 模型）验证提供的配置字典。
4.  **注入**：使用已验证的配置和任何额外依赖项（例如将 `backend` 实例注入到 `model` 中）实例化类。

```python
# 示例：带依赖注入的模型实例化
model = registry.create(
    "model:kerr_cavity",
    config={"chi": 1.0},
    backend=numpy_backend_instance
)
```

## 依赖解析

虽然注册表提供了创建插件的机制，但**调度器**充当协调器。它使用所选引擎的 `EngineManifest` 来确定从注册表请求哪些插件。
