---
description: 核心 API 参考
---

# 核心 API 参考

本节记录了 QPhase 框架的核心组件，包括调度器、注册表和配置模型。

## 调度器

`Scheduler` 是负责协调仿真任务执行的中央组件。它处理依赖解析、参数扫描和结果持久化。

### `class qphase.core.Scheduler`

**参数：**

*   `system_config` (`SystemConfig`, 可选)：系统配置对象。如果未提供，则从 `system.yaml` 加载。
*   `default_output_dir` (`str`, 可选)：覆盖系统配置中指定的默认输出目录。
*   `on_progress` (`Callable[[JobProgressUpdate], None]`, 可选)：在任务执行期间调用的回调函数，用于接收进度更新。
*   `on_run_dir` (`Callable[[Path], None]`, 可选)：每个任务完成后调用的回调函数，用于接收运行目录的路径。

**方法：**

#### `run(job_list: JobList) -> list[JobResult]`

串行执行任务列表。此方法处理：
1.  **依赖解析**：确保任务按照其依赖关系以正确的顺序执行。
2.  **参数扫描**：将具有可扫描参数的任务展开为多个任务。
3.  **目录管理**：为每次任务执行创建唯一的运行目录。
4.  **快照**：保存配置快照以确保可复现性。

**返回：**
*   `list[JobResult]`：包含每个任务执行状态和元数据的结果对象列表。

---

## 配置

### `class qphase.core.JobConfig`

表示单个仿真任务的配置。

**字段：**

*   `name` (`str`)：**必需。** 任务的唯一标识符。
*   `engine` (`dict[str, Any]`)：**必需。** 仿真引擎的配置。必须恰好包含一个键（引擎名称），映射到其配置字典。
*   `plugins` (`dict[str, dict[str, Any]]`)：**可选。** 插件配置，按插件类型组织（例如 `backend`、`model`）。
*   `params` (`dict[str, Any]`)：**可选。** 任务特定参数的字典。
*   `input` (`str | None`)：**可选。** 上游任务的名称或用作输入的文件路径。
*   `output` (`str | None`)：**可选。** 输出目标（文件名或下游任务名称）。
*   `tags` (`list[str]`)：**可选。** 用于分类的标签列表。
*   `depends_on` (`list[str]`)：**可选。** 此任务依赖的任务名称列表。

### `class qphase.core.SystemConfig`

表示全局系统设置。

**字段：**

*   `paths` (`PathsConfig`)：配置、插件和输出的目录路径。
*   `auto_save_results` (`bool`)：是否自动将结果保存到磁盘。
*   `parameter_scan` (`dict`)：批量执行和参数扫描策略的设置。

---

## 注册表

### `class qphase.core.RegistryCenter`

管理插件的中央注册表。它支持动态发现、注册和工厂式组件实例化。

**方法：**

#### `register(namespace: str, name: str, target: Any)`
注册新插件。

*   `namespace`：插件的类别（例如 "backend"、"model"）。
*   `name`：插件在其命名空间内的唯一名称。
*   `target`：插件类或工厂函数。

#### `create(full_name: str, config: Any = None, **kwargs) -> Any`
实例化插件。

*   `full_name`：插件的完整标识符（例如 "backend:numpy"）。
*   `config`：传递给插件构造函数的配置对象。
*   `**kwargs`：传递给构造函数的额外关键字参数。

#### `list(namespace: str | None = None) -> dict`
列出已注册的插件。

*   `namespace`：如果提供，则将列表过滤到特定命名空间。
*   **返回**：将插件名称映射到其元数据的字典。
