---
layout: default
title: Scheduler System
parent: Developer Guide
nav_order: 2
---

# Scheduler System

The **Scheduler** is the execution management layer of QPhase. It is responsible for translating a `JobList` into an actual computational workflow.

## Design Goals

*   **Orchestration**: Convert high-level job definitions into executable steps.
*   **Dependency Management**: Handle data flow between jobs (input/output).
*   **Plugin Coordination**: Build and coordinate Backends, Integrators, and Engines.
*   **Isolation**: Create independent directories and environments for each run.
*   **Fault Tolerance**: Ensure that a single failed job does not crash the entire batch.

## JobResult: Encapsulating Execution

The `JobResult` dataclass encapsulates the outcome of a single job execution.

```python
@dataclass
class JobResult:
    job_index: int           # Position in the job list (0-based)
    job_name: str            # Job name (for logging)
    run_dir: Path            # Path to the run directory
    run_id: str              # Unique run identifier
    success: bool            # Whether execution was successful
    error: str | None = None # Error message (if failed)
```

### Key Fields

1.  **`run_id`**: A unique string like `"2024-01-01T12-00-00Z_abc123ef"`. It combines a UTC timestamp (to avoid timezone confusion) with a UUID suffix (to prevent filename collisions).
2.  **`run_dir`**: The directory where all outputs for this job are saved. This ensures isolation between runs.
3.  **`success` / `error`**: Provides a clear status signal to the scheduler, allowing it to decide whether to proceed with dependent jobs.

## Execution Flow

The `Scheduler.run()` method implements a serial execution loop.

```python
def run(self, job_list: JobList) -> list[JobResult]:
    # 0. Pre-processing: Expand parameter scans
    job_list = self._expand_parameter_scans(job_list)

    results: list[JobResult] = []
    job_outputs: dict[str, Any] = {}  # Cache for job outputs

    for job_idx, job in enumerate(job_list.jobs):
        # 1. Resolve Input (Memory / File / Loader)
        input_data = self._resolve_input(job, job_outputs)

        # 2. Execute Job (Build plugins, create engine, run)
        result, output = self._run_job(job, job_idx, len(job_list.jobs), input_data)
        results.append(result)

        # 3. Cache Output (if successful)
        if result.success:
            job_outputs[job.name] = output

    return results
```

### Step 0: Parameter Scan Expansion
Before execution, the scheduler detects any parameters that are lists (e.g., `mu: [1.0, 2.0]`) and expands them into multiple individual jobs. This allows for easy parameter sweeps without modifying the core execution logic.

### Step 1: Input Resolution
The `_resolve_input` method handles data dependency.

*   **No Input**: If `job.input` is None, the job runs independently.
*   **Memory Transfer**: If `job.input` matches the name of a previous job, its output object is passed directly in memory. This avoids serialization overhead for complex objects like GPU tensors.
*   **File Loading**: If `job.input` is a file path, a configured `Loader` plugin is used to read it.

### Step 2: Job Execution (`_run_job`)

This method manages the lifecycle of a single job:

1.  **Generate ID**: Create a unique `run_id` and `run_dir`.
2.  **Merge Config**: Combine global defaults with job-specific overrides.
3.  **Build Plugins**: Instantiate all required plugins (Backend, Integrator, etc.).
4.  **Instantiate Engine**: Create the Engine, passing the plugins to it.
5.  **Run**: Call `engine.run(data=input_data)`.
6.  **Snapshot**: Save the full configuration to `config_snapshot.json`.

## Plugin Construction

The `_build_plugins` method unifies the instantiation of all plugins.

```python
def _build_plugins(self, plugins_config: dict[str, Any]) -> dict[str, Any]:
    plugins: dict[str, Any] = {}

    for plugin_type, config_data in plugins_config.items():
        # config_data: {name: "...", params: {...}}
        instance = registry.create_plugin_instance(plugin_type, config_data)
        plugins[plugin_type] = instance

    return plugins
```

This design ensures that the Scheduler doesn't need to know about specific plugin types. It simply delegates creation to the Registry based on the configuration.

## Engine Instantiation

The Engine is a special plugin that receives the other plugins as dependencies.

```python
engine = registry.create_plugin_instance(
    "engine",
    engine_config_raw,
    plugins=plugins  # Engine manages other plugins
)
```

**Why pass plugins to the Engine?**
1.  **Coordination**: The Engine needs to use the Backend to perform math and the Integrator to advance time.
2.  **Lifecycle**: The Engine is responsible for the simulation loop, so it "owns" the execution context.
- `data` 参数：来自 `_resolve_input()` 的结果
- 支持 None、内存对象、文件路径

**输出**：
- 推荐返回 `ResultBase` 对象（支持 save/load）
- 也可返回任意对象（更灵活但不可重现）

### 5.7 运行目录管理 - _generate_run_id() 与 _create_run_dir()

**运行 ID 生成**：
```python
def _generate_run_id(self) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{ts}_{uuid.uuid4().hex[:8]}"
```

**设计考虑**：
- **UTC 时间戳**：避免时区混淆
- **可排序**：按时间排序即可得到执行顺序
- **UUID 后缀**：防止文件名冲突
- **8 位 UUID**：足够唯一性且文件名简洁

**目录创建**：
```python
def _create_run_dir(self, job: JobConfig, run_id: str) -> Path:
    # 获取有效的系统配置（job.system 覆盖全局配置）
    effective_system = job.system if job.system is not None else self.system_config
    output_dir = effective_system.paths.output_dir

    output_root = Path(output_dir).resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
```

**目录层级**：
```
output_root/           # 从配置获取
    └── {run_id}/      # 每次运行唯一目录
        ├── config_snapshot/  # 运行快照
        ├── logs/             # 日志文件
        └── ...               # 其他输出文件
```

**目录隔离的优势**：
1. **可重现性**：每次运行产生独立目录，避免覆盖
2. **调试**：失败的运行可保留现场供分析
3. **并行安全**：即使并行执行也不会相互干扰

### 5.8 快照机制 - _write_snapshot()

**功能**：保存任务运行的完整配置快照，保证可重现性。

**快照数据**：
使用 `ConfigSnapshot` 模型保存完整的运行信息：
```python
snapshot_data = ConfigSnapshot(
    job_name=job.name,
    job_config=job.model_dump(),  # 完整JobConfig
    job_index=job_idx,
    system_config=system_config.model_dump(),
    plugin_configs=validated_plugins,  # 已验证的插件配置
    engine_config=engine_config,
    run_id=run_id,
    run_dir=run_dir,
    input_job=job.input,
    output_job=job.output,
)
```

**快照保存**：
```python
from qphase.core.snapshot import SnapshotManager

# 创建快照管理器
snapshot_manager = SnapshotManager(self.system_config.paths.output_dir)

# 创建快照
snapshot = snapshot_manager.create_snapshot(
    job=job,
    job_index=job_idx,
    system_config=self.system_config,
    validated_plugins=validated_plugins,
    engine_config=engine_config,
    run_id=run_id,
    run_dir=run_dir,
    input_job=job.input,
    output_job=job.output,
    metadata={
        "scheduler_version": "2.0",
        "snapshot_created_by": "scheduler",
    },
)

# 保存快照到JSON文件
snapshot_path = snapshot_manager.save_snapshot(snapshot, run_dir)
```

**设计特点**：

1. **完整配置**：
   - 包含 JobConfig 和 SystemConfig 的完整信息
   - 保存已验证的插件配置（确保配置有效性）
   - 包含运行元数据（时间戳、路径等）
   - 便于重现实验结果

2. **结构化存储**：
   - 使用 Pydantic 模型确保数据一致性
   - 自动处理 Path 对象序列化
   - JSON 格式易于阅读和解析

3. **可扩展功能**：
   - `list_snapshots()`：列出所有快照
   - `get_latest_snapshot()`：获取最新快照
   - `compare_snapshots()`：比较两个快照
   - `export_snapshot()`：导出独立快照文件
   - `create_reproduction_script()`：生成重现脚本

4. **错误处理**：
   - 快照失败不影响任务执行（使用 try/except 包装）
   - 记录警告日志但不中断流程
   - 保证核心功能的稳定性

### 5.9 错误处理与容错设计

**错误分层**：

1. **插件构建错误**（`_build_plugins()`）：
   - 原因：插件不存在、配置无效、依赖缺失
   - 处理：抛出 `QPhasePluginError`，标记任务失败
   - 影响：不影响其他任务

2. **引擎实例化错误**（`_run_job()`）：
   - 原因：引擎类不存在、插件传递失败
   - 处理：包装为 `QPhasePluginError`，标记任务失败
   - 影响：不影响其他任务

3. **执行错误**（`engine.run()`）：
   - 原因：数值计算异常、资源不足、算法失败
   - 处理：捕获异常，包装为 `QPhaseRuntimeError`
   - 影响：不影响其他任务

**容错策略**：

1. **早期验证**：
   - 配置阶段验证插件存在性和参数有效性
   - 避免运行时才发现根本性错误

2. **详细错误**：
   - 所有异常包含上下文信息（任务名、插件名等）
   - 便于快速定位和解决问题

3. **继续执行**：
   - 单个任务失败不影响其他任务
   - 适合批量实验场景（部分失败仍能得到部分结果）

### 5.10 回调机制 - on_progress 与 on_run_dir

**进度回调**（`on_progress`）：

进度回调使用 `JobProgressUpdate` dataclass 传递进度信息：
```python
@dataclass
class JobProgressUpdate:
    job_name: str                    # 任务名称
    job_index: int                   # 任务索引（从0开始）
    total_jobs: int                  # 总任务数
    percent: float                   # 进度百分比（0.0-100.0）
    message: str                     # 状态消息
    stage: str | None                # 当前阶段名称（可选）
    total_duration_estimate: float | None  # 预估总耗时（秒，可选）
    has_progress: bool               # Engine 是否支持进度报告

on_progress: Callable[[JobProgressUpdate], None]
```

**用途**：
- CLI 进度条显示
- Web UI 实时更新
- 日志记录和监控

**目录回调**（`on_run_dir`）：
```python
on_run_dir: Callable[[Path], None]
# 参数：刚完成的运行目录路径
```

**用途**：
- UI 实时显示生成的图片、动画
- 自动打开输出目录
- 文件系统监控和同步

**设计价值**：
- 解耦核心逻辑和用户界面
- 支持多种 UI 实现（CLI、Web、桌面应用）
- 提供扩展点满足定制需求

### 5.11 调度器设计原则

**1. 串行优先**：
- 简化错误处理和资源管理
- 降低内存使用和复杂度
- 未来可扩展并行执行（基于 `depends_on`）

**2. 隔离运行**：
- 每次运行创建独立目录
- 避免文件冲突和数据污染
- 便于调试和结果分析

**3. 容错设计**：
- 单任务失败不影响其他任务
- 详细错误信息便于调试
- 支持部分成功的批量实验

**4. 配置驱动**：
- 通过配置控制所有行为
- 无需修改代码即可调整执行流程
- 支持动态插件加载

**5. 可扩展**：
- 插件机制支持新功能扩展
- 回调机制支持 UI 集成
- 快照机制支持可重现性

**6. 解耦设计**：
- 调度器与资源包解耦（通过 Engine 接口）
- 调度器与 UI 解耦（通过回调机制）
- 调度器与存储解耦（通过插件系统）
