---
layout: default
title: 5 调度器 (Scheduler) - 任务执行与生命周期管理
---

# 5 调度器 (Scheduler) - 任务执行与生命周期管理

### 5.0 设计目标与架构

调度器是 qphase 核心的"执行管理层"，负责将 JobList 中的任务转换为实际的计算执行。

**核心职责**：
- **任务编排**：将 JobList 转换为可执行的计算流程
- **依赖管理**：处理任务间的输入/输出传递
- **插件协调**：构建并协调 Backend、Integrator 等插件
- **运行隔离**：为每次运行创建独立的目录和环境
- **容错处理**：单个任务失败不影响其他任务

**设计权衡**：
- **串行 vs 并行**：当前版本采用串行执行，简化错误处理和资源管理
- **隔离 vs 共享**：每次运行创建独立目录，避免相互干扰
- **容错 vs 快速失败**：单个任务失败后继续执行其他任务

### 5.1 JobResult - 任务结果封装

**功能**：封装单个任务的执行结果，包括元数据、运行目录和状态信息。

**结构**：
```python
@dataclass
class JobResult:
    job_index: int           # 任务在列表中的位置（从0开始）
    job_name: str            # 任务名称（用于日志和调试）
    run_dir: Path            # 运行目录路径
    run_id: str              # 运行唯一标识
    success: bool            # 执行是否成功
    error: str | None = None # 错误信息（失败时）
```

**字段设计**：

1. **执行位置**（`job_index`）：
   - 反映任务在 JobList 中的顺序
   - 用于进度显示和日志记录
   - 不依赖任务名称（名称可能重复）

2. **运行标识**（`run_id`）：
   - 格式：`"2024-01-01T12-00-00Z_abc123ef"`
   - 包含 UTC 时间戳（避免时区问题）
   - 包含 UUID 前缀（避免文件名冲突）

3. **运行目录**（`run_dir`）：
   - 任务的所有输出保存在此目录
   - 便于隔离不同运行的输出
   - 支持任务级别的输出目录覆盖

4. **状态信息**（`success` + `error`）：
   - 区分成功和失败两种情况
   - 错误信息便于调试和重试
   - 为上层调度逻辑提供决策依据

### 5.2 串行执行模式 - run() 方法

**核心流程**：
```python
def run(self, job_list: JobList) -> list[JobResult]:
    # 0. 参数扫描预处理：展开包含列表参数的作业
    job_list = self._expand_parameter_scans(job_list)

    results: list[JobResult] = []
    job_outputs: dict[str, Any] = {}  # 任务输出缓存

    for job_idx, job in enumerate(job_list.jobs):
        # 1. 解析任务输入（内存/文件/loader）
        input_data = self._resolve_input(job, job_outputs)

        # 2. 执行任务（构建插件、创建引擎、执行计算）
        result, output = self._run_job(job, job_idx, len(job_list.jobs), input_data)
        results.append(result)

        # 3. 缓存成功任务的输出（供后续任务使用）
        if result.success:
            job_outputs[job.name] = output

    return results
```

**第 0 步：参数扫描预处理**：
```python
job_list = self._expand_parameter_scans(job_list)
```
在正式执行前，调度器会自动展开包含列表参数的作业：
- 检测所有作业中的列表值参数（仅支持插件特定路径）
- 根据配置的扩展方法（cartesian 或 zipped）生成作业组合
- 为每个组合创建独立的作业实例（自动编号）
- 如果参数扫描已禁用，直接返回原始作业列表

**设计特点**：

1. **串行执行**：
   - 优点：内存使用可预测、错误隔离简单、无需锁机制
   - 缺点：无法利用多核 CPU、总体执行时间长
   - 未来可能支持并行执行（`depends_on` 字段已预留）

2. **任务输出缓存**（`job_outputs`）：
   - 键：任务名称（唯一标识）
   - 值：任务的输出对象（内存传递）
   - 作用：实现任务间的数据依赖传递

3. **容错设计**：
   - 单个任务异常不影响其他任务
   - 异常被捕获并记录在 JobResult.error 中
   - 失败任务的输出不加入缓存（避免污染下游任务）

4. **进度回调**（可选）：
   - `on_progress` 回调：实时显示执行进度
   - `on_run_dir` 回调：每个任务完成后调用（用于 UI 更新）

### 5.3 任务输入解析 - _resolve_input() 方法

**功能**：根据 JobConfig.input 字段解析任务输入数据，支持三种输入模式。

**三种输入模式**：

**1. 无输入**（`job.input` 为 None）：
```python
def _resolve_input(job: JobConfig, job_outputs: Dict[str, Any]) -> Any:
    if not job.input:
        return None  # 首次运行或独立任务
```

**2. 内存传递**（上游任务输出）：
```python
if job.input in job_outputs:
    return job_outputs[job.input]  # 直接使用缓存的输出对象
```

**3. 文件或 Loader 加载**：
```python
# 检查是否为外部文件
input_path = Path(job.input)
if input_path.exists():
    if job.input_loader:
        # 使用插件加载器（支持自定义格式）
        loader = registry.create(f"loader:{job.input_loader}")
        return loader.load(job.input)
    else:
        # 强制要求指定 Loader，避免歧义
        raise QPhaseConfigError(f"Job '{job.name}' has file input but no input_loader")
```

**设计灵活性**：

1. **即插即用 Loader**：
   - Loader 是插件，可在 Registry 中注册
   - 支持 CSV、HDF5、自定义格式等
   - 通过 `job.input_loader` 指定使用的 Loader

2. **透明传递**：
   - 上游任务输出对象直接传递给下游
   - 避免不必要的序列化/反序列化
   - 支持复杂对象（numpy 数组、PyTorch 张量等）

3. **严格校验**：
   - 明确区分"任务引用"和"文件路径"
   - 文件输入必须指定 Loader，防止隐式错误
   - 输入不存在时抛出明确的配置错误

### 5.4 任务执行流程 - _run_job() 方法

**执行流程**（6 步）：

**第 1 步：生成运行 ID 和目录**：
```python
run_id = self._generate_run_id()
run_dir = self._create_run_dir(job, run_id)
```

**第 2 步：确定系统配置**：
```python
system_cfg = job.system if job.system is not None else self.system_config
# 任务级配置覆盖全局配置
```

**第 3 步：合并配置**：
```python
job_override = {
    "plugins": job.plugins,  # 插件选择和参数
    "engine": job.engine,    # 引擎配置
    "params": job.params     # 任务参数
}
merged_config = get_config_for_job(system_cfg, job_name=job.name, job_config_dict=job_override)
```

**第 4 步：构建插件实例**：
```python
plugins = self._build_plugins(merged_config.get("plugins", {}))
```

**第 5 步：实例化引擎**：
```python
# 处理引擎配置的嵌套结构 {"engine_name": {params...}}
engine_config_dict = merged_config.get("engine", {})
if engine_config_dict:
    engine_name = list(engine_config_dict.keys())[0]
    engine_config_raw = engine_config_dict[engine_name].copy()
    engine_config_raw["name"] = engine_name  # 注入 name 字段
else:
    # 回退逻辑
    engine_name = job.get_engine_name()
    engine_config_raw = job.engine.get(engine_name, {}).copy()
    engine_config_raw["name"] = engine_name

engine = registry.create_plugin_instance("engine", engine_config_raw, plugins=plugins)
```

**第 6 步：执行引擎**：
```python
output = engine.run(data=input_data)
```

### 5.5 插件构建机制 - _build_plugins() 方法

**功能**：根据插件配置字典构建插件实例，统一处理不同类型的插件。

**构建流程**：
```python
def _build_plugins(self, plugins_config: dict[str, Any]) -> dict[str, Any]:
    plugins: dict[str, Any] = {}

    for plugin_type, config_data in plugins_config.items():
        if not config_data:
            continue  # 跳过空配置

        # config_data: {name: "...", params: {...}}
        instance = registry.create_plugin_instance(plugin_type, config_data)
        plugins[plugin_type] = instance

    return plugins
```

**设计优势**：

1. **统一接口**：
   - 所有插件使用相同的构建方式
   - 减少调度器对插件类型的感知
   - 支持动态发现的新插件类型

2. **配置驱动**：
   - 通过配置控制插件选择和参数
   - 无需修改代码即可更换插件
   - 支持 A/B 测试和参数扫描

3. **错误隔离**：
   - 单个插件构建失败不影响其他插件
   - 详细错误信息便于定位问题
   - 使用 QPhasePluginError 统一异常类型

**常见插件类型**：
- `backend`：计算后端（NumPy、PyTorch 等）
- `integrator`：数值积分器（Euler、Milstein 等）
- `state`：状态管理（NumPy、CuPy 等）
- `noise`：噪声模型（Gaussian、Poisson 等）

### 5.6 引擎实例化与执行

**引擎特殊性**：
引擎是特殊的插件，需要额外的 `plugins` 参数：

```python
engine = registry.create_plugin_instance(
    "engine",
    engine_config_raw,
    plugins=plugins  # 引擎管理其他插件
)
```

**为什么引擎需要 plugins 参数？**

1. **管理职责**：
   - Engine 需要协调 Backend、Integrator 等组件
   - plugins 字典提供组件访问接口
   - 例如：`self.backend = plugins.get("backend")`

2. **生命周期管理**：
   - Engine 负责插件的初始化和销毁
   - 插件共享同一资源（如 GPU 设备）
   - 确保插件间的兼容性

3. **配置传递**：
   - 插件配置通过 plugins 传递
   - 引擎可检查插件配置的有效性
   - 支持插件间的参数依赖

**执行接口**：
```python
output = engine.run(data=input_data)
```

**输入**：
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
