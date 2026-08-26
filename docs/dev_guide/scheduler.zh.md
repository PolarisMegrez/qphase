---
description: 调度器系统
---

# 调度器系统

scheduler 是 QPhase 的逻辑工作流编排器，负责解析 engine 与插件、校验 DAG、在
job 间传递结果、建立可复现运行目录、报告进度，并把持久化交给 artifact store。
它不把参数点展开成 job，也不决定 engine 的数值并行策略。

## 逻辑 Job 生命周期

scheduler 按拓扑顺序对每个 job 执行：

1. 解析 engine，并校验其 `EngineManifest`。
2. 通过 registry 实例化必需与可选插件。
3. 将可选 `ScanSpec` 编译为 `ParameterGrid`。
4. 解析结构化上游输入。
5. 创建 `ExecutionContext` 并调用 engine。
6. 根据配置通过 `ArtifactStore` 保存最终逻辑结果。
7. 在 `session_manifest.json` 中记录一个状态条目。

shape 为 `(101, 101)` 的 Cartesian scan 在 scheduler 中仍然只有一个 job、一个
配置快照、一个 manifest 条目和一个目录。engine 内部可以处理 10,201 个 point、
tile、chunk、进程或融合 GPU 工作，但这些不是 scheduler 节点。

## Engine Manifest

engine 在执行前声明插件要求：

```python
from qphase.core.protocols import EngineManifest

class MyEngine:
    manifest = EngineManifest(
        required_plugins={"backend", "model"},
        optional_plugins={"analyser"},
    )
```

scheduler 根据校验后的配置实例化插件并传给 engine；资源包仍负责判断哪些插件
组合在数值上有意义。

## ExecutionContext

scheduler 提供的 context 包含：

- `parameter_grid`：编译后的 `ParameterGrid`，无 scan 时为 `None`。
- `resources`：工作站资源提示快照。
- `progress`：供 engine 报告内部进度的接口。
- `cancellation`：预留给 CLI/service 客户端的取消令牌。
- `artifacts`：当前逻辑 job 的 `ArtifactStore`。
- `checkpoints`：chunk 级 `CheckpointStore`。

推荐的 engine 签名为：

```python
def run(self, input_data=None, *, context=None):
    ...
```

旧 `progress_cb` 参数保留一个兼容周期，但 scheduler 自身始终使用
`ExecutionContext`。

## 扫描职责

core 提供 `ParameterGrid` 和可复用的 `execute_pointwise()`。engine 决定调用该
helper，还是把 grid 编译为专用策略。算法层 batching 不能只根据 backend 名称
正确决定，因此不会由 core 提供通用规划器。

典型分工如下：

- CAM `multistability` 管理多进程 tile。
- CAM `batched_newton` 管理 NumPy/CuPy 批量 Newton 数组。
- SDE 把 grid 转换为现有的逐 trajectory 参数重复与 trajectory fusion 表示。
- 简单 engine 可以直接调用 `execute_pointwise()`，复用 chunk checkpoint。

此路径中不再存在由 core 调度的参数点展开、batch negotiation、资源包 batch
planner 或 scheduler result splitter。

## 结构化数据流

上游输入写为：

```yaml
input:
  from: simulation
  mode: dataset
```

`dataset` 将完整结果传入一次。`map` 根据 `select` 与 `group_by` 惰性迭代
point/group view，逐 view 调用下游 engine，最终包装为一个
`MappedDatasetResult`。map 迭代不会获得单独运行目录或 manifest 条目。

## Session 与持久化

```text
runs/<session-id>/
  session_manifest.json
  scan_job/
    config_snapshot.json
    artifact_manifest.json
    result.npz                 # single layout
    # 或 result/shard_*.npz    # sharded layout
    .checkpoints/              # 失败时保留，或由配置要求保留
```

`artifact_manifest.json` 记录 result 类型、schema、axes、shape、物理布局、文件与
loader。`per_point` 只作为外部兼容布局，所有文件仍位于同一个逻辑 job 目录。

checkpoint 兼容性绑定配置 hash、插件版本、backend 与 dtype。最终 dataset 成功
保存后，除非启用 `keep_on_success`，否则清理 checkpoint。当前 checkpoint 只覆盖
已完成 scan chunk，不覆盖 SDE 积分器的中间时间状态。

## Plan 与进度

正常 CLI 只显示逻辑 job、scan shape、状态和结果目录。轴与 chunk 细节只在
`--plan` 或 verbose 输出中展示。service DTO 已提供 scan summary 与内部进度事件，
供未来客户端使用，本轮不要求 GUI 改动。

engine 按 stage 报告 `completed`、`total` 和自然工作单位。core 经过短暂
warm-up 后，仅在当前 `(stage, unit)` 范围内估算速率和剩余时间；stage 切换会
重置估算器。总量未知时只显示 elapsed，不会根据异构 job 外推 workflow ETA；
workflow 层只显示已完成逻辑 job 数。

## Phase 2 执行边界

Scheduler 执行 `CompiledWorkflow.topological_order`，而不是直接按 YAML 中的 Job
排列。`input` 与 `depends_on` 都会参与依赖失败传播；上游失败或被跳过时，下游
Job 标记为 `skipped_dependency`，不会启动 Engine。

单 Job 的插件构造、Engine 调用、上下文、map view、快照和错误转换集中在
`qphase.core.job_runner`。Scheduler 保留逻辑图、取消和终态管理。计划提交到异步
队列前会保存 resolved compiled request，恢复时不重新读取 Project defaults。

Artifact 引用的 materialize 必须显式提供 resolver；执行上下文提供当前 Project 的
resolver。本地插件发现和导入不会永久修改 `sys.path`，不同 Project 的同名模块不会
通过控制进程的全局模块表互相覆盖。

对于 `input.mode=map`，scheduler 只按已完成 view 计数。子 engine 的进度作为
map stage 的 verbose status 暴露，避免大型 dataset 为每个 point 产生独立终端流。
