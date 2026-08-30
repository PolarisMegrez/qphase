---
description: QPhase 2 架构
---

# 架构

QPhase 是面向 Project 的科学工作流运行时。core 负责可迁移 Project 上下文、
Workflow 校验、逻辑 Job 编排、Execution 控制、Session 记录、进度/日志和 Artifact
持久化；资源包负责领域算法及其 CPU/GPU 执行策略。

```text
CLI / GUI / Python client
          |
qphase.service
          |
qphase.core: Project -> Workflow -> Execution -> Session -> Artifact
          |
resource-package Engine
          |
model / backend / solver / analyser / postprocessor plugins
```

CLI 与 GUI 是同一服务层的平级客户端。CLI 始终提供完整接口；GUI 只增加可视化交互，
不引入第二套执行语义。

## 核心对象

- `ProjectContext` 解析严格的 `qphase.project/2` manifest 和 Project 相对路径；
- `WorkflowSpec` 表示严格的 `qphase.workflow/2` 文档；
- `ExecutionManager` 管理本地异步队列与协作式控制；
- `Scheduler` 执行一个 Workflow 图并持久化一个 Session；
- `ArtifactStore` 保存逻辑结果并描述其物理布局；
- `ProjectService` 索引 Workflow 文档与 Session 历史。

Project、Workflow、Execution、Session 与 Artifact 使用稳定 ID。路径只是位置，不得
成为跨客户端身份。

## Engine 与插件

Engine 是资源包面向 scheduler 的唯一入口，通过 `EngineManifest` 声明插件槽位，
scheduler 校验并注入插件实例。Engine 不是 Workflow；一个 Workflow 可以连接多个
使用不同 Engine 的 Job。

插件拥有严格 Pydantic schema 和 capability protocol。子插件槽位可以表达 PSD
estimator 等内部策略族，而不需要把它们拆成无关的顶层插件类。

core 不根据 backend 名称猜测科学并行策略。`ScanSpec` 编译为 `ParameterGrid` 后，
Engine 自行选择逐点、tile、融合、进程或 GPU 策略。101 x 101 scan 仍然是一个逻辑
Job 和一个 Dataset Artifact。

## 配置所有权

| 所有者 | 内容 |
| --- | --- |
| `qphase.toml` | Project 身份与 Workflow/default/plugin/Session 相对路径。 |
| `configs/defaults.yaml` | Project 范围插件默认值。 |
| Workflow 文档 | 科学意图、Job 图、scan 和数据流。 |
| `SystemConfig` | 与 Project 无关的机器策略、进度/日志、存储策略和资源提示。 |
| 插件 schema | 插件专用字段和校验。 |
| Engine manifest | 必需和可选插件命名空间。 |

SystemConfig 不得包含 Project 路径。动态硬件事实采样为 `ResourceSnapshot`，不作为
持久化配置真值。

## 执行生命周期

1. 发现 Project 并加载 `qphase.toml`；
2. 发现包 entry point 与 Project 本地插件；
3. 按稳定 ID 或 Project 相对路径解析 Workflow；
4. 校验 Workflow schema、Job 图、Engine manifest 和插件 schema；
5. 创建 Execution，scheduler 为本次尝试建立一个 Session；
6. 对每个 Job 解析输入和插件、编译 `ParameterGrid`、构造 `ExecutionContext`；
7. 对该逻辑 Job 调用一次资源包 Engine；
8. 持久化快照、事件、日志、Artifact 和 manifest 状态。

暂停/修订发生在 Job 边界；取消由 Engine 在支持的位置检查 token。core 当前不强制
终止 GPU kernel，也不负责多个 Execution 的共享资源调度。

## Session 布局

```text
runs/YYYY/MM/<session-id>/
  session_manifest.json
  events.jsonl
  qphase.log
  <job-name>/
    config_snapshot.json
    artifact_manifest.json
    00_<product>.npz               # single layout: one file per product
    00_<product>__<var>__0000.npz  # sharded layout: chunk files
```

`single`、`sharded` 或兼容性 `per_point` 只改变物理布局，不改变逻辑 Dataset shape。
参数点不会获得独立 Session 或 Job 目录。

可复用的生命周期与基础设施行为属于 core；科学决策、内存模型、batching、融合 kernel
和领域后处理属于资源包。只有至少两个资源包需要同一个领域无关契约时，才应把能力
加入 core。

## Phase 2 核心边界

`core/job_runner.py` 是单个逻辑 Job 的唯一执行边界：它负责解析后的插件实例化、
`ExecutionContext`、Engine 调用、map view、进度适配、快照和 Job 级错误转换。
Scheduler 只负责 compiled DAG 的拓扑顺序、依赖传播、取消和终态汇总。

ExecutionManager 将低频生命周期记录保存到 Project 内的 `.qphase/executions`。
重启时 queued 记录重新入队；进程停止时处于 running 或已开始 paused 的记录标记为
interrupted failure，不伪造数值积分状态恢复。

Artifact-backed Dataset 必须由调用方显式传入 `ArtifactResolver`。Project 运行路径
使用 Project-scoped resolver，不依赖进程级默认绑定。当前本地插件由应用进程在编译
前发现并导入。QPhase 尚未提供独立 worker 或同进程多 Project 模块隔离，这两项能力
留待 Global Phase 6；在此之前，一个应用进程应只使用一个 Project 的本地插件集合。
